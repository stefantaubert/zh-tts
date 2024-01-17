
import os
import pickle
import re
import shutil
import zipfile
from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, Dict, Generator, Literal, cast

import numpy as np
import pkuseg
import torch
import wget
import zhon
from dict_from_dict import create_dict_from_dict
from dict_from_pypinyin import convert_chinese_to_pinyin
from ffmpy import FFmpeg
from ordered_set import OrderedSet
from pinyin_to_ipa import pinyin_to_ipa
from pronunciation_dictionary import (DeserializationOptions, MultiprocessingOptions,
                                      PronunciationDict, SerializationOptions, get_phoneme_set,
                                      load_dict, save_dict)
from pronunciation_dictionary_utils import merge_dictionaries, select_single_pronunciation
from pronunciation_dictionary_utils_cli.pronunciations_map_symbols_json import \
  identify_and_apply_mappings
from pypinyin import Style
from tacotron import Synthesizer as TacotronSynthesizer
from tacotron_cli import *
from tqdm import tqdm
from txt_utils_cli import extract_vocabulary_from_text
from txt_utils_cli.replacement import replace_text
from txt_utils_cli.transcription import transcribe_text_using_dict
from waveglow import CheckpointWaveglow
from waveglow import Synthesizer as WaveglowSynthesizer
from waveglow import convert_glow_files, float_to_wav, normalize_wav, try_copy_to
from waveglow_cli import download_pretrained_model

from zh_tts_cli.cn_tn import TextNorm
from zh_tts_cli.types import ExecutionResult

SPEAKER_DICT_ZIP = "https://zenodo.org/records/7528596/files/pronunciations-narrow-speakers.zip"
SPEAKERS_DICT = "https://zenodo.org/records/7528596/files/pronunciations-narrow.dict"
TACO_CKP = "https://zenodo.org/records/10209990/files/103500.pt"


def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda:0")
  return torch.device("cpu")


def init_synthesize_eng_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "Command-line interface for synthesizing Chinese texts into speech."
  parser.add_argument("input", type=str, metavar="INPUT",
                      help="text input")
  return synthesize_ns


def synthesize_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  text = cast(str, ns.input)
  synthesize(text, "Chinese", logger, flogger)


def normalize_chn_text(text: str) -> str:
  normalize = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="t2s",
  )

  result = normalize(text)
  result = remove_unallowed(result)
  result = re.sub(r":", r"：", result)
  result = re.sub(r";", r"；", result)
  result = re.sub(r"[,､、]", r"，", result)
  result = re.sub(r"[｡．.]", r"。", result)
  # will result in hanzi + ：；，。！？
  return result


def remove_unallowed(text: str) -> str:
  unallowed_chars_pattern = re.compile(
    rf"[^{zhon.hanzi.characters}{re.escape(':;,. ､、：；，｡。！．？')}\n]")
  unallowed_chars = set(unallowed_chars_pattern.findall(text))
  logger = getLogger(__name__)
  logger.debug(f"Removed unallowed characters: {' '.join(sorted(unallowed_chars))}")
  text = unallowed_chars_pattern.sub("", text)
  return text


def segment_words(text: str) -> str:
  seg = pkuseg.pkuseg()
  lines = text.split("\n")
  norm_lines = []
  whitespace_pattern = re.compile(r"\s+")
  punctuation_pattern = re.compile(r" ([：；，。！？])")
  for line in lines:
    line = whitespace_pattern.sub(r"", line)
    result = seg.cut(line)
    result_str: str = " ".join(result)
    result_str = punctuation_pattern.sub(r"\1", result_str)
    norm_lines.append(result_str)
  result_str = "\n".join(norm_lines)
  return result_str


def get_sentences(text: str) -> Generator[str, None, None]:
  pattern = re.compile(r"([。！？]) +")  # [^$]
  sentences = pattern.split(text)
  for i in range(0, len(sentences), 2):
    if i + 1 < len(sentences):
      sentence = sentences[i] + sentences[i + 1]
    else:
      sentence = sentences[i]
    if len(sentence) > 0:
      yield sentence


def get_dicts(conf_dir: Path, logger: Logger) -> Dict[str, PronunciationDict]:
  dict_pkl_path = conf_dir / "dictionaries.pkl"

  if not dict_pkl_path.is_file():
    result = {}
    dict_dl_path = conf_dir / "dictionary.dict"
    logger.info("Downloading speakers narrow dictionary ...")
    wget.download(SPEAKERS_DICT, str(dict_dl_path.absolute()))
    all_dict = load_dict(dict_dl_path, "UTF-8", DeserializationOptions(
      False, False, False, True), MultiprocessingOptions(1, None, 1_000_000))
    result["all"] = all_dict

    dicts_dir_path = conf_dir / "dictionaries"
    dicts_dl_path = conf_dir / "dictionaries.zip"
    logger.info("Downloading speaker narrow dictionaries ...")
    wget.download(SPEAKER_DICT_ZIP, str(dicts_dl_path.absolute()))

    speaker_name_pattern = re.compile(r"pronunciations-narrow-([ABCD]\d+);.+\.dict")
    if not dicts_dir_path.is_dir():
      dicts_dir_path.mkdir(parents=True, exist_ok=False)
      with zipfile.ZipFile(dicts_dl_path, 'r') as zip_ref:
        zip_ref.extractall(dicts_dir_path)
      for file in tqdm(os.listdir(dicts_dir_path), desc="Initial loading of speaker dictionaries"):
        if file.endswith(".dict"):
          speaker_dict_path = dicts_dir_path / file
          speaker_name = speaker_name_pattern.match(file).group(1)
          speaker_dict = load_dict(speaker_dict_path, "UTF-8", DeserializationOptions(
            False, False, False, True), MultiprocessingOptions(1, None, 1_000_000))
          result[speaker_name] = speaker_dict

    save_obj(result, dict_pkl_path)
  else:
    logger.info("Loading dictionaries...")
    result = load_obj(dict_pkl_path)
  return result


def get_wg_model(conf_dir: Path, device: torch.device, logger: Logger):
  wg_path = conf_dir / "waveglow.pt"
  wg_orig_path = conf_dir / "waveglow_orig.pt"

  if not wg_path.is_file():
    logger.info("Downloading Waveglow checkpoint...")
    download_pretrained_model(wg_orig_path, version=3)
    wg_checkpoint = convert_glow_files(wg_orig_path, wg_path, device, keep_orig=False)
    # wget.download(WG_CKP, str(wg_path.absolute()))
  else:
    logger.info("Loading Waveglow checkpoint...")
    wg_checkpoint = CheckpointWaveglow.load(wg_path, device, logger)
  return wg_checkpoint


def get_taco_model(conf_dir: Path, device: torch.device, logger: Logger):
  taco_path = conf_dir / "tacotron.pt"

  if not taco_path.is_file():
    logger.info("Downloading Tacotron checkpoint...")
    wget.download(TACO_CKP, str(taco_path.absolute()))

  logger.info(f"Loading Tacotron checkpoint from: {taco_path.absolute()} ...")
  checkpoint = load_checkpoint(taco_path, device)
  return checkpoint


def convert_chn_to_ipa(text: str, conf_dir: Path, work_dir: Path, logger: Logger, flogger: Logger) -> str:
  serialize_log_opts = SerializationOptions("TAB", False, True)
  loglevel = 1
  n_jobs = 1
  maxtasksperchild = None
  speaker = "D6"

  if loglevel >= 1:
    logfile = work_dir / "text.txt"
    logfile.write_text(text, "utf-8")
    flogger.info(f"Text: {logfile.absolute()}")

  text_normed = normalize_chn_text(text)
  if text_normed == text:
    flogger.info("No normalization applied.")
  else:
    text = text_normed
    flogger.info("Normalization was applied.")
    if loglevel >= 1:
      logfile = work_dir / "text.normed.txt"
      logfile.write_text(text, "utf-8")
      flogger.info(f"Text (normed): {logfile.absolute()}")

  text_segmented = segment_words(text)
  if text_segmented == text:
    flogger.info("No word segmentation applied.")
  else:
    text = text_segmented
    flogger.info("Word segmentation was applied.")
    if loglevel >= 1:
      logfile = work_dir / "text.segmented.txt"
      logfile.write_text(text, "utf-8")
      flogger.info(f"Text (word segmented): {logfile.absolute()}")

  sentences = get_sentences(text)
  text_sentenced = "\n".join(sentences)
  if text == text_sentenced:
    flogger.info("No sentence separation applied.")
  else:
    text = text_sentenced
    flogger.info("Sentence separation was applied.")
    if loglevel >= 1:
      logfile = work_dir / "text.sentences.txt"
      logfile.write_text(text, "utf-8")
      flogger.info(f"Text (sentences): {logfile.absolute()}")

  vocabulary = extract_vocabulary_from_text(
    text, "\n", " ", False, n_jobs, maxtasksperchild, 2_000_000)

  if loglevel >= 1:
    logfile = work_dir / "vocabulary.txt"
    logfile.write_text("\n".join(vocabulary), "utf-8")
    flogger.info(f"Vocabulary: {logfile.absolute()}")

  dicts = get_dicts(conf_dir, logger)
  speaker_dict = dicts[speaker]

  dict1, oov1 = create_dict_from_dict(vocabulary, speaker_dict, trim={
  }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

  if loglevel >= 1:
    logfile = work_dir / "dict1.dict"
    save_dict(dict1, logfile, "utf-8", serialize_log_opts)
    flogger.info(f"Dict1: {logfile.absolute()}")
    if len(oov1) > 0:
      logfile = work_dir / "oov1.txt"
      logfile.write_text("\n".join(oov1), "utf-8")
      flogger.info(f"OOV1: {logfile.absolute()}")

  changed_word_count = select_single_pronunciation(dict1, mode="highest-weight", seed=None,
                                                   mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

  if loglevel >= 1 and changed_word_count > 0:
    logfile = work_dir / "dict1.single.dict"
    save_dict(dict1, logfile, "utf-8", serialize_log_opts)
    flogger.info(f"Dict1 (single pronunciation): {logfile.absolute()}")

  oov2 = OrderedSet()
  if len(oov1) > 0:
    dict2, oov2 = create_dict_from_dict(oov1, speaker_dict, trim=set("：；，。！？"), split_on_hyphen=False,
                                        ignore_case=True, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

    if loglevel >= 1:
      logfile = work_dir / "dict2.dict"
      save_dict(dict2, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1: {logfile.absolute()}")
      if len(oov2) > 0:
        logfile = work_dir / "oov2.txt"
        logfile.write_text("\n".join(oov2), "utf-8")
        flogger.info(f"OOV2: {logfile.absolute()}")

    changed_word_count = select_single_pronunciation(dict2, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

    if loglevel >= 1 and changed_word_count > 0:
      logfile = work_dir / "dict2.single.dict"
      save_dict(dict2, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict2 (single pronunciation): {logfile.absolute()}")

    merge_dictionaries(dict1, dict2, mode="add")

    if loglevel >= 1:
      logfile = work_dir / "dict1+2.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1+2: {logfile.absolute()}")

  all_dict = dicts["all"]
  oov3 = OrderedSet()
  if len(oov2) > 0:
    dict3, oov3 = create_dict_from_dict(oov2, all_dict, trim={
    }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

    if loglevel >= 1:
      logfile = work_dir / "dict3.dict"
      save_dict(dict3, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict3: {logfile.absolute()}")
      if len(oov3) > 0:
        logfile = work_dir / "oov3.txt"
        logfile.write_text("\n".join(oov3), "utf-8")
        flogger.info(f"OOV3: {logfile.absolute()}")

    changed_word_count = select_single_pronunciation(dict3, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

    if loglevel >= 1 and changed_word_count > 0:
      logfile = work_dir / "dict3.single.dict"
      save_dict(dict3, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict3 (single pronunciation): {logfile.absolute()}")

    merge_dictionaries(dict1, dict3, mode="add")

    if loglevel >= 1:
      logfile = work_dir / "dict1+2+3.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1+2+3: {logfile.absolute()}")

  oov4 = OrderedSet()
  if len(oov3) > 0:
    dict4, oov4 = create_dict_from_dict(oov3, all_dict, trim=set("：；，。！？"), split_on_hyphen=False,
                                        ignore_case=True, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

    if loglevel >= 1:
      logfile = work_dir / "dict4.dict"
      save_dict(dict4, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict4: {logfile.absolute()}")
      if len(oov4) > 0:
        logfile = work_dir / "oov4.txt"
        logfile.write_text("\n".join(oov4), "utf-8")
        flogger.info(f"OOV4: {logfile.absolute()}")

    changed_word_count = select_single_pronunciation(dict4, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

    if loglevel >= 1 and changed_word_count > 0:
      logfile = work_dir / "dict4.single.dict"
      save_dict(dict4, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict4 (single pronunciation): {logfile.absolute()}")

    merge_dictionaries(dict1, dict4, mode="add")

    if loglevel >= 1:
      logfile = work_dir / "dict1+2+3+4.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1+2+3+4: {logfile.absolute()}")

  oov5 = OrderedSet()
  if len(oov4) > 0:
    dict5, oov5 = convert_chinese_to_pinyin(
      oov4, Style.TONE3, True, True, True, 1.0, set("：；，。！？"), False, 1, None, 10_000)

    if loglevel >= 1:
      logfile = work_dir / "dict5.pinyin.dict"
      save_dict(dict5, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict5 (Pinyin): {logfile.absolute()}")
      if len(oov5) > 0:
        logfile = work_dir / "oov5.txt"
        logfile.write_text("\n".join(oov5), "utf-8")
        flogger.info(f"OOV5: {logfile.absolute()}")
    
    if len(oov5) > 0:
      logger.warning(f"OOV exist (will be ignored): {' '.join(sorted(oov5))}")
    
    changed_word_count = select_single_pronunciation(dict5, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

    if loglevel >= 1 and changed_word_count > 0:
      logfile = work_dir / "dict5.pinyin.single.dict"
      save_dict(dict5, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict5 (Pinyin, single pronunciation): {logfile.absolute()}")

    pinyins = get_phoneme_set(dict5)
    pinyin_mappings = {
      p: " ".join(pinyin_to_ipa(p)[0]) for p in pinyins
      if p not in set("：；，。！？")
    }
    identify_and_apply_mappings(logger, flogger, dict5, pinyin_mappings, partial_mapping=False,
                                mp_options=MultiprocessingOptions(1, maxtasksperchild, 100_000))

    if loglevel >= 1:
      logfile = work_dir / "dict5.dict"
      save_dict(dict5, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict5: {logfile.absolute()}")

    merge_dictionaries(dict1, dict5, mode="add")

    if loglevel >= 1:
      logfile = work_dir / "dict1+2+3+4+5.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1+2+3+4+5: {logfile.absolute()}")

  text_ipa = transcribe_text_using_dict(dict1, text, "\n", "|", " ", seed=None, ignore_missing=False,
                                        n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=2_000_000)

  if loglevel >= 1:
    logfile = work_dir / "ipa.txt"
    logfile.write_text(text_ipa, "utf-8")
    flogger.info(f"IPA: {logfile.absolute()}")

    logfile = work_dir / "ipa.readable.txt"
    logfile.write_text(text_ipa.replace("|", ""), "utf-8")
    flogger.info(f"IPA (readable): {logfile.absolute()}")

  text_ipa = replace_text(text_ipa, " ", "SIL0", disable_regex=True)
  text_ipa = replace_text(text_ipa, "，|SIL0", "SIL1", disable_regex=True)
  text_ipa = replace_text(text_ipa, r"([。？])", r"\1|SIL2", disable_regex=False)
  text_ipa = replace_text(text_ipa, r"([：；])\|SIL0", r"SIL2", disable_regex=False)
  text_ipa = replace_text(text_ipa, "！", "。|SIL2", disable_regex=True)

  if loglevel >= 1:
    logfile = work_dir / "ipa.silence.txt"
    logfile.write_text(text_ipa, "utf-8")
    flogger.info(f"IPA: {logfile.absolute()}")
  return text_ipa


def synthesize(text: str, input_format: Literal["Chinese", "IPA"], logger: Logger, flogger: Logger):
  conf_dir = Path(gettempdir()) / "zh-tts"
  conf_dir.mkdir(parents=True, exist_ok=True)
  work_dir = conf_dir / "tmp"
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)

  max_decoder_steps = 5000
  sigma = 1.0
  denoiser_strength = 0.0005
  seed = 1
  device = get_device()
  silence_sentences = 0.2
  silence_paragraphs = 1.0
  loglevel = 1

  paragraph_sep = "\n\n"
  sentence_sep = "\n"

  if input_format == "Chinese":
    text_ipa = convert_chn_to_ipa(text, conf_dir, work_dir, logger, flogger)
  elif input_format == "IPA":
    text_ipa = text
    if loglevel >= 1:
      logfile = work_dir / "ipa.txt"
      logfile.write_text(text_ipa, "utf-8")
      flogger.info(f"IPA: {logfile.absolute()}")
  else:
    raise NotImplementedError()

  taco_checkpoint = get_taco_model(conf_dir, device, logger)

  synth = TacotronSynthesizer(
    checkpoint=taco_checkpoint,
    custom_hparams=None,
    device=device,
    logger=logger,
  )

  wg_checkpoint = get_wg_model(conf_dir, device, logger)

  wg_synth = WaveglowSynthesizer(
    checkpoint=wg_checkpoint,
    custom_hparams=None,
    device=device,
    logger=logger,
  )

  first_speaker = "D6"
  resulting_wavs = []
  paragraphs = text_ipa.split(paragraph_sep)
  for paragraph_nr, paragraph in enumerate(tqdm(paragraphs, position=0, desc="Paragraph")):
    sentences = paragraph.split(sentence_sep)
    sentences = [x for x in sentences if x != ""]
    for sentence_nr, sentence in enumerate(tqdm(sentences, position=1, desc="Sentence")):
      sentence_id = f"{paragraph_nr+1}-{sentence_nr+1}"

      symbols = sentence.split("|")
      flogger.info(f"Synthesizing {sentence_id} step 1/2...")
      inf_sent_output = synth.infer(
        symbols=symbols,
        speaker=first_speaker,
        include_stats=False,
        max_decoder_steps=max_decoder_steps,
        seed=seed,
      )

      if loglevel >= 2:
        logfile = work_dir / f"{sentence_id}.npy"
        np.save(logfile, inf_sent_output.mel_outputs_postnet)
        flogger.info(f"Tacotron output: {logfile.absolute()}")

      mel_var = torch.FloatTensor(inf_sent_output.mel_outputs_postnet)
      del inf_sent_output
      mel_var = try_copy_to(mel_var, device)
      mel_var = mel_var.unsqueeze(0)
      flogger.info(f"Synthesizing {sentence_id} step 2/2...")
      inference_result = wg_synth.infer(mel_var, sigma, denoiser_strength, seed)
      wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)
      del mel_var

      if loglevel >= 2:
        logfile = work_dir / f"{sentence_id}.wav"
        float_to_wav(wav_inferred_denoised_normalized, logfile)
        flogger.info(f"WaveGlow output: {logfile.absolute()}")

      resulting_wavs.append(wav_inferred_denoised_normalized)
      is_last_sentence_in_paragraph = sentence_nr == len(sentences) - 1
      if silence_sentences > 0 and not is_last_sentence_in_paragraph:
        pause_samples = np.zeros(
          (get_sample_count(wg_synth.hparams.sampling_rate, silence_sentences),))
        resulting_wavs.append(pause_samples)

    is_last_paragraph = paragraph_nr == len(paragraphs) - 1
    if silence_paragraphs > 0 and not is_last_paragraph:
      pause_samples = np.zeros(
        (get_sample_count(wg_synth.hparams.sampling_rate, silence_paragraphs),))
      resulting_wavs.append(pause_samples)

  if len(resulting_wavs) > 0:
    resulting_wav = np.concatenate(tuple(resulting_wavs), axis=-1)
    float_to_wav(resulting_wav, work_dir / "result.unnormed.wav",
                 sample_rate=wg_synth.hparams.sampling_rate)
    ffmpeg_normalization = FFmpeg(
        inputs={
          str((work_dir / "result.unnormed.wav").absolute()): None
        },
        outputs={
          str((work_dir / "result.wav").absolute()): "-acodec pcm_s16le -ar 22050 -ac 1 -af loudnorm=I=-16:LRA=11:TP=-1.5 -y -hide_banner -loglevel error"
        },
    )
    ffmpeg_normalization.run()
    logger.info(f'Saved to: {work_dir / "result.wav"}')

  return True


def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))


def save_obj(obj: Any, path: Path) -> None:
  assert isinstance(path, Path)
  assert path.parent.exists() and path.parent.is_dir()
  with open(path, mode="wb") as file:
    pickle.dump(obj, file)


def load_obj(path: Path) -> Any:
  assert isinstance(path, Path)
  assert path.is_file()
  with open(path, mode="rb") as file:
    return pickle.load(file)


def remove_urls(text: str) -> str:
  pattern = re.compile(
    r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])")
  result = pattern.sub("U R L", text)
  return result
