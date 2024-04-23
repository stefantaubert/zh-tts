import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from ordered_set import OrderedSet
from pronunciation_dictionary import PronunciationDict, SerializationOptions, save_dict

from zho_tts.helper import get_default_device, normalize_audio
from zho_tts.io import save_audio
from zho_tts.synthesizer import Synthesizer
from zho_tts.transcriber import Transcriber
from zho_tts_app.globals import get_conf_dir, get_log_path, get_work_dir
from zho_tts_app.logging_configuration import get_app_logger, get_file_logger, log_sysinfo

CACHE_TRANSCRIBER = "transcriber"
CACHE_SYNTHESIZER = "synthesizer"


def ensure_conf_dir_exists():
  conf_dir = get_conf_dir()
  if not conf_dir.is_dir():
    logger = get_app_logger()
    logger.debug("Creating configuration directory ...")
    conf_dir.mkdir(parents=False, exist_ok=False)


def reset_work_dir():
  logger = get_app_logger()
  work_dir = get_work_dir()

  if work_dir.is_dir():
    logger.debug("Deleting working directory ...")
    shutil.rmtree(work_dir)
  logger.debug("Creating working directory ...")
  work_dir.mkdir(parents=False, exist_ok=False)


def reset_log() -> None:
  get_log_path().write_text("", "utf-8")


def load_models_to_cache(custom_device: Optional[torch.device] = None) -> Dict:
  cli_logger = get_app_logger()
  cache: Dict[str, Union[Synthesizer, Transcriber]] = {}

  ensure_conf_dir_exists()
  conf_dir = get_conf_dir()

  cli_logger.info("Initializing Transcriber ...")
  cache[CACHE_TRANSCRIBER] = Transcriber(conf_dir)

  if custom_device is None:
    custom_device = get_default_device()

  cli_logger.info("Initializing Synthesizer ...")
  cache[CACHE_SYNTHESIZER] = Synthesizer(conf_dir, custom_device)
  return cache


def synthesize_ipa(text_ipa: str, cache: Dict, *, speaker: str = "D6", max_decoder_steps: int = 5000, sigma: float = 1.0, denoiser_strength: float = 0.0005, seed: int = 0, silence_sentences: float = 0.2, silence_paragraphs: float = 1.0, loglevel: int = 2, custom_output: Optional[Path] = None):
  cli_logger = get_app_logger()
  reset_work_dir()
  log_sysinfo()

  if loglevel == 0:
    cli_logger.setLevel(logging.WARNING)

  if loglevel >= 1:
    try_log_text(text_ipa, "text")

  if custom_output is None:
    custom_output = get_work_dir() / "output.wav"

  output_path = synthesize_ipa_core(
    text_ipa, cache[CACHE_SYNTHESIZER], custom_output, speaker=speaker, max_decoder_steps=max_decoder_steps, sigma=sigma, denoiser_strength=denoiser_strength, seed=seed, silence_sentences=silence_sentences, silence_paragraphs=silence_paragraphs, loglevel=loglevel,
  )

  return output_path


def synthesize_zho(text: str, cache: Dict, *, speaker: str = "D6", max_decoder_steps: int = 5000, sigma: float = 1.0, denoiser_strength: float = 0.0005, seed: int = 0, silence_sentences: float = 0.2, silence_paragraphs: float = 1.0, loglevel: int = 2, skip_normalization: bool = False, skip_word_segmentation: bool = False, skip_sentence_separation: bool = False, custom_output: Optional[Path] = None) -> Path:
  cli_logger = get_app_logger()
  reset_work_dir()
  log_sysinfo()

  if loglevel == 0:
    cli_logger.setLevel(logging.WARNING)

  if loglevel >= 1:
    try_log_text(text, "text")

  if custom_output is None:
    custom_output = get_work_dir() / "output.wav"

  text_ipa = convert_zho_to_ipa(
    text, cache[CACHE_TRANSCRIBER],
    loglevel=loglevel,
    speaker=speaker,
    skip_normalization=skip_normalization,
    skip_word_segmentation=skip_word_segmentation,
    skip_sentence_separation=skip_sentence_separation,
  )

  output_path = synthesize_ipa_core(
    text_ipa, cache[CACHE_SYNTHESIZER], custom_output,
    max_decoder_steps=max_decoder_steps, sigma=sigma, denoiser_strength=denoiser_strength,
    seed=seed, silence_sentences=silence_sentences, silence_paragraphs=silence_paragraphs, loglevel=loglevel
  )

  return output_path


def convert_zho_to_ipa(text: str, transcriber: Transcriber, *, speaker: str = "D6", loglevel: int = 1, skip_normalization: bool = False, skip_word_segmentation: bool = False, skip_sentence_separation: bool = False) -> str:
  t = transcriber
  text_ipa = t.transcribe_to_ipa(text, speaker, skip_normalization,
                                 skip_word_segmentation, skip_sentence_separation)

  if loglevel >= 1:
    for txt, name in (
      (t.text_normed, "text.normed"),
      (t.text_segmented, "text.segmented"),
      (t.text_sentenced, "text.sentenced"),
      (t.text_ipa, "text.ipa"),
      (t.text_ipa_readable, "text.ipa.readable"),
      (text_ipa, "text.ipa.silenced"),
    ):
      try_log_text(txt, name)

    for v, name in (
      (t.vocabulary, "vocabulary"),
      (t.oov1, "oov1"),
      (t.oov2, "oov2"),
      (t.oov3, "oov3"),
      (t.oov4, "oov4"),
      (t.oov5, "oov5"),
    ):
      try_log_voc(v, name)

    for d, name in (
      (t.dict1, "dict1"),
      (t.dict1_single, "dict1.single"),
      (t.dict2, "dict2"),
      (t.dict2_single, "dict2.single"),
      (t.dict1_2, "dict1+2"),
      (t.dict3, "dict3"),
      (t.dict3_single, "dict3.single"),
      (t.dict1_2_3, "dict1+2+3"),
      (t.dict4, "dict4"),
      (t.dict4_single, "dict4.single"),
      (t.dict1_2_3_4, "dict1+2+3+4"),
      (t.dict5, "dict5"),
      (t.dict5_pinyin, "dict5.pinyin"),
      (t.dict5_single, "dict5.pinyin.single"),
      (t.dict1_2_3_4_5, "dict1+2+3+4+5"),
    ):
      try_log_dict(d, name)
  return text_ipa


def synthesize_ipa_core(text_ipa: str, synthesizer: Synthesizer, output: Path, *, speaker: str = "D6", max_decoder_steps: int = 5000, sigma: float = 1.0, denoiser_strength: float = 0.0005, seed: int = 0, silence_sentences: float = 0.4, silence_paragraphs: float = 1.0, loglevel: int = 1) -> Path:
  logger = logging.getLogger(__name__)
  work_dir = get_work_dir()

  audio = synthesizer.synthesize(text_ipa, speaker, max_decoder_steps, seed, sigma,
                                 denoiser_strength, silence_sentences, silence_paragraphs, silent=loglevel == 0)
  unnormed_out = work_dir / "output.unnormed.wav"
  save_audio(audio, unnormed_out)
  work_dir_output = unnormed_out

  try:
    normalize_audio(unnormed_out, work_dir / "output.wav")
    work_dir_output = work_dir / "output.wav"
  except Exception as error:
    logger.warning("Normalization was not possible!", exc_info=error, stack_info=True)
    logger.info(f"Saved audio to: '{unnormed_out.absolute()}'")

  if output != work_dir_output:
    try:
      output.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(work_dir_output, output)
    except Exception as ex:
      logger.exception(
        f"Output couldn't be created at: '{output.absolute()}'", exc_info=ex, stack_info=True)
      logger.info(f"Saved audio to: '{work_dir_output.absolute()}'")
  logger.info(f"Saved audio to: '{output.absolute()}'")
  return output


def try_log_dict(dictionary: Optional[PronunciationDict], name: str) -> None:
  if dictionary:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.dict"
    save_dict(dictionary, logfile, "utf-8", SerializationOptions("TAB", False, True))
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")


def try_log_voc(vocabulary: Optional[OrderedSet[str]], name: str) -> None:
  if vocabulary:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.txt"
    logfile.write_text("\n".join(vocabulary), "utf-8")
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")


def try_log_text(text: Optional[str], name: str) -> None:
  if text:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.txt"
    logfile.write_text(text, "utf-8")
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")
