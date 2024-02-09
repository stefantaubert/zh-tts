import os
import re
import zipfile
from logging import getLogger
from pathlib import Path
from typing import Dict

import torch
import wget
from pronunciation_dictionary import (DeserializationOptions, MultiprocessingOptions,
                                      PronunciationDict, load_dict)
from tacotron import CheckpointDict
from tacotron_cli import load_checkpoint
from tqdm import tqdm
from waveglow import CheckpointWaveglow, convert_glow_files
from waveglow_cli import download_pretrained_model

from zh_tts.io import load_obj, save_obj

SPEAKER_DICT_ZIP = "https://zenodo.org/records/7528596/files/pronunciations-narrow-speakers.zip"
SPEAKERS_DICT = "https://zenodo.org/records/7528596/files/pronunciations-narrow.dict"
TACO_CKP = "https://zenodo.org/records/10209990/files/103500.pt"


def get_dicts(conf_dir: Path, silent: bool) -> Dict[str, PronunciationDict]:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
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
      for file in tqdm(os.listdir(dicts_dir_path), desc="Initial loading of speaker dictionaries", disable=silent):
        if file.endswith(".dict"):
          speaker_dict_path = dicts_dir_path / file
          match = speaker_name_pattern.match(file)
          assert match is not None
          speaker_name = match.group(1)
          speaker_dict = load_dict(speaker_dict_path, "UTF-8", DeserializationOptions(
            False, False, False, True), MultiprocessingOptions(1, None, 1_000_000))
          result[speaker_name] = speaker_dict
    save_obj(result, dict_pkl_path)
  else:
    logger.info("Loading dictionaries...")
    result = load_obj(dict_pkl_path)
  return result


def get_wg_model(conf_dir: Path, device: torch.device) -> CheckpointWaveglow:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  wg_path = conf_dir / "waveglow.pt"
  wg_orig_path = conf_dir / "waveglow_orig.pt"

  if not wg_path.is_file():
    logger.info("Downloading WaveGlow checkpoint ...")
    download_pretrained_model(wg_orig_path, version=3)
    wg_checkpoint = convert_glow_files(wg_orig_path, wg_path, device, keep_orig=False)
    # wget.download(WG_CKP, str(wg_path.absolute()))
  else:
    logger.info("Loading WaveGlow checkpoint ...")  # from: {wg_path.absolute()} ...")
    wg_checkpoint = CheckpointWaveglow.load(wg_path, device)
  return wg_checkpoint


def get_taco_model(conf_dir: Path, device: torch.device) -> CheckpointDict:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  taco_path = conf_dir / "tacotron.pt"

  if not taco_path.is_file():
    logger.info("Downloading Tacotron checkpoint ...")
    wget.download(TACO_CKP, str(taco_path.absolute()))

  logger.info("Loading Tacotron checkpoint ...")  # from: {taco_path.absolute()} ...")
  checkpoint = load_checkpoint(taco_path, device)
  return checkpoint
