import logging
import re
from collections import OrderedDict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Generator, Optional

import spacy_pkuseg
import zhon
from dict_from_dict import create_dict_from_dict
from dict_from_pypinyin import convert_chinese_to_pinyin
from ordered_set import OrderedSet
from pinyin_to_ipa import pinyin_to_ipa
from pronunciation_dictionary import MultiprocessingOptions, PronunciationDict, get_phoneme_set
from pronunciation_dictionary_utils import (map_symbols_dict, merge_dictionaries,
                                            select_single_pronunciation)
from pypinyin import Style
from txt_utils import extract_vocabulary_from_text, replace_text, transcribe_text_using_dict

from zho_tts.cn_tn import TextNorm
from zho_tts.globals import DEFAULT_CONF_DIR, DEFAULT_SPEAKER
from zho_tts.resources import get_dicts


class Transcriber():
  def __init__(
      self,
      conf_dir: Path = DEFAULT_CONF_DIR,
  ) -> None:
    logger = getLogger(__name__)

    tmp_logger = getLogger("txt_utils")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("pinyin_to_ipa")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("dict_from_dict")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("dict_from_pypinyin")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("pronunciation_dictionary")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("pronunciation_dictionary_utils")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    self._conf_dir = conf_dir
    self._dicts = get_dicts(conf_dir, silent=False)
    self._symbol_separator = "|"
    self._punctuation = set("：；，。！？")

    self.text_normed: Optional[str] = None
    self.text_segmented: Optional[str] = None
    self.text_sentenced: Optional[str] = None
    self.vocabulary: OrderedSet[str] = OrderedSet()
    self.dict1: PronunciationDict = OrderedDict()
    self.oov1: Optional[OrderedSet[str]] = None
    self.dict1_single: Optional[PronunciationDict] = None
    self.dict2: Optional[PronunciationDict] = None
    self.dict2_single: Optional[PronunciationDict] = None
    self.oov2: Optional[OrderedSet[str]] = None
    self.dict1_2: Optional[PronunciationDict] = None
    self.dict3: Optional[PronunciationDict] = None
    self.dict3_single: Optional[PronunciationDict] = None
    self.oov3: Optional[OrderedSet[str]] = None
    self.dict1_2_3: Optional[PronunciationDict] = None
    self.dict4: Optional[PronunciationDict] = None
    self.dict4_single: Optional[PronunciationDict] = None
    self.oov4: Optional[OrderedSet[str]] = None
    self.dict1_2_3_4: Optional[PronunciationDict] = None
    self.dict5: Optional[PronunciationDict] = None
    self.dict5_pinyin: Optional[PronunciationDict] = None
    self.dict5_single: Optional[PronunciationDict] = None
    self.oov5: Optional[OrderedSet[str]] = None
    self.dict1_2_3_4_5: Optional[PronunciationDict] = None
    self.text_ipa: str = ""
    self.text_ipa_readable: str = ""

  def _reset_locals(self) -> None:
    self.text_normed = None
    self.text_segmented = None
    self.text_sentenced = None
    self.vocabulary = OrderedSet()
    self.dict1 = OrderedSet()
    self.oov1 = None
    self.dict1_single = None
    self.dict2 = None
    self.dict2_single = None
    self.oov2 = None
    self.dict1_2 = None
    self.dict3 = None
    self.dict3_single = None
    self.oov3 = None
    self.dict1_2_3 = None
    self.dict4 = None
    self.dict4_single = None
    self.oov4 = None
    self.dict1_2_3_4 = None
    self.dict5 = None
    self.dict5_pinyin = None
    self.dict5_single = None
    self.oov5 = None
    self.dict1_2_3_4_5 = None
    self.text_ipa = ""
    self.text_ipa_readable = ""

  def transcribe_to_ipa(self, text: str, speaker: str = DEFAULT_SPEAKER, skip_normalization: bool = False, skip_word_segmentation: bool = False, skip_sentence_segmentation: bool = False) -> str:
    logger = getLogger(__name__)
    self._reset_locals()

    if skip_normalization:
      logger.debug("Normalization was skipped.")
    else:
      logger.info("Normalizing ...")
      text_normed = normalize_chn_text(text)
      if text_normed == text:
        logger.debug("Normalization was not necessary.")
      else:
        self.text_normed = text_normed
        text = text_normed
        logger.debug("Normalization was applied.")

    if skip_word_segmentation:
      logger.debug("Word segmentation was skipped.")
    else:
      logger.info("Segmenting words ...")
      text_segmented = segment_words(text)
      if text_segmented == text:
        logger.debug("Word segmentation was not necessary.")
      else:
        self.text_segmented = text_segmented
        text = text_segmented
        logger.debug("Word segmentation was applied.")

    if skip_sentence_segmentation:
      logger.debug("Sentence segmentation was skipped.")
    else:
      logger.info("Segmenting sentences ...")
      sentences = get_sentences(text)
      text_sentenced = "\n".join(sentences)
      if text == text_sentenced:
        logger.debug("No sentence segmentation applied.")
      else:
        self.text_sentenced = text_sentenced
        text = text_sentenced
        logger.debug("Sentence segmentation was applied.")

    logger.debug("Extracting vocabulary ...")
    vocabulary = extract_vocabulary_from_text(text, silent=True)
    self.vocabulary = vocabulary

    logger.info("Looking up vocabulary ...")
    dict1, oov1 = create_dict_from_dict(vocabulary, self._dicts[speaker], trim={
    }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)

    self.dict1 = deepcopy(dict1)
    if len(oov1) > 0:
      self.oov1 = oov1

    changed_word_count = select_single_pronunciation(dict1, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)

    self.dict1_single = None
    if changed_word_count > 0:
      self.dict1_single = deepcopy(dict1)

    oov2: OrderedSet[str] = OrderedSet()
    if len(oov1) > 0:
      dict2, oov2 = create_dict_from_dict(oov1, self._dicts[speaker], trim=self._punctuation, split_on_hyphen=False,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)
      self.dict2 = deepcopy(dict2)
      if len(oov2) > 0:
        self.oov2 = oov2

      changed_word_count = select_single_pronunciation(dict2, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict2_single = deepcopy(dict2)

      merge_dictionaries(dict1, dict2, mode="add")

      self.dict1_2 = deepcopy(dict1)

    oov3: OrderedSet[str] = OrderedSet()
    if len(oov2) > 0:
      dict3, oov3 = create_dict_from_dict(oov2, self._dicts["all"], trim={}, split_on_hyphen=False,
                                          ignore_case=False, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)
      self.dict3 = deepcopy(dict3)
      if len(oov3) > 0:
        self.oov3 = oov3

      changed_word_count = select_single_pronunciation(dict3, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict3_single = deepcopy(dict3)

      merge_dictionaries(dict1, dict3, mode="add")
      self.dict1_2_3 = deepcopy(dict1)

    oov4: OrderedSet[str] = OrderedSet()
    if len(oov3) > 0:
      dict4, oov4 = create_dict_from_dict(oov3, self._dicts["all"], trim=self._punctuation, split_on_hyphen=False,
                                          ignore_case=False, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)
      self.dict4 = deepcopy(dict4)

      if len(oov4) > 0:
        self.oov4 = oov4

      changed_word_count = select_single_pronunciation(dict4, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict4_single = deepcopy(dict4)

      merge_dictionaries(dict1, dict4, mode="add")

      self.dict1_2_3_4 = deepcopy(dict1)

    if len(oov4) > 0:
      dict5, oov5 = convert_chinese_to_pinyin(
          oov4, Style.TONE3, True, True, True, 1.0, self._punctuation, False, 1, None, 10_000, silent=True)
      self.dict5_pinyin = deepcopy(dict5)

      if len(oov5) > 0:
        self.oov5 = oov5
        logger.warning(f"OOV exist (will be ignored): {' '.join(sorted(oov5))}")

      changed_word_count = select_single_pronunciation(dict5, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict5_single = deepcopy(dict5)

      pinyins = get_phoneme_set(dict5)
      pinyin_mappings = {
        p: " ".join(pinyin_to_ipa(p)[0]) for p in pinyins
        if p not in self._punctuation
      }
      map_symbols_dict(dict5, pinyin_mappings, partial_mapping=False,
                       mp_options=MultiprocessingOptions(1, None, 100_000), silent=True)
      self.dict5 = deepcopy(dict5)

      merge_dictionaries(dict1, dict5, mode="add")

      self.dict1_2_3_4_5 = deepcopy(dict1)

    logger.debug("Transcribing to IPA ...")
    text_ipa = transcribe_text_using_dict(
      text, dict1,
      phoneme_sep=self._symbol_separator, silent=True
    )
    self.text_ipa = text_ipa
    self.text_ipa_readable = text_ipa.replace(self._symbol_separator, "")

    text_ipa = replace_text(text_ipa, " ", "SIL0", disable_regex=True)
    text_ipa = replace_text(text_ipa, f"，{self._symbol_separator}SIL0", "SIL1", disable_regex=True)
    text_ipa = replace_text(
      text_ipa, r"([。？])", rf"\1{self._symbol_separator}SIL2", disable_regex=False)
    text_ipa = replace_text(
      text_ipa, rf"([：；]){re.escape(self._symbol_separator)}SIL0", r"SIL2", disable_regex=False)
    text_ipa = replace_text(text_ipa, "！", f"。{self._symbol_separator}SIL2", disable_regex=True)

    return text_ipa


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

  result: str = normalize(text)
  result = remove_urls(result)
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
  if len(unallowed_chars) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Removed unallowed characters: {' '.join(sorted(unallowed_chars))}")
  text = unallowed_chars_pattern.sub("", text)
  text = re.compile(r"\s{2,}").sub(" ", text)
  return text


def segment_words(text: str) -> str:
  # seg = pkuseg.pkuseg()
  seg = spacy_pkuseg.pkuseg()
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


def remove_urls(text: str) -> str:
  pattern = re.compile(
    r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])")
  zh_url = "網址"
  result = pattern.sub(zh_url, text)
  return result
