
from functools import partial
from typing import Dict

from zho_tts_app.app import run_main
from zho_tts_app.globals import get_log_path
from zho_tts_app.main import (load_models_to_cache, reset_log, reset_work_dir, synthesize_ipa,
                              synthesize_zho)


def dummy_method1_zho():
  reset_work_dir()
  cache = load_models_to_cache()
  result = synthesize_zho("有一次， 北风 和 太阳！ 正在 争论 谁 比较 test 21 有本事。", cache)
  print(result)


def dummy_method1_ipa():
  reset_work_dir()
  cache = load_models_to_cache()
  result = synthesize_ipa(
    'j|ou̯˧˩˧|SIL0|i˥|SIL0|tsʰ|ɹ̩˥˩|SIL1|p|ei̯˧˩˧|f|ə˥|ŋ|SIL0|x|ɤ˧˥|SIL0|tʰ|ai̯˥˩|j|a˧˥˘|ŋ˘|。|SIL2\nʈʂ|ə˥˩|ŋ|ts|ai̯˥˩|SIL0|ʈʂˑ|ə˥|ŋˑ|l|w|ə˥˩ː|nˑ|SIL0|ʂ|w˘|ei̯˧˥ː|SIL0|p|i˧˩˧|tɕ˘|j|au̯˥˩˘|SIL0|ɚ˥˩|ʂ˘|ɻ̩˧˥|i˥ˑ|SIL0|j|ou̯˧˩˧|SIL0|p|ə˧˩˧|n|ʂ|ɻ̩˥˩|。|SIL2', cache)
  print(result)


def dummy_method2_zho(cache: Dict):
  reset_work_dir()
  result = synthesize_zho("有一次， 北风 和 太阳！ 正在 争论 谁 比较 test 21 有本事。", cache)
  print(result)


def test_component_zho():
  exit_code = run_main(dummy_method1_zho)
  assert exit_code == 0


def test_component_ipa():
  exit_code = run_main(dummy_method1_ipa)
  assert exit_code == 0


def test_component_zho_twice():
  cache = load_models_to_cache()

  method = partial(dummy_method2_zho, cache=cache)
  exit_code = run_main(method)
  log1 = get_log_path().read_text("utf8")

  reset_log()

  method()
  log2 = get_log_path().read_text("utf8")

  assert exit_code == 0
  assert len(log1) > 0
  assert len(log2) > 0
