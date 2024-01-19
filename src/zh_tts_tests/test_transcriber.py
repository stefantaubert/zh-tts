from zh_tts.transcriber import Transcriber
from zh_tts_tests.helper import get_tests_conf_dir


def test_component():
  conf_dir = get_tests_conf_dir()

  t = Transcriber(conf_dir)

  text = "有一次， 北风 和 太阳！ 正在 争论 谁 比较 test 21 有本事。"
  text_ipa = t.transcribe_to_ipa(text)

  assert text_ipa == 'j|ou̯˧˩˧|SIL0|i˥|SIL0|tsʰ|ɹ̩˥˩|SIL1|p|ei̯˧˩˧|f|ə˥|ŋ|SIL0|x|ɤ˧˥|SIL0|tʰ|ai̯˥˩|j|a˧˥˘|ŋ˘|。|SIL2\nʈʂ|ə˥˩|ŋ|ts|ai̯˥˩|SIL0|ʈʂˑ|ə˥|ŋˑ|l|w|ə˥˩ː|nˑ|SIL0|ʂ|w˘|ei̯˧˥ː|SIL0|p|i˧˩˧|tɕ˘|j|au̯˥˩˘|SIL0|ɚ˥˩|ʂ˘|ɻ̩˧˥|i˥ˑ|SIL0|j|ou̯˧˩˧|SIL0|p|ə˧˩˧|n|ʂ|ɻ̩˥˩|。|SIL2'

