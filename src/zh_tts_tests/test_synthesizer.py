from logging import getLogger

from zh_tts.helper import normalize_audio
from zh_tts.io import save_audio
from zh_tts.synthesizer import Synthesizer
from zh_tts_tests.helper import get_tests_conf_dir


def test_component():
  conf_dir = get_tests_conf_dir()

  s = Synthesizer(conf_dir)

  text = 'j|ou̯˧˩˧|SIL0|i˥|SIL0|tsʰ|ɹ̩˥˩|SIL1|p|ei̯˧˩˧|f|ə˥|ŋ|SIL0|x|ɤ˧˥|SIL0|tʰ|ai̯˥˩|j|a˧˥˘|ŋ˘|。|SIL2\nʈʂ|ə˥˩|ŋ|ts|ai̯˥˩|SIL0|ʈʂˑ|ə˥|ŋˑ|l|w|ə˥˩ː|nˑ|SIL0|ʂ|w˘|ei̯˧˥ː|SIL0|p|i˧˩˧|tɕ˘|j|au̯˥˩˘|SIL0|ɚ˥˩|ʂ˘|ɻ̩˧˥|i˥ˑ|SIL0|j|ou̯˧˩˧|SIL0|p|ə˧˩˧|n|ʂ|ɻ̩˥˩|。|SIL2'

  audio = s.synthesize(text)
  save_audio(audio, conf_dir / "output.wav")
  normalize_audio(conf_dir / "output.wav", conf_dir / "output_norm.wav")
  logger = getLogger(__name__)
  logger.info(conf_dir / "output_norm.wav")
  assert len(audio) > 0
