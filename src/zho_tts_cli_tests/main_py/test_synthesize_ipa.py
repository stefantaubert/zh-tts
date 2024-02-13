from zho_tts.helper import get_default_device
from zho_tts_cli.cli import reset_work_dir
from zho_tts_cli.globals import get_work_dir
from zho_tts_cli.main import synthesize_ipa


def test_component():
  text = 'j|ou̯˧˩˧|SIL0|i˥|SIL0|tsʰ|ɹ̩˥˩|SIL1|p|ei̯˧˩˧|f|ə˥|ŋ|SIL0|x|ɤ˧˥|SIL0|tʰ|ai̯˥˩|j|a˧˥˘|ŋ˘|。|SIL2\nʈʂ|ə˥˩|ŋ|ts|ai̯˥˩|SIL0|ʈʂˑ|ə˥|ŋˑ|l|w|ə˥˩ː|nˑ|SIL0|ʂ|w˘|ei̯˧˥ː|SIL0|p|i˧˩˧|tɕ˘|j|au̯˥˩˘|SIL0|ɚ˥˩|ʂ˘|ɻ̩˧˥|i˥ˑ|SIL0|j|ou̯˧˩˧|SIL0|p|ə˧˩˧|n|ʂ|ɻ̩˥˩|。|SIL2'
  work_dir = get_work_dir()

  reset_work_dir()
  synthesize_ipa(text, "D6", 5000, 1.0, 0.0005, 0, get_default_device(),
                 0.2, 1.0, 2, work_dir / "output.wav")
