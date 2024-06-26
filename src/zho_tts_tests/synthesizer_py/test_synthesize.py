from logging import getLogger

import numpy as np
from pytest import raises

from zho_tts.helper import normalize_audio
from zho_tts.io import save_audio
from zho_tts.synthesizer import Synthesizer
from zho_tts_tests.helper import get_tests_conf_dir


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

  np.testing.assert_array_almost_equal(
    audio[:10],
    np.array(
      [
        -0.00558978, -0.00825027, -0.00817811, -0.00901944, -0.00506526,
        -0.00400619, -0.00676229, -0.0055937 , -0.00873097, -0.00480694
      ],
      dtype=np.float64,
    )
  )
  assert audio.dtype == np.float64
  assert audio.shape == (95546,)


def test_empty():
  conf_dir = get_tests_conf_dir()

  s = Synthesizer(conf_dir)

  text = ''

  audio = s.synthesize(text)
  assert audio.dtype == np.float64
  assert audio.shape == (0,)


def test_invalid_speaker_raise_value_error():
  conf_dir = get_tests_conf_dir()

  s = Synthesizer(conf_dir)

  text = 'j|ou̯˧˩˧|SIL0|i˥|SIL0|tsʰ|ɹ̩˥˩|SIL1|p|ei̯˧˩˧|f|ə˥|ŋ|SIL0|x|ɤ˧˥|SIL0|tʰ|ai̯˥˩|j|a˧˥˘|ŋ˘|。|SIL2\nʈʂ|ə˥˩|ŋ|ts|ai̯˥˩|SIL0|ʈʂˑ|ə˥|ŋˑ|l|w|ə˥˩ː|nˑ|SIL0|ʂ|w˘|ei̯˧˥ː|SIL0|p|i˧˩˧|tɕ˘|j|au̯˥˩˘|SIL0|ɚ˥˩|ʂ˘|ɻ̩˧˥|i˥ˑ|SIL0|j|ou̯˧˩˧|SIL0|p|ə˧˩˧|n|ʂ|ɻ̩˥˩|。|SIL2'
  with raises(ValueError) as error:
    s.synthesize(text, speaker="abc")
  assert error.value.args[0] == "Speaker 'abc' is not available! Available speakers: A11, A12, A13, A14, A19, A2, A22, A23, A32, A33, A34, A35, A36, A4, A5, A6, A7, A8, A9, B11, B12, B15, B2, B21, B22, B31, B32, B33, B34, B4, B6, B7, B8, C12, C13, C14, C17, C18, C19, C2, C20, C21, C22, C23, C31, C32, C4, C6, C7, C8, D11, D12, D13, D21, D31, D32, D4, D6, D7, D8."
