from zho_tts.helper import get_default_device
from zho_tts_cli.cli import reset_work_dir
from zho_tts_cli.globals import get_work_dir
from zho_tts_cli.main import synthesize_zh


def test_component():
  text = "有一次， 北风 和 太阳！ 正在 争论 谁 比较 test 21 有本事。"
  work_dir = get_work_dir()

  reset_work_dir()
  synthesize_zh(text, "D6", 5000, 1.0, 0.0005, 0, get_default_device(),
                     0.2, 1.0, 2, False, False, False, work_dir / "output.wav")
