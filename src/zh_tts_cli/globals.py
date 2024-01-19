from pathlib import Path
from tempfile import gettempdir


def get_conf_dir() -> Path:
  conf_dir = Path.home() / ".zh-tts"
  return conf_dir


def get_work_dir() -> Path:
  work_dir = Path(gettempdir()) / "zh-tts"
  return work_dir
