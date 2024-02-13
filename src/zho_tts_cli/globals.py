from pathlib import Path
from tempfile import gettempdir


def get_conf_dir() -> Path:
  conf_dir = Path.home() / ".zho-tts"
  return conf_dir


def get_work_dir() -> Path:
  work_dir = Path(gettempdir()) / "zho-tts"
  return work_dir
