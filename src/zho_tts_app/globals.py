from importlib.metadata import version
from pathlib import Path
from tempfile import gettempdir

APP_NAME = "zho-tts"

APP_VERSION = version(APP_NAME)


def get_conf_dir() -> Path:
  conf_dir = Path.home() / ".zho-tts"
  return conf_dir


def get_work_dir() -> Path:
  work_dir = Path(gettempdir()) / "zho-tts"
  return work_dir


def get_log_path() -> Path:
  return Path(gettempdir()) / "zho-tts.log"
