from pathlib import Path


def get_tests_conf_dir() -> Path:
  result = Path("/tmp/zh-tts.tests")
  result.mkdir(parents=True, exist_ok=True)
  return result
  # return DEFAULT_CONF_DIR
