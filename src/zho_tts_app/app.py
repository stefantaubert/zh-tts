from logging import getLogger
from time import perf_counter
from typing import Callable

from zho_tts_app.logging_configuration import get_file_logger


def run_main(method: Callable) -> int:
  start = perf_counter()
  success = True

  try:
    method()
  except ValueError as error:
    success = False
    logger = getLogger(__name__)
    logger.debug("ValueError occurred.", exc_info=error)
  except Exception as error:
    success = False
    logger = getLogger(__name__)
    logger.debug("Exception occurred.", exc_info=error)

  duration = perf_counter() - start
  flogger = get_file_logger()
  flogger.debug(f"Total duration (seconds): {duration}")

  exit_code = 0 if success else 1
  return exit_code
