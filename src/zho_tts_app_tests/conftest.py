import logging

from zho_tts_app.logging_configuration import initialize_logging


def pytest_configure():
  loggers = ["numba", "matplotlib"]
  for l in loggers:
    logger = logging.getLogger(l)
    logger.disabled = True
    logger.propagate = False

  initialize_logging()
