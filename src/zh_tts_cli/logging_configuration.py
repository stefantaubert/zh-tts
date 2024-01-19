import logging
import os
from logging import Formatter, Handler, Logger, StreamHandler, getLogger
from logging.handlers import MemoryHandler
from pathlib import Path


class ConsoleFormatter(logging.Formatter):
  """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

  purple = '\x1b[34m'
  blue = '\x1b[36m'
  # blue = '\x1b[38;5;39m'
  yellow = '\x1b[38;5;226m'
  red = '\x1b[1;49;31m'
  bold_red = '\x1b[1;49;31m'
  reset = '\x1b[0m'

  def __init__(self):
    super().__init__()
    self.datefmt = '%H:%M:%S'
    fmt = '(%(levelname)s) %(message)s'
    fmt_info = '%(message)s'

    self.fmts = {
        logging.NOTSET: self.purple + fmt + self.reset,
        logging.DEBUG: self.blue + fmt + self.reset,
        logging.INFO: fmt_info,
        logging.WARNING: self.yellow + fmt + self.reset,
        logging.ERROR: self.red + fmt + self.reset,
        logging.CRITICAL: self.bold_red + fmt + self.reset,
    }

  def format(self, record):
    log_fmt = self.fmts.get(record.levelno)
    formatter = logging.Formatter(log_fmt, self.datefmt)

    return formatter.format(record)


def add_console_out(logger: Logger) -> StreamHandler:
  console = StreamHandler()
  logger.addHandler(console)
  set_console_formatter(console)
  return console


def set_console_formatter(handler: Handler) -> None:
  logging_formatter = ConsoleFormatter()
  handler.setFormatter(logging_formatter)


def set_logfile_formatter(handler: Handler) -> None:
  fmt = '[%(asctime)s.%(msecs)03d] %(name)s (%(levelname)s) %(message)s'
  datefmt = '%Y/%m/%d %H:%M:%S'
  logging_formatter = Formatter(fmt, datefmt)
  handler.setFormatter(logging_formatter)


def get_cli_logger() -> Logger:
  logger = getLogger("zh_tts_cli")
  return logger


def get_file_logger() -> Logger:
  flogger = getLogger("zh_tts_cli_file")
  if flogger.propagate:
    flogger.propagate = False
  return flogger


def configure_cli_logger() -> None:
  cli_logger = getLogger("zh_tts_cli")
  cli_logger.handlers.clear()
  assert len(cli_logger.handlers) == 0
  add_console_out(cli_logger)

  core_logger = getLogger("zh_tts")
  core_logger.parent = cli_logger

  file_logger = get_file_logger()
  cli_logger.parent = file_logger


def configure_root_logger() -> None:
  # productive = False
  # loglevel = logging.INFO if productive else logging.DEBUG
  root_logger = getLogger()
  root_logger.setLevel(logging.DEBUG)
  root_logger.manager.disable = logging.NOTSET
  if len(root_logger.handlers) > 0:
    console = root_logger.handlers[0]
    set_console_formatter(console)
  else:
    console = add_console_out(root_logger)

  console.setLevel(logging.DEBUG)


def configure_file_logger(path: Path, debug: bool = False, buffer_capacity: int = 1):
  flogger = get_file_logger()
  assert len(flogger.handlers) == 0
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.is_file():
    os.remove(path)
  path.write_text("")
  fh = logging.FileHandler(path)
  set_logfile_formatter(fh)
  level = logging.DEBUG if debug else logging.INFO
  fh.setLevel(level)
  mh = MemoryHandler(buffer_capacity, logging.ERROR, fh, True)
  flogger.addHandler(mh)
