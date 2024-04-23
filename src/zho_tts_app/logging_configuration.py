import logging
import os
import platform
import sys
from logging import Formatter, Handler, Logger, StreamHandler, getLogger
from pathlib import Path
from pkgutil import iter_modules
from typing import Set

from zho_tts_app.globals import APP_VERSION, get_log_path


def initialize_logging() -> None:
  # CLI logging = INFO
  # External loggers go to file-logger
  # file-logger = DEBUG

  # # disable mpl temporarily
  # mpl_logger = getLogger("matplotlib")
  # mpl_logger.disabled = True
  # mpl_logger.propagate = False

  configure_root_logger(logging.INFO)
  root_logger = getLogger()

  logfile = get_log_path()
  configure_file_logger(logfile, logging.DEBUG)

  configure_app_logger(logging.INFO)
  configure_external_loggers()

  # path not encapsulated in "" because it is only console out
  root_logger.info(f"Log will be written to: {logfile.absolute()}")


def configure_external_loggers() -> None:
  file_logger = get_file_logger()
  for logger_name in ("httpcore", "httpx", "asyncio", "matplotlib"):
    logger = getLogger(logger_name)
    logger.parent = file_logger
    logger.disabled = False
    logger.propagate = True
    logger.level = logging.DEBUG


class ConsoleFormatter(logging.Formatter):
  """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

  purple = '\x1b[34m'
  blue = '\x1b[36m'
  # blue = '\x1b[38;5;39m'
  yellow = '\x1b[38;5;226m'
  red = '\x1b[1;49;31m'
  bold_red = '\x1b[1;49;31m'
  reset = '\x1b[0m'

  collected_loggers: Set[str] = set()

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
    self.collected_loggers.add(record.name)

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


def get_app_logger() -> Logger:
  logger = getLogger("zho_tts_app")
  return logger


def get_file_logger() -> Logger:
  flogger = getLogger("file.zho_tts_app")
  return flogger


def configure_app_logger(level: int) -> None:
  app_logger = get_app_logger()
  app_logger.handlers.clear()
  assert len(app_logger.handlers) == 0
  console_handler = add_console_out(app_logger)
  # console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
  app_logger.setLevel(level)

  core_logger = getLogger("zho_tts")
  core_logger.parent = app_logger

  file_logger = get_file_logger()
  app_logger.parent = file_logger


def configure_root_logger(level: int) -> None:
  # productive = False
  # loglevel = logging.INFO if productive else logging.DEBUG
  root_logger = getLogger()
  root_logger.setLevel(level)
  root_logger.manager.disable = logging.NOTSET
  if len(root_logger.handlers) > 0:
    console = root_logger.handlers[0]
    set_console_formatter(console)
  else:
    console = add_console_out(root_logger)

  # console.setLevel(logging.DEBUG if debug else logging.INFO)


def configure_file_logger(path: Path, level: int):
  flogger = get_file_logger()
  assert len(flogger.handlers) == 0
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.is_file():
    os.remove(path)
  path.write_text("")
  fh = logging.FileHandler(path)
  set_logfile_formatter(fh)
  # fh.setLevel(logging.DEBUG if debug else logging.INFO)
  flogger.setLevel(level)
  # mh = MemoryHandler(buffer_capacity, logging.ERROR, fh, True)
  flogger.addHandler(fh)

  if flogger.propagate:
    flogger.propagate = False


def log_sysinfo():
  flogger = get_file_logger()

  sys_version = sys.version.replace('\n', '')
  flogger.debug(f"CLI version: {APP_VERSION}")
  flogger.debug(f"Python version: {sys_version}")
  flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

  my_system = platform.uname()
  flogger.debug(f"System: {my_system.system}")
  flogger.debug(f"Node Name: {my_system.node}")
  flogger.debug(f"Release: {my_system.release}")
  flogger.debug(f"Version: {my_system.version}")
  flogger.debug(f"Machine: {my_system.machine}")
  flogger.debug(f"Processor: {my_system.processor}")
