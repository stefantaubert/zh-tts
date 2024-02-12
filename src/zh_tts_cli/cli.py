import argparse
import logging
import platform
import shutil
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, Generator, List, Tuple, cast

from zh_tts_cli.globals import get_conf_dir, get_work_dir
from zh_tts_cli.logging_configuration import (configure_cli_logger, configure_file_logger,
                                              configure_root_logger, get_file_logger)
from zh_tts_cli.main import init_synthesize_ipa_parser, init_synthesize_zh_parser

__APP_NAME = "zh-tts"

__version__ = version(__APP_NAME)

INVOKE_HANDLER_VAR = "invoke_handler"


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def print_features():
  parsers = get_parsers()
  for command, description, method in parsers:
    print(f"- `{command}`: {description}")


def get_parsers() -> Generator[Tuple[str, str, Callable], None, None]:
  yield from (
    ("synthesize", "synthesize Chinese texts", init_synthesize_zh_parser),
    ("synthesize-ipa", "synthesize Chinese texts transcribed in IPA", init_synthesize_ipa_parser),
  )


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="Command-line interface for synthesizing Chinese texts into speech.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")

  for command, description, method in get_parsers():
    method_parser = subparsers.add_parser(
      command, help=description, formatter_class=formatter)
    # init parser
    invoke_method = method(method_parser)
    method_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: invoke_method,
    })

    logging_group = method_parser.add_argument_group("logging arguments")
    # logging_group.add_argument("--work-directory", type=parse_path, metavar="DIRECTORY",
    #                            help="path to write the log", default=Path(gettempdir()) / "en-tts")
    logging_group.add_argument("--loglevel", metavar="LEVEL", type=int,
                               choices=[0, 1, 2], help="log-level", default=1)
    logging_group.add_argument("--debug", action="store_true",
                               help="include debugging information in log")

  return main_parser


def reset_work_dir():
  root_logger = getLogger()
  work_dir = get_work_dir()

  if work_dir.is_dir():
    root_logger.debug("Deleting working directory ...")
    shutil.rmtree(work_dir)
  root_logger.debug("Creating working directory ...")
  work_dir.mkdir(parents=False, exist_ok=False)


def ensure_conf_dir_exists():
  conf_dir = get_conf_dir()
  if not conf_dir.is_dir():
    root_logger = getLogger()
    root_logger.debug("Creating configuration directory ...")
    conf_dir.mkdir(parents=False, exist_ok=False)


def parse_args(args: List[str]) -> None:
  local_debugging = debug_file_exists()

  configure_root_logger(local_debugging)
  root_logger = getLogger()
  root_logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit as error:
    error_code = error.args[0]
    # -v -> 0; invalid arg -> 2
    sys.exit(error_code)

  root_logger.debug(f"Parsed arguments: {str(ns)}")

  if not hasattr(ns, INVOKE_HANDLER_VAR):
    parser.print_help()
    sys.exit(0)

  debug = cast(bool, ns.debug)
  root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

  invoke_handler: Callable[..., bool] = getattr(ns, INVOKE_HANDLER_VAR)
  delattr(ns, INVOKE_HANDLER_VAR)

  ensure_conf_dir_exists()

  try:
    reset_work_dir()
  except Exception as ex:
    root_logger.exception("Working directory couldn't be resetted!", exc_info=ex, stack_info=True)
    sys.exit(1)

  work_dir = get_work_dir()
  logfile = work_dir / "output.log"
  try:
    configure_file_logger(logfile, debug, 1)
  except Exception as ex:
    root_logger.exception("Logging to file is not possible. Exiting.", exc_info=ex, stack_info=True)
    sys.exit(1)

  configure_cli_logger(debug)

  flogger = get_file_logger()

  if debug:
    sys_version = sys.version.replace('\n', '')
    flogger.debug(f"CLI version: {__version__}")
    flogger.debug(f"Python version: {sys_version}")
    flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

    my_system = platform.uname()
    flogger.debug(f"System: {my_system.system}")
    flogger.debug(f"Node Name: {my_system.node}")
    flogger.debug(f"Release: {my_system.release}")
    flogger.debug(f"Version: {my_system.version}")
    flogger.debug(f"Machine: {my_system.machine}")
    flogger.debug(f"Processor: {my_system.processor}")

    flogger.debug(f"Received arguments: {str(args)}")
    flogger.debug(f"Parsed arguments: {str(ns)}")

  start = perf_counter()
  success = True

  try:
    invoke_handler(ns)
  except ValueError as error:
    success = False
    logger = getLogger(__name__)
    logger.debug("ValueError occurred.", exc_info=error)
  except Exception as error:
    success = False
    logger = getLogger(__name__)
    logger.debug("Exception occurred.", exc_info=error)

  duration = perf_counter() - start
  flogger.debug(f"Total duration (seconds): {duration}")

  exit_code = 0 if success else 1
  if debug:
    # path not encapsulated in "" because it is only console out
    root_logger.info(f"See log: {logfile.absolute()}")

  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def debug_file_exists():
  return (Path(gettempdir()) / f"{__APP_NAME}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{__APP_NAME}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run()
