import argparse
import logging
import sys
from argparse import ArgumentParser
from functools import partial
from logging import getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Callable, Generator, List, Tuple, cast

from zho_tts_app import APP_NAME, APP_VERSION, get_file_logger, initialize_logging, run_main
from zho_tts_cli.main import init_synthesize_ipa_parser, init_synthesize_zho_parser

INVOKE_HANDLER_VAR = "invoke_handler"


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def print_features():
  parsers = get_parsers()
  for command, description, method in parsers:
    print(f"- `{command}`: {description}")


def get_parsers() -> Generator[Tuple[str, str, Callable], None, None]:
  yield from (
    ("synthesize", "synthesize Chinese texts", init_synthesize_zho_parser),
    ("synthesize-ipa", "synthesize Chinese texts transcribed in IPA", init_synthesize_ipa_parser),
  )


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="Command-line interface for synthesizing Chinese texts into speech.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + APP_VERSION)
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


def parse_args(args: List[str]) -> None:
  root_logger = getLogger()

  try:
    initialize_logging()
  except ValueError as ex:
    root_logger.warning("Logging not possible as intended!", exc_info=ex)

  local_debugging = debug_file_exists()

  if local_debugging:
    root_logger.level = logging.DEBUG

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

  flogger = get_file_logger()
  flogger.setLevel(logging.DEBUG if debug else logging.INFO)
  flogger.debug(f"Received arguments: {str(args)}")
  flogger.debug(f"Parsed arguments: {str(ns)}")

  main_fn = partial(invoke_handler, ns=ns)

  exit_code = run_main(main_fn)

  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def debug_file_exists():
  return (Path(gettempdir()) / f"{APP_NAME}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{APP_NAME}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run()
