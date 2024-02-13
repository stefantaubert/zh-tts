import argparse
import codecs
from argparse import ArgumentTypeError
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar

import torch
from ordered_set import OrderedSet

T = TypeVar("T")


class ConvertToOrderedSetAction(argparse._StoreAction):
  def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Optional[Sequence], option_string: Optional[str] = None):
    # Note: normal set is not possible because set is not a Sequence
    val: Optional[OrderedSet] = None
    if values is not None:
      val = OrderedSet(values)
    super().__call__(parser, namespace, val, option_string)


def parse_device(value: str) -> torch.device:
  try:
    device = torch.device(value)
  except Exception as ex:
    raise ArgumentTypeError("Device was not found!") from ex
  return device


def get_torch_devices():
  yield "cpu"
  cuda_count = torch.cuda.device_count()
  if cuda_count == 1:
    yield "cuda"
  else:
    for i in range(cuda_count):
      yield f"cuda:{i}"


def parse_codec(value: str) -> str:
  value = parse_required(value)
  try:
    codecs.lookup(value)
  except LookupError as error:
    raise ArgumentTypeError("Codec was not found!") from error
  return value


def parse_path(value: str) -> Path:
  value = parse_required(value)
  try:
    path = Path(value)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a path!") from error
  return path


def parse_optional_value(value: str, method: Callable[[str], T]) -> Optional[T]:
  if value is None:
    return None
  return method(value)


def parse_float_between_zero_and_one(value: str) -> float:
  result = parse_float(value)
  if not 0 <= result <= 1:
    raise ArgumentTypeError("Value needs to be in interval [0, 1]!")
  return result


def parse_character(value: str) -> str:
  if len(value) != 1:
    raise ArgumentTypeError("Value needs to be one character!")
  return value


def get_optional(method: Callable[[str], T]) -> Callable[[str], Optional[T]]:
  result = partial(
    parse_optional_value,
    method=method,
  )
  return result


def parse_existing_file(value: str) -> Path:
  path = parse_path(value)
  if not path.is_file():
    raise ArgumentTypeError("File was not found!")
  return path


def parse_existing_directory(value: str) -> Path:
  path = parse_path(value)
  if not path.is_dir():
    raise ArgumentTypeError("Directory was not found!")
  return path


def parse_required(value: Optional[str]) -> str:
  if value is None:
    raise ArgumentTypeError("Value must not be None!")
  return value


def parse_non_empty(value: Optional[str]) -> str:
  value = parse_required(value)
  if value == "":
    raise ArgumentTypeError("Value must not be empty!")
  return value


def parse_non_empty_or_whitespace(value: str) -> str:
  value = parse_required(value)
  if value.strip() == "":
    raise ArgumentTypeError("Value must not be empty or whitespace!")
  return value


def parse_float(value: str) -> float:
  value_str = parse_required(value)
  try:
    result = float(value_str)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a decimal number!") from error
  return result


def parse_positive_float(value: str) -> float:
  result = parse_float(value)
  if not result > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return result


def parse_non_negative_float(value: str) -> float:
  result = parse_float(value)
  if not result >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return result


def parse_integer(value: str) -> int:
  value_str = parse_required(value)
  if not value_str.isdigit():
    raise ArgumentTypeError("Value needs to be an integer!")
  result = int(value_str)
  return result


def parse_positive_integer(value: str) -> int:
  result = parse_integer(value)
  if not result > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return result


def parse_integer_greater_one(value: str) -> int:
  result = parse_integer(value)
  if not result > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return result


def parse_float_greater_one(value: str) -> float:
  result = parse_float(value)
  if not result > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return result


def parse_non_negative_integer(value: str) -> int:
  result = parse_integer(value)
  if not result >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return result
