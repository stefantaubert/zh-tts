import pickle
from pathlib import Path
from typing import Any

import numpy as np
from waveglow import float_to_wav


def save_obj(obj: Any, path: Path) -> None:
  assert isinstance(path, Path)
  assert path.parent.exists() and path.parent.is_dir()
  with open(path, mode="wb") as file:
    pickle.dump(obj, file)


def load_obj(path: Path) -> Any:
  assert isinstance(path, Path)
  assert path.is_file()
  with open(path, mode="rb") as file:
    return pickle.load(file)


def save_audio(wav: np.ndarray, path: Path):
  float_to_wav(wav, str(path.absolute()), sample_rate=22050)
