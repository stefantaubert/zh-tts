from pathlib import Path

import torch
from ffmpy import FFExecutableNotFoundError, FFmpeg


def get_default_device():
  cuda_count = torch.cuda.device_count()
  if cuda_count == 1:
    return torch.device("cuda")
  if cuda_count > 1:
    return torch.device("cuda:0")
  return torch.device("cpu")


def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))


def normalize_audio(path: Path, output: Path):
  ffmpeg_normalization = FFmpeg(
    inputs={
      str(path.absolute()): None
    },
    outputs={
      str(output.absolute()): "-acodec pcm_s16le -ar 22050 -ac 1 -af loudnorm=I=-16:LRA=11:TP=-1.5 -y -hide_banner -loglevel error"
    },
  )

  try:
    ffmpeg_normalization.run()
  except FFExecutableNotFoundError as error:
    raise Exception("FFmpeg was not found, therefore no normalization was applied!") from error
  except Exception as error:
    raise Exception(
      "FFmpeg couldn't be executed, therefore no normalization was applied!") from error
