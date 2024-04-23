# zho-tts

[![PyPI](https://img.shields.io/pypi/v/zho-tts.svg)](https://pypi.python.org/pypi/zho-tts)
![PyPI](https://img.shields.io/pypi/pyversions/zho-tts.svg)
[![Hugging Face 🤗](https://img.shields.io/badge/%20%F0%9F%A4%97_Hugging_Face-zho--tts-blue.svg)](https://huggingface.co/spaces/stefantaubert/zho-tts)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/pytorch-2.0/)
[![MIT](https://img.shields.io/github/license/stefantaubert/zh-tts.svg)](https://github.com/stefantaubert/zh-tts/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/zho-tts.svg)](https://pypi.python.org/pypi/zho-tts/#files)
![PyPI](https://img.shields.io/pypi/implementation/zho-tts.svg)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/zh-tts/latest/master.svg)](https://github.com/stefantaubert/zh-tts/compare/v0.0.2...master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11048515.svg)](https://doi.org/10.5281/zenodo.11048515)

Web app, command-line interface and Python library for synthesizing Chinese texts into speech.

## Installation

```sh
pip install zho-tts --user
```

## Usage as web app

Visit [🤗 Hugging Face](https://huggingface.co/spaces/stefantaubert/zho-tts) for a live demo.

<a href="https://huggingface.co/spaces/stefantaubert/zho-tts">
<img src="https://github.com/stefantaubert/zh-tts/raw/master/img/hf.png" alt="Screenshot Hugging Face" style="max-width: 600px; width: 100%"/>
</a>

You can also run it locally be executing `zho-tts-web` in CLI and opening your browser on [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Usage as CLI

```sh
zho-tts-cli synthesize "长江 航务 管理局 和 长江 轮船 总公司 最近 决定 安排 一百三十三 艘 客轮 迎接 长江 干线 春运。"
```

The output can be listened [here](https://github.com/stefantaubert/zh-tts/raw/master/examples/synthesize.wav).

```sh
# Same example using IPA input
zho-tts-cli synthesize-ipa "ʈʂː|a˧˩˧˘|ŋ|tɕ˘|j|a˥˘|ŋ˘|SIL0|x|a˧˥˘|ŋ|u˥˩|SIL0|k|w|a˧˩˧|n|l˘|i˧˩˧|tɕː|y˧˥ˑ|SIL0|x|ɤ˧˥|SIL0|ʈʂː|a˧˩˧˘|ŋ|tɕ˘|j|a˥˘|ŋ|SIL0|l|w|ə˧˥|n|ʈʂʰ˘|w|a˧˥|n|SIL0|ts˘|ʊ˧˩˧|ŋ˘|kː|ʊ˥|ŋ|s|ɹ̩˥ˑ|SIL0|ts|w˘|ei̯˥˩|tɕ|i˥˩˘|n|SIL0|tɕ|ɥ|e˧˥|t|i˥˩|ŋ|SIL3|a˥|n|pʰ|ai̯˧˥|SIL0|i˥ˑ|p|ai̯˧˩˧|s|a˥˘|n|ʂ˘|ɻ̩˧˥|s|a˥|n|SIL0|s˘|ou̯˥|SIL0|kʰˑ|ɤ˥˩|lː|wˑ|ə˧˥ˑ|n|SIL0|i˧˥ː|ŋ|tɕ˘|j˘|e˥|SIL0|ʈʂː|a˧˩˧|ŋ|tɕ˘|j|a˥˘|ŋ|SIL0|k˘|a˥˩|n|ɕ|j˘|ɛ˥˩|n˘|SIL0|ʈʂʰˑ|w˘|ə˥˘|nː|y˥˩ˑ|nː|。"
```

The output can be listened [here](https://github.com/stefantaubert/zh-tts/raw/master/examples/synthesize-ipa.wav).

## Usage as library

```py
from pathlib import Path
from tempfile import gettempdir

from zho_tts import Synthesizer, Transcriber, normalize_audio, save_audio

text = "长江 航务 管理局 和 长江 轮船 总公司 最近 决定 安排 一百三十三 艘 客轮 迎接 长江 干线 春运。"

transcriber = Transcriber()
synthesizer = Synthesizer()

text_ipa = transcriber.transcribe_to_ipa(text)
audio = synthesizer.synthesize(text_ipa)

tmp_dir = Path(gettempdir())
save_audio(audio, tmp_dir / "output.wav")

# Optional: normalize output
normalize_audio(tmp_dir / "output.wav", tmp_dir / "output_norm.wav")
```

## Model info

The used TTS model is published [here](https://doi.org/10.5281/zenodo.10209990).

### Phoneme set

- Vowels: a ɛ e ə ɚ ɤ i o u ʊ y
- Diphthongs: ai̯ au̯ ei̯ ou̯
- Consonants: f j k kʰ l m n p pʰ ɹ̩¹ ɻ¹ ɻ̩¹ s t ts tsʰ tɕ tɕʰ tʰ w x ŋ ɕ ɥ ʂ ʈʂ ʈʂʰ
- Breaks:
  - SIL0 (no break)
  - SIL1 (short break)
  - SIL2 (break)
  - SIL3 (long break)
- special characters: 。 ?

Vowels and diphthongs contain one of these tones:

- ˥ (first tone)
- ˧˥ (second tone)
- ˧˩˧ (third tone)
- ˥˩ (fourth tone)
- (none)

¹ These consonants contain also tones.

Vowels, diphthongs and consonants contain one of these duration markers:

- ˘ -> very short, e.g., ou̯˘
- nothing -> normal, e.g., ou̯
- ˑ -> half long, e.g., ou̯ˑ
- ː -> long, e.g., ou̯ː

Tones and duration markers can be combined, e.g., ə˧˥ː

### Speakers

![Objective Evaluation](https://github.com/stefantaubert/zh-tts/raw/master/img/eval.png)

## Citation

If you want to cite this repo, you can use the BibTeX-entry generated by GitHub (see *About => Cite this repository*).

- Taubert, S. (2024). zho-tts (Version 0.0.2) [Computer software]. [https://doi.org/10.5281/zenodo.11048515](https://doi.org/10.5281/zenodo.11048515)

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

The authors gratefully acknowledge the GWK support for funding this project by providing computing time through the Center for Information Services and HPC (ZIH) at TU Dresden.

The authors are grateful to the Center for Information Services and High Performance Computing [Zentrum fur Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden for providing its facilities for high throughput calculations.
