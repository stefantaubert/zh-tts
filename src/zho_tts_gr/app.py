import shutil
import sys
import zipfile
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, Tuple

import gradio as gr
import numpy.typing as npt
from scipy.io.wavfile import read

from zho_tts_app import (APP_VERSION, get_log_path, get_work_dir, initialize_logging,
                         load_models_to_cache, reset_log, run_main, synthesize_zho)

speakers = OrderedDict()
speakers["♀ A02"] = "A2"
speakers["♀ A04"] = "A4"
speakers["♀ A05"] = "A5"
speakers["♀ A06"] = "A6"
speakers["♀ A07"] = "A7"
speakers["♀ A11"] = "A11"
speakers["♀ A12"] = "A12"
speakers["♀ A13"] = "A13"
speakers["♀ A14"] = "A14"
speakers["♀ A19"] = "A19"
speakers["♀ A22"] = "A22"
speakers["♀ A23"] = "A23"
speakers["♀ A32"] = "A32"
speakers["♀ A34"] = "A34"
speakers["♀ A36"] = "A36"
speakers["♀ B02"] = "B2"
speakers["♀ B04"] = "B4"
speakers["♀ B06"] = "B6"
speakers["♀ B07"] = "B7"
speakers["♀ B11"] = "B11"
speakers["♀ B12"] = "B12"
speakers["♀ B15"] = "B15"
speakers["♀ B22"] = "B22"
speakers["♀ B31"] = "B31"
speakers["♀ B32"] = "B32"
speakers["♀ B33"] = "B33"
speakers["♀ C02"] = "C2"
speakers["♀ C04"] = "C4"
speakers["♀ C06"] = "C6"
speakers["♀ C07"] = "C7"
speakers["♀ C12"] = "C12"
speakers["♀ C13"] = "C13"
speakers["♀ C14"] = "C14"
speakers["♀ C17"] = "C17"
speakers["♀ C18"] = "C18"
speakers["♀ C19"] = "C19"
speakers["♀ C20"] = "C20"
speakers["♀ C21"] = "C21"
speakers["♀ C22"] = "C22"
speakers["♀ C23"] = "C23"
speakers["♀ C31"] = "C31"
speakers["♀ C32"] = "C32"
speakers["♀ D04"] = "D4"
speakers["♀ D06"] = "D6"
speakers["♀ D07"] = "D7"
speakers["♀ D11"] = "D11"
speakers["♀ D12"] = "D12"
speakers["♀ D13"] = "D13"
speakers["♀ D21"] = "D21"
speakers["♀ D31"] = "D31"
speakers["♀ D32"] = "D32"
speakers["♂ A08"] = "A8"
speakers["♂ A09"] = "A9"
speakers["♂ A33"] = "A33"
speakers["♂ A35"] = "A35"
speakers["♂ B08"] = "B8"
speakers["♂ B21"] = "B21"
speakers["♂ B34"] = "B34"
speakers["♂ C08"] = "C8"
speakers["♂ D08"] = "D8"


def run() -> None:
  try:
    initialize_logging()
  except ValueError as ex:
    print("Logging not possible!")
    sys.exit(1)

  interface = build_interface(cache_examples=False)
  interface.queue()

  launch_method = partial(
    interface.launch,
    share=False,
    debug=True,
    inbrowser=True,
    quiet=False,
    show_api=False,
  )

  exit_code = run_main(launch_method)
  sys.exit(exit_code)


def build_interface(cache_examples: bool = False) -> gr.Blocks:
  cache = load_models_to_cache()

  fn = partial(synt, cache=cache)

  with gr.Blocks(
    title="zho-tts"
  ) as web_app:
    gr.Markdown(
      """
      # Chinese Speech Synthesis

      Enter or paste your text into the provided text box and click the **Synthesize** button to convert it into speech. You can adjust settings as desired before synthesizing.
      """
    )

    with gr.Tab("Synthesis"):
      with gr.Row():
        with gr.Column():
          with gr.Group():
            input_txt_box = gr.Textbox(
              None,
              label="Input",
              placeholder="Enter the text you want to synthesize (or load an example from below).",
              lines=10,
              max_lines=5000,
            )

            speaker_drop = gr.Dropdown(
              choices=[s for s in speakers],
              allow_custom_value=False,
              label="Speaker",
            )

            with gr.Accordion("Settings", open=False):
              sent_norm_check_box = gr.Checkbox(
                False,
                label="Skip normalization",
                info="Skip normalization of numbers, units and abbreviations."
              )

              word_seg_check_box = gr.Checkbox(
                False,
                label="Skip word segmentation",
                info="Skip segmentation of words."
              )

              sent_sep_check_box = gr.Checkbox(
                False,
                label="Skip sentence separation",
                info="Skip sentence separation after these characters: .?!"
              )

              sil_sent_txt_box = gr.Number(
                0.2,
                minimum=0.0,
                maximum=60,
                step=0.1,
                label="Silence between sentences (s)",
                info="Insert silence between each sentence."
              )

              sil_para_txt_box = gr.Number(
                1.0,
                minimum=0.0,
                maximum=60,
                step=0.1,
                label="Silence between paragraphs (s)",
                info="Insert silence between each paragraph."
              )

              seed_txt_box = gr.Number(
                0,
                minimum=0,
                maximum=999999,
                label="Seed",
                info="Seed used for inference in order to be able to reproduce the results."
              )

              sigma_txt_box = gr.Number(
                1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                label="Sigma",
                info="Sigma used for inference in WaveGlow."
              )

              max_decoder_steps_txt_box = gr.Number(
                5000,
                minimum=1,
                step=500,
                label="Maximum decoder steps",
                info="Stop the synthesis after this number of decoder steps at the latest."
              )

              denoiser_txt_box = gr.Number(
                0.005,
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                label="Denoiser strength",
                info="Level of noise reduction used to remove the noise bias from WaveGlow."
              )

          synt_btn = gr.Button("Synthesize", variant="primary")

        with gr.Column():
          with gr.Group():

            with gr.Row():
              with gr.Column():

                out_audio = gr.Audio(
                  type="numpy",
                  label="Output",
                  autoplay=True,
                )

                with gr.Accordion(
                    "Transcription",
                    open=False,
                  ):
                  output_txt_box = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="The IPA transcription of the input text will be displayed here.",
                    lines=10,
                    max_lines=5000,
                  )

                with gr.Accordion(
                    "Log",
                    open=False,
                  ):
                  out_md = gr.Textbox(
                    interactive=False,
                    show_copy_button=True,
                    lines=15,
                    max_lines=10000,
                    placeholder="Log will be displayed here.",
                    show_label=False,
                  )

                  dl_btn = gr.DownloadButton(
                    "Download working directory",
                    variant="secondary",
                  )

      with gr.Row():
        gr.Examples(
          examples=[
            [
              "长江 航务 管理局 和 长江 轮船 总公司 最近 决定 安排 一百三十三 艘 客轮 迎接 长江 干线 春运。",
              "♀ D06", 5000, 1.0, 0.0005, 0, 0.2, 1.0, False, True, True
            ],
            [
              "他们走到四马路一家茶食铺里阿九说要熏鱼他给买了又给转儿买了饼干。",
              "♀ B15", 5000, 1.0, 0.0005, 0, 0.2, 1.0, False, False, True
            ],
            [
              "有 一次， 北風 和 太陽 正在 爭論 誰 比較 有 本事。\n他們 正好 看 到 有 個人 走過， 那 個人 穿著 一 件 斗篷。\n他們 就 說了， 誰 可以 讓 那 個人 脫 掉 那 件 斗篷， 就算 誰 比較 有 本事。\n於是， 北風 就 拼命 地 吹。\n怎 料， 他 吹 得 越 厲害， 那 個人 就 越 是 用 斗篷 包緊 自己。\n最後， 北風 沒 辦法， 只好 放棄。\n接著， 太陽 出來 曬了 一下， 那 個人 就 立刻 把 斗篷 脫 掉了。\n於是， 北風 只好 認輸了。",
              "♂ A08", 5000, 1.0, 0.0005, 0, 0.2, 1.0, False, True, True
            ],
          ],
          fn=fn,
          inputs=[
            input_txt_box,
            speaker_drop,
            max_decoder_steps_txt_box,
            sigma_txt_box,
            denoiser_txt_box,
            seed_txt_box,
            sil_sent_txt_box,
            sil_para_txt_box,
            sent_norm_check_box,
            word_seg_check_box,
            sent_sep_check_box,
          ],
          outputs=[
            out_audio,
            output_txt_box,
            out_md,
            dl_btn,
          ],
          label="Examples",
          cache_examples=cache_examples,
        )

    with gr.Tab("Info"):
      with gr.Column():
        gr.Markdown(
          f"""
          ### General information

          - Language: Chinese
          - Supported special characters: `：；，。！？`

          ### Evaluation results

          |Metric|Value|
          |---|---|
          |Mean MCD-DTW|24.58|
          |Mean penalty|0.1072|

          ### Components

          |Component|Name|URLs|
          |---|---|---|
          |Acoustic model|Tacotron|[Checkpoint](https://zenodo.org/records/10209990), [Code](https://github.com/stefantaubert/tacotron)|
          |Vocoder|WaveGlow|[Checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/models/waveglow_ljs_256channels/files?version=3), [Code](https://github.com/stefantaubert/waveglow)
          |Dataset|THCHS-30|[Link](https://www.openslr.org/18), [Transcriptions](https://zenodo.org/records/7528596)|

          ### Citation

          Taubert, S. (2024). zho-tts (Version {APP_VERSION}) [Computer software]. https://doi.org/10.5281/zenodo.10512789

          ### Acknowledgments

          Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – [CRC 1410](https://gepris.dfg.de/gepris/projekt/416228727?context=projekt&task=showDetail&id=416228727)

          The authors gratefully acknowledge the GWK support for funding this project by providing computing time through the Center for Information Services and HPC (ZIH) at TU Dresden.

          The authors are grateful to the Center for Information Services and High Performance Computing [Zentrum fur Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden for providing its facilities for high throughput calculations.

          ### App information

          - Version: {APP_VERSION}
          - License: [MIT](https://github.com/stefantaubert/zh-tts?tab=MIT-1-ov-file#readme)
          - GitHub: [stefantaubert/zh-tts](https://github.com/stefantaubert/zh-tts)
          """
        )

    # pylint: disable=E1101:no-member
    synt_btn.click(
      fn=fn,
      inputs=[
        input_txt_box,
        speaker_drop,
        max_decoder_steps_txt_box,
        sigma_txt_box,
        denoiser_txt_box,
        seed_txt_box,
        sil_sent_txt_box,
        sil_para_txt_box,
        sent_norm_check_box,
        word_seg_check_box,
        sent_sep_check_box,
      ],
      outputs=[
        out_audio,
        output_txt_box,
        out_md,
        dl_btn,
      ],
      queue=True,
    )

  return web_app


def synt(text: str, speaker: str, max_decoder_steps: int, sigma: float, denoiser_strength: float, seed: int, silence_sentences: float, silence_paragraphs: float, skip_normalization: bool, skip_word_segmentation: bool, skip_sentence_separation: bool, cache: Dict) -> Tuple[Tuple[int, npt.NDArray], str, str, Path]:
  reset_log()
  speaker_name = speakers[speaker]

  result_path = synthesize_zho(
    text, cache,
    speaker=speaker_name,
    max_decoder_steps=max_decoder_steps,
    seed=seed,
    sigma=sigma,
    denoiser_strength=denoiser_strength,
    silence_paragraphs=silence_paragraphs,
    silence_sentences=silence_sentences,
    skip_normalization=skip_normalization, skip_word_segmentation=skip_word_segmentation,
    skip_sentence_separation=skip_sentence_separation,
  )

  rate, audio_int = read(result_path)
  logs = get_log_path().read_text("utf-8")
  ipa_out = ""
  ipa_out_path = get_work_dir() / "text.ipa.readable.txt"
  if ipa_out_path.is_file():
    ipa_out = ipa_out_path.read_text("utf-8")
  zip_dl_path = create_zip_file_of_output()
  return (rate, audio_int), ipa_out, logs, zip_dl_path


def create_zip_file_of_output() -> Path:
  work_dir = get_work_dir()

  name = f"zho-tts-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"

  res = shutil.make_archive(str((Path(gettempdir()) / name).absolute()), 'zip', root_dir=work_dir)

  resulting_zip = Path(res)

  with zipfile.ZipFile(resulting_zip, "a", compression=zipfile.ZIP_DEFLATED) as zipf:
    source_path = get_log_path()
    destination = 'output.log'
    zipf.write(source_path, destination)

  return resulting_zip


if __name__ == "__main__":
  run()
