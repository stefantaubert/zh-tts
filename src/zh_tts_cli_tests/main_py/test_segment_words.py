
from zh_tts_cli.main import segment_words


def test_component():
  res = segment_words("这个暑假前所，未有的 悠闲。")
  assert res == "这个 暑假 前 所， 未 有的 悠闲。"
