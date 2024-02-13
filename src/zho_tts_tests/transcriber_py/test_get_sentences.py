from zho_tts.transcriber import get_sentences


def test_component():
  res = list(get_sentences("气於是， 北風 只好 認輸了。 气於是， 北風 只好 認輸了！ 气於是， 北風 只好 認輸了？"))
  assert res == [
    "气於是， 北風 只好 認輸了。",
    "气於是， 北風 只好 認輸了！",
    "气於是， 北風 只好 認輸了？",
  ]
