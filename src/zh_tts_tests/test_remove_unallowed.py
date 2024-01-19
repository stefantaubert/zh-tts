from zh_tts.transcriber import remove_unallowed


def test_component():
  res = remove_unallowed("气息才让 trad ”。")
  assert res == "气息才让  。"
