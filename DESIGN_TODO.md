# DESIGN_TODO

A terse catalogue of open tasks only. Gaps on the host side of the glass2glass
Python-element host are tracked in that repo's `DESIGN_TODO.md`, under
"Python-element host", not here.

## g2g backend coverage

- **Which README pipelines run under `PYML_BACKEND=g2g` is not established
  yet.** `PYML_BACKEND=g2g uv run pytest tests/test_pipelines.py -q
  -p no:randomly` gives 11 passing and 67 failing: `pyml_classifier` (6
  variants), `pyml_inference` (executorch, candle), `pyml_llm` (llamacpp),
  `pyml_vlm`, `pyml_embedding` (clip, dinov2). Every one of the 67 failures is
  `PIPELINE_TIMEOUT` firing at 30 s while the model loads, so the split
  measures the card rather than g2g support. Getting the real answer needs the
  same suite run under `gst` at the same timeout and diffed: a pipeline that
  passes on gst and fails on g2g is a genuine gap, one that fails on both is the
  environment. The full suite takes 10-18 minutes and wants the GPU, so do not
  run the two backends at once on a 6 GB card.

- **Eleven elements have no per-frame seam, so they cannot run on g2g at all.**
  `alert`, `tracker`, `vad`, `clap`, `overlay_counter`, `kafkasink`,
  `streammux`, `streamdemux`, `coalescehistory` and `llm_remote` subclass a
  GStreamer base directly. `stablediffusion` is hosted but fills in neither
  `process_frames` nor `process_payload`. Reparenting a family onto one of those
  two seams in `backend/core.py` is what makes its pipelines runnable.
  `overlay_counter` inherits `overlay`, which the launcher rewrites to g2g's
  native `analyticsoverlay`, so the plain overlay line works regardless.

- **A hosted element's properties are only checked once its class loads.** The
  g2g host takes any name it does not read itself and hands it to the Python
  class, which is the only thing that knows the real set, so a typo fails at
  pipeline start rather than at parse. `gst-inspect` on `pyelement` lists the
  host's own properties and says the rest come from the class.

## Elements

- **`WhisperSpeechTTS.do_generate_speech` returns a `(1, n)` array**, which
  `soundfile` rejects with `LibsndfileError: Format not recognised`, so the
  element emits no audio. `CoquiTTS` returns 1-D and is fine. Pre-existing on
  both backends.

- **`AnomalyEngine._transform` is assigned only in `do_load_model`**, so
  `_get_transform` raises `AttributeError` on an engine whose model never
  loaded. Pre-existing on both backends.
