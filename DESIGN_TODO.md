# DESIGN_TODO

A terse catalogue of open tasks only. Gaps on the host side of the glass2glass
Python-element host are tracked in that repo's `DESIGN_TODO.md`, under
"Python-element host", not here.

## g2g backend coverage

- **32 README pipelines run on gst and fail under `PYML_BACKEND=g2g`.**
  `tests/test_pipelines.py` at a 30 s timeout passes 48 of 79 on gst and 16 on
  g2g. 31 fail on both, which is the environment rather than g2g, and nothing
  passes on g2g that fails on gst. Most of the 32 report `pipeline error:
  Hardware(Other)`, which is how g2g reports a hosted element raising, so each
  one needs its log in `tests/logs` read to name the cause. `pyml_overlay` is in
  16 of them, the largest cluster. Two causes known: `pyml_kafkasink` calls
  `Gst.Pad` APIs directly and dies on `Gst.init`, and `demo_soccer`'s engine
  raises `TypeError: MLEngine.__init__() got an unexpected keyword argument
  'device'`. The suite takes about 17 minutes per backend and wants the GPU, so
  run one backend at a time on a 6 GB card, and keep the machine otherwise idle
  or the 30 s timeout starts measuring load instead.

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

- **An engine that fails to load its model keeps running with `model=None`**,
  so the first frame raises somewhere further on instead of naming what went
  wrong. The README caption line wants `gptqmodel` for its AWQ model; without it
  `CaptionQwen` logs the load failure, then dies on `captioning returned None`.
  Failing at load time would name the missing package.
