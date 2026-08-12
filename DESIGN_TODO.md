# DESIGN_TODO

A terse catalogue of open tasks only. Gaps on the host side of the glass2glass
Python-element host are tracked in that repo's `DESIGN_TODO.md`, under
"Python-element host", not here.

## g2g backend coverage

- **How many README pipelines run under `PYML_BACKEND=g2g` needs measuring.**
  Run `tests/test_pipelines.py` under each backend and compare: one that passes
  on gst and fails on g2g is a gap, one that fails on both is the environment.
  Only the error categories count as gaps. `pipeline error: Hardware(Other)` is
  how g2g reports a hosted element raising, so each needs its log in
  `tests/logs` read to name the cause. Known so far: `pyml_kafkasink` calls
  `Gst.Pad` APIs directly and dies on `Gst.init`, `demo_soccer`'s engine raises
  `TypeError: MLEngine.__init__() got an unexpected keyword argument 'device'`,
  and `pyml_streammux` is refused with `pyelement: more than one input links
  here, but it is not a registered muxer`. The suite wants the GPU for about 20
  minutes per backend, so run one backend at a time on a 6 GB card and leave the
  machine otherwise idle, including between backends: a model still resident
  from the previous run fails the next one at preroll.

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
