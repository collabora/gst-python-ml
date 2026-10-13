# DESIGN_TODO

A terse catalogue of open tasks only. Gaps whose fix sits in glass2glass are
tracked in that repo's `design/TODO.md`, not here.

## g2g backend coverage

- **24 PIPELINES.md pipelines pass on gst and fail under `PYML_BACKEND=g2g`.**
  Measured with `tests/test_pipelines.py` on 2026-10-13: gst 74 passed, g2g 51,
  26 skipped on both. A pass on gst and a fail on g2g is a gap, a fail on both is
  the environment. The suite holds the GPU for about 20 minutes per backend, so
  run one backend at a time with the machine otherwise idle: a model still
  resident from the previous run fails the next one at preroll. `G2G_CAPS_TRACE=1`
  prints the caps solve, `RUST_LOG` does nothing. The gaps fixed on this side:
  - The launcher has no rewrite for the gst CUDA chain `nvh264dec ! cudaconvert
    ! cudascale`, so the yolo and classifier CUDA pipelines fail with `unknown
    element: cudaconvert`. g2g spells it `nvdec`, `cudaupload !
    cudaconvertscale`, `cudadownload`.
  - `pyml_yolo engine-name=onnx` keeps a view of the frame after the hook
    returns: `BufferError: g2g_process retained a frame buffer view past return`.
  - `pyml_yolo_pose` with the default `visualize` reads `self.sinkpad`, which a
    hosted element does not have. `visualize=false` passes.
  - `initial_prompt = "..."` with spaces round `=` is accepted by gst and refused
    by the g2g host as `malformed property 'initial_prompt'`.
  - `pipewiresrc do-timestamp=true` has no g2g property, a candidate for
    `DEFAULTED_PROPERTIES`.
  - `pyml_overlay meta-path=` has no counterpart on `analyticsoverlay`, so the
    launcher refuses the pipeline.
  - `pyml_caption_qwen` with `cap.text_src` and `overlay.text_sink` is wired
    wrongly: the hosted element has one output and the launcher ignores pad
    names, so textdigest receives video and textoverlay is treated as a muxer.
  - `pyml_streammux` is refused with `pyelement: more than one input links here,
    but it is not a registered muxer`, and `pyml_streamdemux` fails with
    CapsMismatch. Waits on native `streammux` and `streamdemux` in glass2glass,
    then `NATIVE_EQUIVALENTS` entries for both.
  - The elements in the next bullet.

- **Seven elements have no per-frame seam, so they cannot run on g2g at all.**
  `overlay_counter`, `kafkasink`, `streammux`, `streamdemux`, `coalescehistory`
  and `llm_remote` subclass a GStreamer base directly. `stablediffusion` is
  hosted but fills in neither `process_frames` nor `process_payload`, so it fails
  with `PyTransform produces no output for text/x-raw`. Reparenting a family onto
  one of those two seams in `backend/core.py` is what makes its pipelines
  runnable. `overlay_counter` inherits `overlay`, which the launcher rewrites to
  g2g's native `analyticsoverlay`, so the plain overlay line works regardless.

- **`g2g-mcp` takes a g2g-spelled pipeline only**, so a gst-spelled one has to
  go through the launcher's rewrite before it can be started there.

- **A hosted element's properties are only checked once its class loads.** The
  g2g host takes any name it does not read itself and hands it to the Python
  class, which is the only thing that knows the real set, so a typo fails at
  pipeline start rather than at parse. `gst-inspect` on `pyelement` lists the
  host's own properties and says the rest come from the class.
