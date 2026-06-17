#!/usr/bin/env python3
# Run a video through the ONNX (fp16) football pipeline.
#
#   detector (onnx) -> pyml_tracker -> pyml_football_overlay
#
# Usage:
#   python demo/football/onnx_loop.py INPUT.mp4             # live display, looping
#   python demo/football/onnx_loop.py INPUT.mp4 OUTPUT.mp4  # write annotated mp4
# (self-contained: finds the repo venv + plugins and re-execs into them)
import os
import sys
import glob

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENV = os.path.join(REPO, ".venv")
MODEL = os.path.join(REPO, "models/football/football_fp16.onnx")
os.environ["GST_PLUGIN_PATH"] = (
    os.path.join(REPO, "plugins") + os.pathsep + os.environ.get("GST_PLUGIN_PATH", "")
)
if not os.environ.get("_ONNX_LOOP_REEXEC") and os.path.isdir(VENV):
    os.environ["VIRTUAL_ENV"] = VENV
    os.environ["PATH"] = (
        os.path.join(VENV, "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    libs = sorted(
        set(
            glob.glob(
                os.path.join(
                    VENV, "lib", "python*", "site-packages", "nvidia", "*", "lib"
                )
            )
        )
    )
    if libs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*libs, os.environ.get("LD_LIBRARY_PATH", "")]
        )
    os.environ["_ONNX_LOOP_REEXEC"] = "1"
    pybin = os.path.join(VENV, "bin", "python")
    exe = pybin if os.path.exists(pybin) else sys.executable
    os.execv(exe, [exe, *sys.argv])

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

Gst.init(None)


def on_message(bus, message, loop, pipeline, do_loop):
    t = message.type

    if t == Gst.MessageType.EOS:
        if do_loop:
            # Display mode: seek back to the start to loop the clip.
            print("Looping...")
            if not pipeline.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            ):
                print("Failed to seek back to start", file=sys.stderr)
                loop.quit()
        else:
            # mp4 mode: end of file, the muxer has finalized the file.
            loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"ERROR: {err}", file=sys.stderr)
        if debug:
            print(f"DEBUG: {debug}", file=sys.stderr)
        loop.quit()


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} INPUT.mp4 [OUTPUT.mp4]", file=sys.stderr)
        print(
            "  no OUTPUT -> live display (looping); OUTPUT -> write annotated mp4",
            file=sys.stderr,
        )
        sys.exit(1)
    video = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None

    # Shared detection + overlay chain. Feed the ORIGINAL resolution:
    # pyml_objectdetector letterboxes to the model's 640 internally for
    # inference and maps boxes back, so the overlay stays full-res.
    chain = (
        f"filesrc location={video} ! "
        "decodebin ! videoconvert ! video/x-raw,format=RGB ! "
        "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
        "pyml_objectdetector engine-name=onnx "
        f"  model-name={MODEL} device=cuda:0 "
        "  input-format=nchw post-process=anchor_free interval=1 "
        "  confidence=0.1 nms-iou=0.7 ! "
        "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
        "pyml_tracker tracker-type=bytetrack new-track-confidence=0.25 ! "
        "videoconvert ! video/x-raw,format=RGBA ! "
        "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
        "pyml_football_overlay class-names=ball,goalkeeper,player,referee "
        "  team-colors=true trails=false show-ids=false show-labels=false "
        "  draw-from-detections=true min-confidence=0 merge-iou=0.5 "
        "  position-smoothing=0.7 highlight-focal=false ! "
    )
    if out:
        pipeline_description = (
            chain + "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 ! "
            "videoconvert ! openh264enc ! h264parse ! mp4mux ! "
            f"filesink location={out}"
        )
        do_loop = False
    else:
        # Pre-roll buffer absorbs inference jitter for smooth real-time display.
        pipeline_description = (
            chain + "queue max-size-buffers=600 max-size-time=0 max-size-bytes=0 "
            "  min-threshold-buffers=30 ! "
            "videoconvert ! autovideosink sync=true"
        )
        do_loop = True

    print(pipeline_description)
    print(f"writing -> {out}" if out else "live display (looping)")

    try:
        pipeline = Gst.parse_launch(pipeline_description)
    except GLib.Error as e:
        print(f"Failed to create pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop, pipeline, do_loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        if out:
            # Finalize the mp4 on Ctrl-C: send EOS and wait for the muxer to
            # flush its trailer, otherwise the file is left unplayable.
            pipeline.send_event(Gst.Event.new_eos())
            bus.timed_pop_filtered(
                5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
    finally:
        pipeline.set_state(Gst.State.NULL)
    if out:
        print(f"Done: {out}")


if __name__ == "__main__":
    main()
