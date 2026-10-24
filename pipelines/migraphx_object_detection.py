#!/usr/bin/env python3
"""Sample pipeline demonstrating MiGraphX inference on AMD GPUs.

Runs YOLO11m object detection using AMD's MiGraphX graph inference engine.
Requires: ROCm, migraphx package, and a YOLO11m ONNX model.

Export a YOLO11 model to ONNX:
    yolo export model=yolo11m.pt format=onnx

Usage:
    python pipelines/migraphx_object_detection.py
"""
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init([])

pipeline_desc = """
  filesrc location=data/people.mp4 ! decodebin name=d
  d. ! queue ! videoconvert ! videoscale
  ! video/x-raw,format=RGB,width=640,height=640
  ! mp_objectdetector engine-name=migraphx model-name=yolo11m.onnx device=gpu
        input-format=nchw post-process=anchor_free
  ! videoconvert ! video/x-raw,format=RGBA
  ! mp_overlay ! videoconvert ! autovideosink
"""

pipeline = Gst.parse_launch(pipeline_desc)


def on_bus_message(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, Debug: {debug}")
        loop.quit()
    elif mtype == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif mtype == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old, new, _ = message.parse_state_changed()
            print(f"Pipeline: {old.value_nick} -> {new.value_nick}")
    return True


bus = pipeline.get_bus()
bus.add_signal_watch()
loop = GLib.MainLoop()
bus.connect("message", on_bus_message, loop)

pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
