#!/usr/bin/env python3
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init([])

# Create pipeline
pipeline_desc = """
  filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! pyml_streammux name=mux
  filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.
  mux. ! pyml_llmstreamfilter name=filter device=cuda:0 downsampled_width=320 downsampled_height=240 num-streams=2 prompt="Choose the {n} most interesting captions from the following list:\n{captions}" ! pyml_streamdemux name=demux
  demux.src_0 ! queue ! videoconvert ! textoverlay name=overlay0 ! videoconvert ! glimagesink sync=false
  demux.src_1 ! queue ! videoconvert ! textoverlay name=overlay1 ! videoconvert ! glimagesink sync=false
  tee name=t ! queue ! overlay0.text_sink
  t.src_1 ! queue ! overlay1.text_sink
"""
pipeline = Gst.parse_launch(pipeline_desc)

# Get elements
filter = pipeline.get_by_name("filter")
tee = pipeline.get_by_name("t")

# Function to link text_src to tee when text_src is created
def on_pad_added(element, pad, tee):
    if pad.get_name() == "text_src":
        print(f"Linking {pad.get_name()} to tee:sink")
        sink_pad = tee.get_static_pad("sink")
        if not sink_pad:
            sink_pad = tee.get_request_pad("sink_%u")
        ret = pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            print(f"Failed to link text_src to tee: {ret}")
        else:
            print("Successfully linked text_src to tee")

# Connect pad-added signal to filter element
filter.connect("pad-added", on_pad_added, tee)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)
    loop.quit()