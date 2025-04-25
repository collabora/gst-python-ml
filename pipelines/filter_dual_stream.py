#!/usr/bin/env python3
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init([])

# Create pipeline
pipeline_desc = """
  filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! pyml_streammux name=mux
  filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.
  mux. ! pyml_llmstreamfilter name=filter device=cuda:0 downsampled_width=320 downsampled_height=240 num-streams=2 llm-model-name=microsoft/phi-2 caption-file=data/sample_captions.json prompt="Choose the {n} most interesting captions from the following list:\n{captions}" ! pyml_streamdemux name=demux
  demux.src_0 ! queue ! videoconvert ! textoverlay name=overlay0 ! videoconvert ! glimagesink sync=false
  demux.src_1 ! queue ! videoconvert ! textoverlay name=overlay1 ! videoconvert ! glimagesink sync=false
  tee name=t ! queue ! overlay0.text_sink
  t.src_1 ! queue ! overlay1.text_sink
"""
pipeline = Gst.parse_launch(pipeline_desc)

# Get elements
filter_elem = pipeline.get_by_name("filter")
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


# Bus message handler for errors and state changes
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
        old_state, new_state, pending = message.parse_state_changed()
        if message.src == pipeline:
            print(
                f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}"
            )
    return True


# Connect pad-added signal to filter element
filter_elem.connect("pad-added", on_pad_added, tee)

# Set up bus to catch errors and state changes
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message, GLib.MainLoop())

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)
    loop.quit()
