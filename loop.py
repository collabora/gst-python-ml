import os
import gi
import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

os.environ["GST_DEBUG"] = "4"  # Enable GStreamer debug logging


def pad_added_handler(decodebin, pad, videoconvert):
    """Handle the pad-added signal from decodebin."""
    Gst.info("New pad added to decodebin. Attempting to link to videoconvert.")
    sink_pad = videoconvert.get_static_pad("sink")
    if not sink_pad.is_linked():
        result = pad.link(sink_pad)
        if result == Gst.PadLinkReturn.OK:
            Gst.info("Successfully linked decodebin pad to videoconvert.")
        else:
            Gst.error(f"Failed to link decodebin pad to videoconvert: {result}")
    else:
        Gst.warning("videoconvert sink pad is already linked.")


def bus_call(bus, msg, loop, pipeline):
    """Handle messages from the GStreamer bus."""
    if msg.type == Gst.MessageType.EOS:
        Gst.info("End of Stream reached. Seeking back to start...")
        # Seek to the beginning
        success = pipeline.seek_simple(
            Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
        )
        if not success:
            Gst.error("Seek operation failed. Quitting.")
            loop.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, debug_info = msg.parse_error()
        Gst.error(f"Error received from element {msg.src.get_name()}: {err.message}")
        Gst.error(f"Debugging information: {debug_info or 'none'}")
        loop.quit()


def main():
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_source> <sink_name>")
        print("Example: python script.py data/people.mp4 autovideosink")
        return -1

    file_source = sys.argv[1]
    sink_name = sys.argv[2]

    Gst.init(None)

    # Create GStreamer pipeline
    pipeline = Gst.Pipeline.new("video-pipeline")

    # Create elements
    source = Gst.ElementFactory.make("filesrc", "source")
    decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
    videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
    sink = Gst.ElementFactory.make(sink_name, "sink")

    # Validate element creation
    if not all([pipeline, source, decodebin, videoconvert, sink]):
        Gst.error("Failed to create GStreamer elements.")
        return -1

    # Configure elements
    source.set_property("location", file_source)

    # Add elements to the pipeline
    pipeline.add(source)
    pipeline.add(decodebin)
    pipeline.add(videoconvert)
    pipeline.add(sink)

    # Link source to decodebin
    if not source.link(decodebin):
        Gst.error("Failed to link source to decodebin.")
        return -1

    # Connect pad-added signal for decodebin
    decodebin.connect("pad-added", pad_added_handler, videoconvert)

    # Link videoconvert to sink
    if not videoconvert.link(sink):
        Gst.error("Failed to link videoconvert to sink.")
        return -1

    # Create a main loop
    loop = GLib.MainLoop()

    # Add a bus watch
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipeline)

    # Start playing the pipeline
    Gst.info("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        Gst.warning("Pipeline interrupted by user.")

    # Stop the pipeline
    Gst.info("Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    Gst.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
