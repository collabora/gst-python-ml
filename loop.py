# Loop
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

# Script to loop an arbitrary pipeline.
# Pass in command line for pipeline and this pipeline will then loop indefinitely

# Example: python loop.py "filesrc location=data/people.mp4 ! decodebin ! videoconvert  ! videoconvert ! autovideosink"

# Example WAYLAND_DISPLAY=wayland-1 XDG_RUNTIME_DIR=/run/user/1000 python loop.py "filesrc location=data/people.mp4 ! decodebin ! videoconvert  ! videoconvert ! autovideosink"


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
    if len(sys.argv) != 2:
        print("Usage: python script.py <gst_pipeline>")
        print(
            "Example: python script.py 'filesrc location=data/people.mp4 ! decodebin ! videoconvert ! autovideosink'"
        )
        return -1

    gst_pipeline = sys.argv[1]

    Gst.init(None)

    try:
        # Parse the pipeline
        pipeline = Gst.parse_launch(gst_pipeline)
    except Gst.ParseError as e:
        Gst.error(f"Failed to parse pipeline: {e}")
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
