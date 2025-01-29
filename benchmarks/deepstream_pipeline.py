import configparser
import re
import gi
import argparse
import os
import sys
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def clean_config(config_path):
    """ Preprocess the config file to remove inline comments. """
    cleaned_lines = []
    with open(config_path, "r") as f:
        for line in f:
            cleaned_line = re.sub(r"#.*", "", line).strip()  # Remove inline comments
            if cleaned_line:  # Ignore empty lines
                cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)

def parse_config(config_path):
    """ Parses DeepStream config file and generates a GStreamer pipeline string. """
    config_str = clean_config(config_path)  # Remove inline comments
    config = configparser.ConfigParser()
    config.read_string(config_str)  # Read cleaned config from string

    pipeline_parts = []

    # 1️⃣ Get the absolute directory path of the config file
    config_dir = os.path.dirname(config_path)

    # 2️⃣ Handle Video Source (File, Camera, RTSP)
    for section in config.sections():
        if section.startswith("source"):
            source_type = int(config[section]["type"])
            if source_type == 1:  # Camera
                device = config[section].get("camera-v4l2-dev-node", "/dev/video0")
                pipeline_parts.append(f"v4l2src device={device} ! videoconvert")
            elif source_type == 2:  # File source
                uri = config[section].get("uri", "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4")
                pipeline_parts.append(f"filesrc location={uri} ! decodebin")
            elif source_type == 3:  # RTSP Stream
                uri = config[section].get("uri", "rtsp://localhost:8554/stream")
                pipeline_parts.append(f"rtspsrc location={uri} latency=100 ! decodebin")

    # 3️⃣ Stream Muxer
    if "streammux" in config:
        width = config["streammux"].get("width", "1280")
        height = config["streammux"].get("height", "720")
        batch_size = config["streammux"].get("batch-size", "1")
        pipeline_parts.append(f"nvstreammux width={width} height={height} batch-size={batch_size} ! videoconvert")

    # 4️⃣ Primary Inference (GIE - GPU Inference Engine)
    if "primary-gie" in config:
        config_file = config["primary-gie"].get("config-file")

        # Convert to full absolute path
        full_config_path = os.path.join(config_dir, config_file)
        if not os.path.isfile(full_config_path):
            print(f"❌ Error: Inference config file not found: {full_config_path}")
            sys.exit(1)

        pipeline_parts.append(f"nvinfer config-file-path={full_config_path}")

    # 5️⃣ Overlay (OSD)
    if "osd" in config and config["osd"].get("enable", "0") == "1":
        pipeline_parts.append("nvdsosd")

    # 6️⃣ Multiple Sinks (FakeSink, Display, File Output, RTSP)
    for section in config.sections():
        if section.startswith("sink"):
            if config[section].get("enable", "0") == "1":
                sink_type = int(config[section]["type"])
                if sink_type == 1:  # FakeSink (No display)
                    pipeline_parts.append("fakesink")
                elif sink_type == 2:  # EGLSink (Display)
                    pipeline_parts.append("nveglglessink")
                elif sink_type == 3:  # File Output
                    output_file = config[section].get("output-file", "out.mp4")
                    pipeline_parts.append(f"filesink location={output_file}")
                elif sink_type == 4:  # RTSP Streaming
                    rtsp_port = config[section].get("rtsp-port", "8554")
                    pipeline_parts.append(f"rtspclientsink location=rtsp://localhost:{rtsp_port}/stream")

    return " ! ".join(pipeline_parts)

def run_pipeline(config_path):
    """ Runs the dynamically generated DeepStream GStreamer pipeline and keeps it alive. """
    pipeline_str = parse_config(config_path)
    
    if not pipeline_str:
        print("❌ Error: Empty pipeline. Check config file!")
        sys.exit(1)

    print("\n🚀 **Generated GStreamer Pipeline:**")
    print(pipeline_str, "\n")  # Print before running

    # Create the pipeline
    pipeline = Gst.parse_launch(pipeline_str)
    
    if not pipeline:
        print("❌ Error: Invalid GStreamer pipeline.")
        sys.exit(1)

    # Set to PLAYING
    pipeline.set_state(Gst.State.PLAYING)

    # Run pipeline loop (ensures it stays running)
    bus = pipeline.get_bus()
    loop = GLib.MainLoop()
    
    def bus_call(bus, message, loop):
        """ GStreamer message handler to keep pipeline running. """
        t = message.type
        if t == Gst.MessageType.EOS:
            print("✅ End of Stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ Error: {err}, Debug Info: {debug}")
            loop.quit()
        return True

    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("🎬 **Pipeline Running... Press Ctrl+C to stop.**\n")
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(0)

# Allow users to specify a DeepStream config file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepStream Config to GStreamer Pipeline Generator")
    parser.add_argument("config_file", help="Path to the DeepStream configuration file")
    args = parser.parse_args()

    run_pipeline(args.config_file)
