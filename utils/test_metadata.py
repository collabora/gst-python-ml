import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GLib, GstBase
import struct

# Initialize GStreamer
Gst.init(None)


class Metadata:
    SIGNATURE = b"GST-PYTHON-ML"

    def __init__(self, format_string: str):
        """
        Initialize with a format string:
        - "ifs" -> single struct with int, float, string
        - "l(ifs)" -> list of structs, each with int, float, string
        """
        self.format_string = format_string
        if format_string.startswith("l(") and format_string.endswith(")"):
            self.is_list = True
            self.struct_format = format_string[2:-1]  # Extract "ifs" from "l(ifs)"
        else:
            self.is_list = False
            self.struct_format = format_string

        # Parse the struct format
        self.fixed_fields = [c for c in self.struct_format if c != "s"]
        self.string_count = self.struct_format.count("s")
        self.fixed_size = (
            struct.calcsize("".join(self.fixed_fields)) if self.fixed_fields else 0
        )
        self.meta_bytes = None

    def write(self, buffer: Gst.Buffer, *values) -> None:
        """Add metadata as a Gst.Memory chunk at the end of the buffer."""
        try:
            if self.is_list:
                if not isinstance(values[0], (list, tuple)):
                    raise ValueError(
                        f"Expected a list for format '{self.format_string}', got {type(values[0])}"
                    )
                items = values[0]
                # Pack list length
                meta_bytes = struct.pack("I", len(items))
                # Pack each struct
                for item in items:
                    if len(item) != len(self.struct_format):
                        raise ValueError(
                            f"Struct length mismatch: expected {len(self.struct_format)}, got {len(item)}"
                        )
                    fixed_values = [
                        v for v, f in zip(item, self.struct_format) if f != "s"
                    ]
                    string_values = [
                        v for v, f in zip(item, self.struct_format) if f == "s"
                    ]
                    fixed_bytes = (
                        struct.pack("".join(self.fixed_fields), *fixed_values)
                        if fixed_values
                        else b""
                    )
                    string_bytes = b""
                    for s in string_values:
                        s_bytes = str(s).encode("utf-8")
                        string_bytes += struct.pack("I", len(s_bytes)) + s_bytes
                    meta_bytes += fixed_bytes + string_bytes
            else:
                if len(values) != len(self.struct_format):
                    raise ValueError(
                        f"Value length mismatch: expected {len(self.struct_format)}, got {len(values)}"
                    )
                fixed_values = [
                    v for v, f in zip(values, self.struct_format) if f != "s"
                ]
                string_values = [
                    v for v, f in zip(values, self.struct_format) if f == "s"
                ]
                fixed_bytes = (
                    struct.pack("".join(self.fixed_fields), *fixed_values)
                    if fixed_values
                    else b""
                )
                string_bytes = b""
                for s in string_values:
                    s_bytes = str(s).encode("utf-8")
                    string_bytes += struct.pack("I", len(s_bytes)) + s_bytes
                meta_bytes = fixed_bytes + string_bytes

            # Combine with signature and format string
            self.meta_bytes = (
                self.SIGNATURE
                + struct.pack("I", len(self.format_string))
                + self.format_string.encode("utf-8")
                + meta_bytes
            )

            meta_memory = Gst.Memory.new_wrapped(
                Gst.MemoryFlags.READONLY,
                self.meta_bytes,
                len(self.meta_bytes),
                0,
                len(self.meta_bytes),
                None,
            )
            buffer.append_memory(meta_memory)
            print(f"Metadata added, size: {len(self.meta_bytes)} bytes")
        except struct.error as e:
            raise ValueError(f"Failed to pack values: {str(e)}") from e

    def read(self, buffer: Gst.Buffer) -> tuple:
        """Read metadata from the last memory chunk."""
        n_mem = buffer.n_memory()
        if n_mem < 2:
            raise ValueError(
                f"No metadata memory chunk found (expected at least 2 chunks, got {n_mem})"
            )

        meta_memory = buffer.peek_memory(n_mem - 1)
        with meta_memory.map(Gst.MapFlags.READ) as map_info:
            data_bytes = bytes(map_info.data)
            if not data_bytes.startswith(self.SIGNATURE):
                raise ValueError(
                    f"Invalid metadata signature (expected {self.SIGNATURE.hex()}, got {data_bytes[:len(self.SIGNATURE)].hex()})"
                )

            offset = len(self.SIGNATURE)
            fmt_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
            offset += 4
            fmt_str = data_bytes[offset : offset + fmt_len].decode("utf-8")
            offset += fmt_len

            if fmt_str != self.format_string:
                raise ValueError(
                    f"Format mismatch: expected {self.format_string}, got {fmt_str}"
                )

            if self.is_list:
                # Read list length
                list_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                offset += 4
                result = []
                for _ in range(list_len):
                    fixed_values = []
                    if self.fixed_fields:
                        fixed_bytes = data_bytes[offset : offset + self.fixed_size]
                        fixed_values = list(
                            struct.unpack("".join(self.fixed_fields), fixed_bytes)
                        )
                        offset += self.fixed_size
                    string_values = []
                    for _ in range(self.string_count):
                        str_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                        offset += 4
                        string_values.append(
                            data_bytes[offset : offset + str_len].decode("utf-8")
                        )
                        offset += str_len
                    # Combine fixed and string values in struct order
                    struct_values = []
                    fixed_idx = 0
                    string_idx = 0
                    for f in self.struct_format:
                        if f == "s":
                            struct_values.append(string_values[string_idx])
                            string_idx += 1
                        else:
                            struct_values.append(fixed_values[fixed_idx])
                            fixed_idx += 1
                    result.append(tuple(struct_values))
                print(f"Read metadata: {tuple(result)}")
                return tuple(result)
            else:
                fixed_values = []
                if self.fixed_fields:
                    fixed_bytes = data_bytes[offset : offset + self.fixed_size]
                    fixed_values = list(
                        struct.unpack("".join(self.fixed_fields), fixed_bytes)
                    )
                    offset += self.fixed_size
                string_values = []
                for _ in range(self.string_count):
                    str_len = struct.unpack("I", data_bytes[offset : offset + 4])[0]
                    offset += 4
                    string_values.append(
                        data_bytes[offset : offset + str_len].decode("utf-8")
                    )
                    offset += str_len
                # Combine fixed and string values in original order
                result = []
                fixed_idx = 0
                string_idx = 0
                for f in self.struct_format:
                    if f == "s":
                        result.append(string_values[string_idx])
                        string_idx += 1
                    else:
                        result.append(fixed_values[fixed_idx])
                        fixed_idx += 1
                print(f"Read metadata: {tuple(result)}")
                return tuple(result)


class MetadataAdder(GstBase.BaseTransform):
    __gstmetadata__ = (
        "MetadataAdder",
        "Transform",
        "Adds custom metadata to video buffers",
        "Your Name",
    )

    _srctemplate = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw"),
    )
    _sinktemplate = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw"),
    )

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    def __init__(self):
        super().__init__()
        self.metadata = Metadata("l(ifs)")  # List of structs: int, float, string
        self.frame_count = 0
        self.set_in_place(True)  # Transform in-place

    def do_set_caps(self, in_caps, out_caps):
        """Accept and forward caps."""
        print(f"Setting caps: {in_caps.to_string()} -> {out_caps.to_string()}")
        return True

    def do_transform_ip(self, buffer):
        """Transform in-place: add metadata to each buffer."""
        self.frame_count += 1
        # Example list of structs
        struct_list = [
            (self.frame_count, 3.14, "hello"),
            (self.frame_count + 1, 6.28, "world"),
        ]
        self.metadata.write(buffer, struct_list)
        print(
            f"Frame {self.frame_count}: Buffer size: {buffer.get_size()}, Chunks: {buffer.n_memory()}"
        )
        return Gst.FlowReturn.OK


# Register the custom element
Gst.Element.register(None, "metadataadder", Gst.Rank.NONE, MetadataAdder)


# Create the pipeline
def create_pipeline():
    pipeline = Gst.Pipeline.new("video-test-pipeline")
    src = Gst.ElementFactory.make("videotestsrc", "source")
    transform = Gst.ElementFactory.make("metadataadder", "transform")
    sink = Gst.ElementFactory.make("glimagesink", "sink")

    if not all([src, transform, sink]):
        raise RuntimeError("Failed to create pipeline elements")

    pipeline.add(src)
    pipeline.add(transform)
    pipeline.add(sink)

    if not src.link(transform):
        raise RuntimeError("Failed to link videotestsrc to metadataadder")
    if not transform.link(sink):
        raise RuntimeError("Failed to link metadataadder to glimagesink")

    src.set_property("num-buffers", 100)
    src.set_property("pattern", "smpte")

    return pipeline


# Bus message handler
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, Debug: {debug}")
        loop.quit()
    return True


# Probe to check metadata
def sink_pad_probe(pad, info, metadata):
    buffer = info.get_buffer()
    print(f"Sink probe: Buffer size: {buffer.get_size()}, Chunks: {buffer.n_memory()}")
    try:
        values = metadata.read(buffer)
        print(f"Sink metadata: {values}")
    except ValueError as e:
        print(f"Sink metadata error: {e}")
    return Gst.PadProbeReturn.OK


# Main function
def main():
    pipeline = create_pipeline()
    metadata = Metadata("l(ifs)")  # List of structs for testing

    # Add probe on sink pad
    sink = pipeline.get_by_name("sink")
    sink_pad = sink.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, sink_pad_probe, metadata)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    bus.connect("message", on_bus_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted")

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
