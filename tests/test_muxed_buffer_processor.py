import os
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from log.logger_factory import LoggerFactory  # noqa: E402
from utils.metadata import Metadata  # noqa: E402
from utils.muxed_buffer_processor import MuxedBufferProcessor  # noqa: E402

WIDTH, HEIGHT = 4, 3
MUX_ID = "mux/demux"


class RgbPad:
    def get_current_caps(self):
        return Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={WIDTH},height={HEIGHT}"
        )


def muxed_buffer(fill_values):
    buf = Gst.Buffer.new()
    for fill in fill_values:
        frame = Gst.Buffer.new_allocate(None, WIDTH * HEIGHT * 3, None)
        frame.memset(0, fill, WIDTH * HEIGHT * 3)
        buf.append_memory(frame.get_memory(0))
    Metadata("si").write(buf, MUX_ID, len(fill_values))
    return buf


def extract(buf):
    processor = MuxedBufferProcessor(
        LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST), WIDTH, HEIGHT, 30, 1
    )
    return processor.extract_frames(buf, RgbPad())


def test_a_two_source_batch_is_stacked():
    frames, id_str, num_sources, fmt = extract(muxed_buffer([1, 2]))
    assert (frames.shape, id_str, num_sources, fmt) == (
        (2, HEIGHT, WIDTH, 3),
        MUX_ID,
        2,
        "RGB",
    )
    assert np.all(frames[0] == 1) and np.all(frames[1] == 2)


# the muxer flushes a partial batch of one source when the others hit eos first
def test_a_one_source_batch_keeps_the_single_frame_shape():
    frames, id_str, num_sources, fmt = extract(muxed_buffer([7]))
    assert (frames.shape, id_str, num_sources, fmt) == (
        (HEIGHT, WIDTH, 3),
        MUX_ID,
        1,
        "RGB",
    )
    assert np.all(frames == 7)
