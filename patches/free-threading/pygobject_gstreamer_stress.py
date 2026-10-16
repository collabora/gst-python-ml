import concurrent.futures
import sys
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

PIPELINE_COUNT = 8
BUFFER_COUNT = 50_000
BUS_TIMEOUT = 60 * Gst.SECOND
PIPELINE_DESCRIPTION = (
    f"fakesrc num-buffers={BUFFER_COUNT} sizetype=fixed sizemax=1 "
    "! fakesink name=sink signal-handoffs=true sync=false"
)

Gst.init(None)
start = threading.Barrier(PIPELINE_COUNT)


def run_pipeline() -> int:
    pipeline = Gst.parse_launch(PIPELINE_DESCRIPTION)
    sink = pipeline.get_by_name("sink")
    handoff_count = 0

    def count_handoff(_sink: object, _buffer: object, _pad: object) -> None:
        nonlocal handoff_count
        handoff_count += 1

    sink.connect("handoff", count_handoff)
    start.wait()
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        BUS_TIMEOUT, Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
    pipeline.set_state(Gst.State.NULL)
    if message is None or message.type != Gst.MessageType.EOS:
        raise AssertionError(message)
    return handoff_count


print("gil", sys._is_gil_enabled(), "pygobject", gi.__version__)
with concurrent.futures.ThreadPoolExecutor(max_workers=PIPELINE_COUNT) as executor:
    counts = list(executor.map(lambda _: run_pipeline(), range(PIPELINE_COUNT)))
print("counts", counts)
