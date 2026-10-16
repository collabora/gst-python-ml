import concurrent.futures
import gc
import sys
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

THREAD_COUNT = 32
ATTEMPT_COUNT = 2_000
PAD_NAME = "src"

Gst.init(None)
element = Gst.ElementFactory.make("fakesrc")
start = threading.Barrier(THREAD_COUNT)


def get_pad() -> list[object]:
    start.wait()
    return [element.get_static_pad(PAD_NAME) for _ in range(ATTEMPT_COUNT)]


print("gil", sys._is_gil_enabled(), "pygobject", gi.__version__)
with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
    pads = [
        pad
        for thread_pads in executor.map(lambda _: get_pad(), range(THREAD_COUNT))
        for pad in thread_pads
    ]
print("wrapper_count", len({id(pad) for pad in pads}))
del pads
gc.collect()
print("pad", element.get_static_pad(PAD_NAME))
