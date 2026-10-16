import concurrent.futures
import sys
import threading

import gi

gi.require_version("GLib", "2.0")
from gi.repository import GLib  # noqa: E402

THREAD_COUNT = 12
CALLS_PER_THREAD = 200_000
EXPECTED_FILENAME = "/tmp/pygobject-ft"
TEST_URI = "file:///tmp/pygobject-ft"

start = threading.Barrier(THREAD_COUNT)


def call_glib() -> int:
    start.wait()
    for call_index in range(CALLS_PER_THREAD):
        filename, hostname = GLib.filename_from_uri(TEST_URI)
        if filename != EXPECTED_FILENAME or hostname is not None:
            raise AssertionError((call_index, filename, hostname))
    return CALLS_PER_THREAD


print("gil", sys._is_gil_enabled(), "pygobject", gi.__version__)
with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
    completed = sum(executor.map(lambda _: call_glib(), range(THREAD_COUNT)))
print("completed", completed)
