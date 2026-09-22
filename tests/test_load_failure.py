import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst  # noqa: E402

Gst.init(None)

from backend import VideoTransform  # noqa: E402
from engine.engine_factory import EngineFactory  # noqa: E402
from engine.ml_engine import MLEngine  # noqa: E402

FAILING_MODEL_NAME = "does-not-exist.pt"
LOAD_FAILURE = "no weights at does-not-exist.pt"
FAILING_ENGINE = "test_failing_engine"
WORKING_ENGINE = "test_working_engine"
LOADABLE_MODEL_NAME = "loads-fine.pt"
PIPELINE_TIMEOUT = 10 * Gst.SECOND


class FailingEngine(MLEngine):
    def do_load_model(self, model_name, **kwargs):
        raise FileNotFoundError(LOAD_FAILURE)

    def do_set_device(self, device):
        self.device = device

    def do_forward(self, frames):
        raise AssertionError("inference ran without a model")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise AssertionError("generation ran without a model")


class WorkingEngine(FailingEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model = model_name


class LoadFails(VideoTransform):
    __gstmetadata__ = (
        "Load Fails",
        "Transform",
        "an element whose model never loads",
        "test",
    )
    # pygobject reads the templates off the class's own dict, not the base's
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        EngineFactory.register(FAILING_ENGINE, FailingEngine)
        self.mgr.engine_name = FAILING_ENGINE

    def process_frames(self, frames, num_sources, fmt, target):
        raise AssertionError("a frame was processed without a model")


GObject.type_register(LoadFails)
Gst.Element.register(None, "loadfails", Gst.Rank.NONE, LoadFails)


def test_a_model_that_cannot_load_fails_the_pipeline_with_the_cause():
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=5 "
        "! video/x-raw,width=64,height=48,format=RGB "
        f"! loadfails model-name={FAILING_MODEL_NAME} "
        "! fakesink"
    )

    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None, "the pipeline neither errored nor finished"
    assert message.type == Gst.MessageType.ERROR, "the pipeline ran with no model"
    error, debug = message.parse_error()
    assert FAILING_MODEL_NAME in error.message
    assert LOAD_FAILURE in error.message
    assert "FileNotFoundError" in debug


def test_the_engine_keeps_no_model_after_a_failed_load():
    element = LoadFails()
    element.set_property("model-name", FAILING_MODEL_NAME)

    with pytest.raises(FileNotFoundError):
        element.do_load_model()

    assert element.engine.model is None
    assert element.engine.tokenizer is None


class LoadWorks(VideoTransform):
    __gstmetadata__ = (
        "Load Works",
        "Transform",
        "an element whose model loads",
        "test",
    )
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        EngineFactory.register(WORKING_ENGINE, WorkingEngine)
        self.mgr.engine_name = WORKING_ENGINE
        self.processed = 0

    def process_frames(self, frames, num_sources, fmt, target):
        self.processed += 1


GObject.type_register(LoadWorks)
Gst.Element.register(None, "loadworks", Gst.Rank.NONE, LoadWorks)


def test_a_model_that_loads_leaves_the_pipeline_running():
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=5 "
        "! video/x-raw,width=64,height=48,format=RGB "
        f"! loadworks name=element model-name={LOADABLE_MODEL_NAME} "
        "! fakesink"
    )
    element = pipeline.get_by_name("element")

    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None and message.type == Gst.MessageType.EOS
    assert element.processed == 5
    assert element.engine.model == LOADABLE_MODEL_NAME
