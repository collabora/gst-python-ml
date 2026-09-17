# pyml-mcp
# Copyright (C) 2024-2026 Collabora Ltd.
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

import json
import os
import re
import sys
import tempfile
import threading
from collections import deque
from importlib.metadata import version
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

import pyml_launch

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
if pyml_launch.CHECKOUT_PLUGINS:
    os.environ["GST_PLUGIN_PATH"] = os.pathsep.join(
        filter(
            None, [str(pyml_launch.CHECKOUT_PLUGINS), os.environ.get("GST_PLUGIN_PATH")]
        )
    )
sys.path.insert(0, str(pyml_launch.plugin_dir()))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, GObject, Gst, GstVideo  # noqa: E402

Gst.init(None)

import metasink  # noqa: E402
import readme_pipelines  # noqa: E402
from alertrecorder import Clip, DEFAULT_ENCODER  # noqa: E402
from embedding_index import EmbeddingIndex  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402

BACKEND = os.environ.get("PYML_BACKEND", "gst").lower()
MODEL_DEVICE = os.environ.get("PYML_MCP_DEVICE", "")
VLM_MODEL = os.environ.get("PYML_MCP_VLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
ELEMENT_PREFIX = "pyml_"
RECENT_RECORDS = 1000
SNAPSHOT_IMAGE_FORMAT = "jpeg"
FRAME_CONVERT_SECONDS = 5
RGB_CAPS = "video/x-raw,format=RGB"
RGB_BYTES_PER_PIXEL = 3
# greedy, so the same frame captions the same way twice
VLM_TEMPERATURE = 0.0
CLIP_DECODE_PIPELINE = (
    "filesrc name=source ! decodebin ! videoconvert ! video/x-raw,format=I420 "
    "! appsink name=frames sync=false"
)
PREROLL_SECONDS = 10
CLIP_FINISH_SECONDS = 20
README_PATH = (
    pyml_launch.CHECKOUT_PLUGINS.parent / "README.md"
    if pyml_launch.CHECKOUT_PLUGINS
    else None
)

server = MCPServer(
    "gst-python-ml",
    version=version("gst-python-ml"),
    instructions=(
        "Pipelines are gst-launch descriptions. End one with pyml_metasink to read "
        "its detections, transcripts and alerts back through latest_metadata."
    ),
)


class Session:
    def __init__(self):
        self.pipeline = None
        self.records = deque(maxlen=RECENT_RECORDS)
        # the deque drops its oldest entries
        self.posted = 0
        self.errors = []
        self.ended = False
        self.lock = threading.Condition()

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.APPLICATION:
            structure = message.get_structure()
            if structure.get_name() == metasink.BUS_MESSAGE_NAME:
                record = json.loads(structure.get_string(metasink.BUS_MESSAGE_FIELD))
                with self.lock:
                    self.records.append(record)
                    self.posted += 1
                    self.lock.notify_all()
        elif message.type == Gst.MessageType.ERROR:
            error, _debug = message.parse_error()
            with self.lock:
                self.errors.append(f"{message.src.get_name()}: {error.message}")
                self.lock.notify_all()
        elif message.type == Gst.MessageType.EOS:
            with self.lock:
                self.ended = True
                self.lock.notify_all()
        # nothing else pops this bus
        return Gst.BusSyncReply.DROP


session = Session()


def running_pipeline():
    if session.pipeline is None:
        raise ToolError("no pipeline is running, call start_pipeline first")
    return session.pipeline


def metadata_sinks(pipeline):
    for element in pipeline.iterate_elements():
        if element.get_factory().get_name() == metasink.MetaSink.GST_PLUGIN_NAME:
            yield element


def named_element(name):
    element = running_pipeline().get_by_name(name)
    if element is None:
        raise ToolError(f"the pipeline has no element named {name!r}")
    return element


def serialized(spec, value):
    if value is None:
        return ""
    # an enum reads as its nick, as on a gst-launch line
    if spec.value_type == GObject.TYPE_STRING:
        return value
    holder = GObject.Value(spec.value_type)
    holder.set_value(value)
    spelled = Gst.value_serialize(holder)
    return str(value) if spelled is None else spelled


@server.tool(
    description="Start a pipeline from a gst-launch description, stopping any running one first. "
    "End it with pyml_metasink so its results reach latest_metadata."
)
def start_pipeline(pipeline: str) -> dict:
    stop_pipeline()
    try:
        parsed = Gst.parse_launch(pipeline)
    except GLib.Error as error:
        raise ToolError(error.message) from error
    session.__init__()
    session.pipeline = parsed
    # records already arrive over the bus
    for sink in metadata_sinks(parsed):
        if not sink.get_property("location"):
            sink.set_property("location", os.devnull)
    parsed.get_bus().set_sync_handler(session.on_message)
    if parsed.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        errors = ", ".join(session.errors) or "the pipeline refused to start"
        stop_pipeline()
        raise ToolError(errors)
    return pipeline_status()


@server.tool(
    description="The running pipeline's state, whether it reached end of stream, its errors, "
    "and how many metadata records it has posted."
)
def pipeline_status() -> dict:
    if session.pipeline is None:
        return {"state": "none"}
    _result, state, _pending = session.pipeline.get_state(0)
    with session.lock:
        return {
            "state": state.value_nick,
            "ended": session.ended,
            "errors": list(session.errors),
            "records": len(session.records),
        }


@server.tool(description="Stop and release the running pipeline.")
def stop_pipeline() -> dict:
    if session.pipeline is not None:
        session.pipeline.set_state(Gst.State.NULL)
        session.pipeline = None
    return {"state": "none"}


@server.tool(
    description="The newest records pyml_metasink posted, oldest first: each has a pts in "
    "seconds plus detections, text, or a blob such as alert or depth."
)
def latest_metadata(count: int = 10) -> list[dict]:
    with session.lock:
        return list(session.records)[-count:]


@server.tool(
    description="Load the JSON lines a pyml_metasink wrote to a file as the current records, "
    "so latest_metadata and wait_for_records read a finished run with no pipeline running."
)
def load_metadata(path: str) -> dict:
    if not os.path.isfile(path):
        raise ToolError(f"no metadata file at {path!r}")
    stop_pipeline()
    session.__init__()
    with open(path) as lines:
        for line in lines:
            if not line.strip():
                continue
            session.records.append(json.loads(line))
            session.posted += 1
    # nothing more will arrive
    session.ended = True
    return {"records": session.posted}


def records_posted_since(posted_before, key):
    fresh = min(session.posted - posted_before, len(session.records))
    recent = list(session.records)[len(session.records) - fresh :]
    if not key:
        return recent
    return [record for record in recent if key in record]


@server.tool(
    description="Wait for pyml_metasink to post new records, oldest first, and return them "
    "with the pipeline status. Give a key such as detections or alert to wait only for "
    "records carrying it. Returns whatever arrived when the pipeline ends, fails, or "
    "the timeout in seconds passes first."
)
def wait_for_records(count: int = 1, key: str = "", timeout: float = 30) -> dict:
    # a run loaded from a file has no pipeline behind it
    if not session.ended:
        running_pipeline()
    with session.lock:
        # an ended run posts nothing more
        posted_before = 0 if session.ended else session.posted
        errors_before = len(session.errors)

        def settled():
            return (
                len(records_posted_since(posted_before, key)) >= count
                or session.ended
                or len(session.errors) > errors_before
            )

        session.lock.wait_for(settled, timeout)
        matched = records_posted_since(posted_before, key)
    return {"records": matched, "status": pipeline_status()}


def newest_frame(element):
    if element:
        target = named_element(element)
    else:
        target = next(metadata_sinks(running_pipeline()), None)
        if target is None:
            raise ToolError("the pipeline has no pyml_metasink to snapshot")
    name = target.get_name()
    if target.find_property("last-sample") is None:
        raise ToolError(f"{name} has no last-sample property")
    sample = target.get_property("last-sample")
    if sample is None:
        raise ToolError(f"no frame has reached {name} yet")
    return name, sample


def converted_frame(name, sample, caps):
    try:
        return GstVideo.video_convert_sample(
            sample, Gst.Caps.from_string(caps), FRAME_CONVERT_SECONDS * Gst.SECOND
        )
    except GLib.Error as error:
        raise ToolError(f"{name} is not carrying video: {error.message}") from error


@server.tool(
    description="The newest frame a pyml_metasink rendered, as a JPEG image. Names a sink of "
    "the running pipeline, or the first pyml_metasink when left empty."
)
def snapshot_frame(element: str = "") -> Image:
    name, sample = newest_frame(element)
    buffer = converted_frame(
        name, sample, f"image/{SNAPSHOT_IMAGE_FORMAT}"
    ).get_buffer()
    return Image(
        data=buffer.extract_dup(0, buffer.get_size()), format=SNAPSHOT_IMAGE_FORMAT
    )


def frame_image(sample):
    import numpy
    from PIL import Image as PillowImage

    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    buffer = sample.get_buffer()
    rows = numpy.frombuffer(buffer.extract_dup(0, buffer.get_size()), dtype=numpy.uint8)
    packed = width * RGB_BYTES_PER_PIXEL
    stride = GstVideo.VideoInfo.new_from_caps(caps).stride[0]
    if stride != packed:
        rows = rows.reshape(height, stride)[:, :packed]
    return PillowImage.fromarray(rows.reshape(height, width, RGB_BYTES_PER_PIXEL))


vlm_engines = {}


def vlm_engine(model_name):
    engine = vlm_engines.get(model_name)
    if engine is None:
        # torch loads only once a caption asks for it
        from engine.vlm_engine import VlmEngine

        engine = VlmEngine()
        engine.do_set_device(model_device())
        engine.do_load_model(model_name)
        vlm_engines[model_name] = engine
    return engine


@server.tool(
    description="Caption the newest frame a pyml_metasink rendered with a vision-language "
    "model, answering the prompt about it. Names a sink of the running pipeline, or the "
    "first pyml_metasink when left empty. On cpu a caption takes tens of seconds."
)
def describe_frame(
    prompt: str = "Describe this image.", element: str = "", max_tokens: int = 64
) -> str:
    name, sample = newest_frame(element)
    image = frame_image(converted_frame(name, sample, RGB_CAPS))
    return vlm_engine(VLM_MODEL).do_generate(
        image, prompt, None, max_tokens, VLM_TEMPERATURE
    )


def model_device():
    if MODEL_DEVICE:
        return MODEL_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


text_embedding_engines = {}


def text_embedding_engine(model_name):
    engine = text_embedding_engines.get(model_name)
    if engine is None:
        # torch loads only once a search asks for it
        from engine.embedding_engine import EmbeddingEngine

        engine = EmbeddingEngine()
        engine.do_set_device(model_device())
        engine.do_load_model(model_name)
        text_embedding_engines[model_name] = engine
    return engine


@server.tool(
    description="Search a video index written by pyml_embeddingsink for the frames a description "
    "matches, closest first: each result has a pts in seconds, the source_id of its "
    "stream, and a cosine similarity score."
)
def search_video(query: str, index: str, count: int = 5) -> list[dict]:
    if not os.path.isfile(index):
        raise ToolError(f"no embedding index at {index!r}")
    opened = EmbeddingIndex.open(index)
    try:
        model_name = opened.model_name()
        if not model_name:
            raise ToolError(f"the index at {index!r} is empty")
        vector = text_embedding_engine(model_name).do_text_embedding(query)
        if vector is None:
            raise ToolError(f"{model_name} cannot embed text, index with a CLIP model")
        return opened.search(vector, count)
    finally:
        opened.close()


def decode_failure(pipeline, source):
    message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
    if message is None:
        return f"cannot decode {source!r}"
    return message.parse_error()[0].message


@server.tool(
    description="Cut the seconds of video around a pts out of a video file into a webm, so a "
    "search_video hit becomes a clip. Returns where it was written, the range it covers "
    "and how many frames it holds."
)
def clip_at(source: str, pts: float, seconds: float = 4.0, location: str = "") -> dict:
    if not os.path.isfile(source):
        raise ToolError(f"no video at {source!r}")
    start = max(pts - seconds / 2, 0)
    end = start + seconds
    path = location or str(
        Path(tempfile.gettempdir()) / f"{Path(source).stem}-{start:.2f}.webm"
    )
    decode = Gst.parse_launch(CLIP_DECODE_PIPELINE)
    # a launch line would split the path on spaces
    decode.get_by_name("source").set_property("location", source)
    frames = decode.get_by_name("frames")
    decode.set_state(Gst.State.PAUSED)
    result, _state, _pending = decode.get_state(PREROLL_SECONDS * Gst.SECOND)
    if result == Gst.StateChangeReturn.FAILURE:
        reason = decode_failure(decode, source)
        decode.set_state(Gst.State.NULL)
        raise ToolError(reason)
    seeked = decode.seek(
        1.0,
        Gst.Format.TIME,
        Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
        Gst.SeekType.SET,
        int(start * Gst.SECOND),
        Gst.SeekType.SET,
        int(end * Gst.SECOND),
    )
    if not seeked:
        decode.set_state(Gst.State.NULL)
        raise ToolError(f"cannot seek {source!r}")
    decode.set_state(Gst.State.PLAYING)
    clip = None
    frame_count = 0
    while True:
        sample = frames.emit("pull-sample")
        if sample is None:
            break
        buffer = sample.get_buffer()
        if clip is None:
            logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
            clip = Clip(path, DEFAULT_ENCODER, sample.get_caps(), buffer.pts, logger)
        clip.push(buffer)
        frame_count += 1
    decode.set_state(Gst.State.NULL)
    if clip is None:
        raise ToolError(f"no frames in {source!r} between {start} and {end}")
    clip.finish().join(CLIP_FINISH_SECONDS)
    return {"path": path, "start": start, "end": end, "frames": frame_count}


@server.tool(
    description="Set a property on a named element of the running pipeline, the value spelled "
    "as it would be on a gst-launch line."
)
def set_property(element: str, property: str, value: str) -> dict:
    target = named_element(element)
    if target.find_property(property) is None:
        raise ToolError(f"{element} has no property {property!r}")
    Gst.util_set_object_arg(target, property, value)
    return {property: get_property(element, property)}


@server.tool(description="Read a property of a named element of the running pipeline.")
def get_property(element: str, property: str) -> str:
    target = named_element(element)
    spec = target.find_property(property)
    if spec is None:
        raise ToolError(f"{element} has no property {property!r}")
    return serialized(spec, target.get_property(property))


@server.tool(description="The gst-python-ml elements this GStreamer can load.")
def list_elements() -> list[dict]:
    factories = Gst.Registry.get().get_feature_list(Gst.ElementFactory)
    return [
        {
            "name": factory.get_name(),
            "description": factory.get_metadata(Gst.ELEMENT_METADATA_DESCRIPTION),
        }
        for factory in sorted(factories, key=lambda factory: factory.get_name())
        if factory.get_name().startswith(ELEMENT_PREFIX)
    ]


def initial_value(instance, spec):
    # a fresh instance's value stands in for the default, as in gst-inspect
    if not spec.flags & GObject.ParamFlags.READABLE:
        return None
    return serialized(spec, instance.get_property(spec.name))


@server.tool(
    description="An element's description and properties, for any element gst-launch knows."
)
def inspect(element: str) -> dict:
    factory = Gst.ElementFactory.find(element)
    if factory is None:
        raise ToolError(f"no element named {element!r}")
    instance = factory.create(None)
    return {
        "name": element,
        "description": factory.get_metadata(Gst.ELEMENT_METADATA_DESCRIPTION),
        "properties": [
            {
                "name": spec.name,
                "type": spec.value_type.name,
                "blurb": spec.blurb,
                "default": initial_value(instance, spec),
            }
            for spec in instance.list_properties()
        ],
    }


def prompt_name(heading):
    return re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")


def section_prompt(heading, descriptions, repository):
    def readme_section():
        opening = (
            f"These are the {heading} pipelines from the gst-python-ml README. "
            "start_pipeline takes each line below as written, swap a display sink such as "
            "autovideosink for pyml_metasink to read the results back, and file paths are "
            f"relative to {repository}."
        )
        return "\n".join([opening, *descriptions])

    return readme_section


def register_readme_prompts(readme_path):
    sections = readme_pipelines.pipelines_by_section(readme_path)
    for heading, descriptions in sections.items():
        server.prompt(
            name=prompt_name(heading),
            description=f"The README pipelines under {heading}",
        )(section_prompt(heading, descriptions, readme_path.parent))


# an installed wheel has no README beside the plugins
if README_PATH is not None:
    register_readme_prompts(README_PATH)


def main():
    if BACKEND != "gst":
        raise SystemExit(
            "pyml-mcp runs the gst backend in-process; for g2g use glass2glass's g2g-mcp"
        )
    server.run()


if __name__ == "__main__":
    main()
