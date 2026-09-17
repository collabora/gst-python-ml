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
import sys
import threading
from collections import deque

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

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
from gi.repository import GLib, GObject, Gst  # noqa: E402

Gst.init(None)

import metasink  # noqa: E402

BACKEND = os.environ.get("PYML_BACKEND", "gst").lower()
ELEMENT_PREFIX = "pyml_"
RECENT_RECORDS = 1000

server = MCPServer(
    "gst-python-ml",
    instructions=(
        "Pipelines are gst-launch descriptions. End one with pyml_metasink to read "
        "its detections, transcripts and alerts back through latest_metadata."
    ),
)


class Session:
    def __init__(self):
        self.pipeline = None
        self.records = deque(maxlen=RECENT_RECORDS)
        self.errors = []
        self.ended = False
        self.lock = threading.Lock()

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.APPLICATION:
            structure = message.get_structure()
            if structure.get_name() == metasink.BUS_MESSAGE_NAME:
                record = json.loads(structure.get_string(metasink.BUS_MESSAGE_FIELD))
                with self.lock:
                    self.records.append(record)
        elif message.type == Gst.MessageType.ERROR:
            error, _debug = message.parse_error()
            with self.lock:
                self.errors.append(f"{message.src.get_name()}: {error.message}")
        elif message.type == Gst.MessageType.EOS:
            self.ended = True
        # nothing else pops this bus
        return Gst.BusSyncReply.DROP


session = Session()


def running_pipeline():
    if session.pipeline is None:
        raise ToolError("no pipeline is running, call start_pipeline first")
    return session.pipeline


def named_element(name):
    element = running_pipeline().get_by_name(name)
    if element is None:
        raise ToolError(f"the pipeline has no element named {name!r}")
    return element


def serialized(spec, value):
    # an enum reads as its nick, as on a gst-launch line
    if spec.value_type == GObject.TYPE_STRING:
        return "" if value is None else value
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
    for element in parsed.iterate_elements():
        is_sink = element.get_factory().get_name() == metasink.MetaSink.GST_PLUGIN_NAME
        if is_sink and not element.get_property("location"):
            element.set_property("location", os.devnull)
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
                "default": serialized(spec, spec.get_default_value()),
            }
            for spec in instance.list_properties()
        ],
    }


def main():
    if BACKEND != "gst":
        raise SystemExit(
            "pyml-mcp runs the gst backend in-process; for g2g use glass2glass's g2g-mcp"
        )
    server.run()


if __name__ == "__main__":
    main()
