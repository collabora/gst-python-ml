# LLM Remote — networked LLM via HTTP (Ollama, OpenAI-compatible endpoints)
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GstBase
    from backend import GObject

    from log.logger_factory import LoggerFactory
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_llm_remote' element will not be available. Error {e}"
    )


class LlmRemote(GstBase.Aggregator):
    """
    GStreamer element that sends text to a remote LLM endpoint (Ollama, OpenAI-compatible)
    via HTTP and pushes the response downstream.

    Properties:
      url:           HTTP endpoint URL (default: http://localhost:11434/api/generate)
      model-name:    Model name to request from the server (default: llama3)
      system-prompt: Optional system prompt
      temperature:   Sampling temperature (default: 0.7)
      timeout:       HTTP request timeout in seconds (default: 120)

    Example (Ollama):
      gst-launch-1.0 filesrc location=prompt.txt ! "text/x-raw,format=utf8" \
        ! pyml_llm_remote url=http://localhost:11434/api/generate model-name=llama3 \
        ! fakesink
    """

    __gstmetadata__ = (
        "LLM Remote",
        "Transform",
        "Sends text to a remote LLM endpoint via HTTP (Ollama, OpenAI-compatible)",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("text/x-raw,format=utf8"),
        ),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("text/x-raw,format=utf8"),
        ),
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.__url = "http://localhost:11434/api/generate"
        self.__model_name = "llama3"
        self.__system_prompt = None
        self.__temperature = 0.7
        self.__timeout = 120
        self.segment_pushed = False

    @GObject.Property(type=str, default="http://localhost:11434/api/generate")
    def url(self):
        "HTTP endpoint URL for the remote LLM server"
        return self.__url

    @url.setter
    def url(self, value):
        self.__url = value

    @GObject.Property(type=str, default="llama3")
    def model_name(self):
        "Model name to request from the remote server"
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value

    @GObject.Property(type=str)
    def system_prompt(self):
        "Optional system prompt for the LLM"
        return self.__system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.__system_prompt = value

    @GObject.Property(type=float, default=0.7, minimum=0.0, maximum=2.0)
    def temperature(self):
        "Sampling temperature (0.0 = deterministic, higher = more creative)"
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        self.__temperature = value

    @GObject.Property(type=int, default=120, minimum=1, maximum=3600)
    def timeout(self):
        "HTTP request timeout in seconds"
        return self.__timeout

    @timeout.setter
    def timeout(self, value):
        self.__timeout = value

    def _call_ollama_generate(self, prompt):
        """Call Ollama /api/generate endpoint."""
        import json
        import urllib.request

        payload = {
            "model": self.__model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.__temperature},
        }
        if self.__system_prompt:
            payload["system"] = self.__system_prompt

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.__url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.__timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        return body.get("response", "")

    def _call_openai_compatible(self, prompt):
        """Call OpenAI-compatible /v1/chat/completions endpoint."""
        import json
        import urllib.request

        messages = []
        if self.__system_prompt:
            messages.append({"role": "system", "content": self.__system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.__model_name,
            "messages": messages,
            "temperature": self.__temperature,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.__url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.__timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def do_generate(self, prompt):
        """Route to the appropriate API based on URL path."""
        if "/v1/chat/completions" in self.__url:
            return self._call_openai_compatible(prompt)
        return self._call_ollama_generate(prompt)

    def push_segment_if_needed(self):
        if not self.segment_pushed:
            segment = Gst.Segment()
            segment.init(Gst.Format.TIME)
            segment.start = 0
            segment.stop = Gst.CLOCK_TIME_NONE
            segment.position = 0

            self.srcpad.push_event(Gst.Event.new_segment(segment))
            self.segment_pushed = True

    def do_aggregate(self, timeout):
        if all(pad.is_eos() for pad in self.sinkpads):
            return Gst.FlowReturn.EOS
        self.push_segment_if_needed()
        self.process_all_sink_pads()
        self.selected_samples(Gst.CLOCK_TIME_NONE, 0, 0, None)
        return Gst.FlowReturn.OK

    def process_all_sink_pads(self):
        if len(self.sinkpads) == 0:
            return
        buf = self.sinkpads[0].pop_buffer()
        if buf:
            self.do_process(buf)

    def do_process(self, buf):
        """Read input text, send to remote LLM, push response downstream."""
        try:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map buffer")
                return Gst.FlowReturn.ERROR

            input_text = bytes(map_info.data).decode("utf-8")
            self.logger.info(f"Sending to remote LLM: {input_text[:100]}...")
            buf.unmap(map_info)

            generated_text = self.do_generate(input_text)
            self.logger.info(f"Remote LLM response: {generated_text[:100]}...")

            return self.push_generated_text(buf, generated_text)

        except Exception as e:
            self.logger.error(f"Error in remote LLM processing: {e}")
            return Gst.FlowReturn.ERROR

    def push_generated_text(self, inbuf, generated_text):
        """Push the generated text downstream."""
        try:
            generated_bytes = generated_text.encode("utf-8")
            outbuf = Gst.Buffer.new_allocate(None, len(generated_bytes), None)
            success, map_info_out = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                self.logger.error("Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            map_info_out.data[: len(generated_bytes)] = generated_bytes
            outbuf.unmap(map_info_out)
            outbuf.pts = inbuf.pts
            outbuf.dts = inbuf.dts
            outbuf.duration = inbuf.duration

            self.logger.info("Pushed generated text downstream")
            return self.srcpad.push(outbuf)

        except Exception as e:
            self.logger.error(f"Error pushing generated text: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_llm_remote", LlmRemote)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_llm_remote' element will not be registered because required modules are missing."
    )
