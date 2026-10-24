# LLM Remote — networked LLM via HTTP (Ollama, OpenAI-compatible endpoints)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend
from backend import GObject

CAN_REGISTER_ELEMENT = True
try:
    from base_llm import BaseLlm
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_llm_remote' element will not be available. Error {e}"
    )

DEFAULT_MODEL_NAME = "llama3"
OPENAI_CHAT_PATH = "/v1/chat/completions"


class LlmRemote(BaseLlm):
    __gstmetadata__ = (
        "LLM Remote",
        "Transform",
        "Sends text to a remote LLM endpoint via HTTP (Ollama, OpenAI-compatible)",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    url = GObject.Property(
        type=str,
        default="http://localhost:11434/api/generate",
        blurb="HTTP endpoint URL for the remote LLM server",
    )
    temperature = GObject.Property(
        type=float,
        default=0.7,
        minimum=0.0,
        maximum=2.0,
        blurb="Sampling temperature (0.0 = deterministic, higher = more creative)",
    )
    timeout = GObject.Property(
        type=int,
        default=120,
        minimum=1,
        maximum=3600,
        blurb="HTTP request timeout in seconds",
    )

    def __init__(self):
        super().__init__()
        self.model_name = DEFAULT_MODEL_NAME
        # the server runs the model
        self.engine_name = None

    def do_load_model(self):
        pass

    def process_payload(self, payload: bytes) -> list[bytes]:
        input_text = payload.decode("utf-8")
        self.logger.info(f"Sending to remote LLM: {input_text[:100]}...")
        if OPENAI_CHAT_PATH in self.url:
            generated_text = self._call_openai_compatible(input_text)
        else:
            generated_text = self._call_ollama_generate(input_text)
        self.logger.info(f"Remote LLM response: {generated_text[:100]}...")
        return [generated_text.encode("utf-8")]

    def _post_json(self, payload):
        import json
        import urllib.request

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call_ollama_generate(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt
        return self._post_json(payload).get("response", "")

    def _call_openai_compatible(self, prompt):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        choices = self._post_json(payload).get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_llm_remote", LlmRemote)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_llm_remote' element will not be registered because required modules are missing."
    )
