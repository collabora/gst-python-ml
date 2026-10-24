# VlmTask
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Backend-agnostic Vision-Language Model task.

`VlmTask` runs the VLM forward pass and serializes the generated text into the
payload that follows the GST-VLM: header. It uses only json and the engine; it
never touches the buffer. The backend element shell does the frame read and
appends the payload via the frameio facade, prepending the header.

Contract expected from the host element:
  self.engine        - the ML engine (provided by MLEngineMixin)
  self.prompt        - user prompt (a backend-declared property)
  self.system_prompt - optional system prompt (a backend-declared property)
  self.max_tokens    - max tokens to generate (a backend-declared property)
  self.temperature   - sampling temperature (a backend-declared property)
"""

import json


class VlmTask:
    """Inference + payload serialization, independent of any framework."""

    def forward(self, frame):
        return self.engine.do_forward(
            frame,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def decode(self, text):
        """Serialize a generated caption into ``(None, payload_bytes)``.

        ``payload_bytes`` is the JSON result that follows the GST-VLM: header;
        the shell prepends the header when appending the blob.
        """
        result = {"text": text}
        payload = json.dumps(result).encode("utf-8")
        return None, payload
