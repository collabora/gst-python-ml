# whisper.cpp
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

from .whisper_engine import OPENAI_WHISPER_PREFIX

# pywhispercpp's params_sampling_strategy for beam search
BEAM_SEARCH_STRATEGY = 1
# pywhispercpp's default
BEAM_PATIENCE = -1.0
TRANSLATE_TASK = "translate"


def whisper_cpp_size(model_name):
    from pywhispercpp.constants import AVAILABLE_MODELS

    size = model_name.removeprefix(OPENAI_WHISPER_PREFIX)
    if not model_name.startswith(OPENAI_WHISPER_PREFIX) or size not in AVAILABLE_MODELS:
        raise ValueError(
            f"whisper.cpp takes {OPENAI_WHISPER_PREFIX}<size> with a size from "
            f"{', '.join(AVAILABLE_MODELS)}, got {model_name}"
        )
    return size


class WhisperCpp:
    def __init__(self, model_name, use_gpu):
        from pywhispercpp.model import Model

        self.model = Model(
            whisper_cpp_size(model_name),
            params_sampling_strategy=BEAM_SEARCH_STRATEGY,
            context_params={"use_gpu": use_gpu},
            print_progress=False,
        )

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        # pywhispercpp keeps each param across calls
        segments = self.model.transcribe(
            audio,
            language=language or "",
            translate=task == TRANSLATE_TASK,
            initial_prompt=initial_prompt or "",
            beam_search={"beam_size": beam_size, "patience": BEAM_PATIENCE},
        )
        return [segment.text.strip() for segment in segments]
