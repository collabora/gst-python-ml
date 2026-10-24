# whisper.cpp
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
