# mlx-whisper
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .whisper_engine import OPENAI_WHISPER_PREFIX

MLX_WHISPER_PREFIX = "mlx-community/whisper-"
MLX_WHISPER_SUFFIX = "-mlx"


# any other id is taken as an mlx repo
def mlx_whisper_repo(model_name):
    if not model_name.startswith(OPENAI_WHISPER_PREFIX):
        return model_name
    size = model_name.removeprefix(OPENAI_WHISPER_PREFIX)
    return f"{MLX_WHISPER_PREFIX}{size}{MLX_WHISPER_SUFFIX}"


class MlxWhisper:
    def __init__(self, model_name):
        from huggingface_hub import snapshot_download

        self.repo = mlx_whisper_repo(model_name)
        snapshot_download(self.repo)

    def transcribe(self, audio, language, task, beam_size, initial_prompt):
        import mlx_whisper

        # mlx-whisper has no beam search
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=language or None,
            task=task,
            initial_prompt=initial_prompt or None,
        )
        return [segment["text"].strip() for segment in result["segments"]]
