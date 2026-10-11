# mlx-whisper
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
