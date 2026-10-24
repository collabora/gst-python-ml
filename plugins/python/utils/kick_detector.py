# KickDetector
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
from collections import deque
from typing import NamedTuple

SPEED_SMOOTHING_FRAMES = 2
# a ball unseen for longer than this has no speed to measure
MAX_BALL_GAP_FRAMES = 5
# a faster step is the detector jumping to another object
MAX_BALL_STEP = 3.0
# a single ball box is a few pixels and its size jitters
BALL_SIZE_SAMPLES = 25
FULL_TURN_DEGREES = 360
SPEED_DECIMALS = 2


class Kick(NamedTuple):
    frame: int
    player: int
    speed: float
    direction: int


class KickDetector:
    def __init__(self, speed_threshold, window_frames, cooldown_frames):
        self.speed_threshold = speed_threshold
        self.window_frames = window_frames
        self.cooldown_frames = cooldown_frames
        self._last_centre = None
        self._last_frame = None
        self._steps = deque(maxlen=SPEED_SMOOTHING_FRAMES)
        self._sizes = deque(maxlen=BALL_SIZE_SAMPLES)
        self._contact = None
        self._last_kick_frame = None

    def update(self, frame_index, ball_box, contact_player_id):
        if ball_box is None:
            return None
        speed, direction = self._ball_speed(frame_index, ball_box)
        slow = speed is None or speed <= self.speed_threshold
        if contact_player_id is not None and slow:
            self._contact = (contact_player_id, frame_index)
        if self._contact is None or slow:
            return None
        player, contact_frame = self._contact
        self._contact = None
        late = frame_index - contact_frame > self.window_frames
        if late or self._cooling_down(frame_index):
            return None
        self._last_kick_frame = frame_index
        return Kick(
            frame_index,
            player,
            round(speed, SPEED_DECIMALS),
            round(direction) % FULL_TURN_DEGREES,
        )

    def _cooling_down(self, frame_index):
        last = self._last_kick_frame
        return last is not None and frame_index - last < self.cooldown_frames

    def _ball_speed(self, frame_index, ball_box):
        x1, y1, x2, y2 = ball_box
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        previous, previous_frame = self._last_centre, self._last_frame
        self._last_centre, self._last_frame = centre, frame_index
        size = max(x2 - x1, y2 - y1)
        if size > 0:
            self._sizes.append(size)
        unmeasurable = (
            previous is None
            or frame_index - previous_frame > MAX_BALL_GAP_FRAMES
            or not self._sizes
        )
        if unmeasurable:
            self._steps.clear()
            return None, None
        dx = centre[0] - previous[0]
        dy = centre[1] - previous[1]
        ball_size = sorted(self._sizes)[len(self._sizes) // 2]
        step = math.hypot(dx, dy) / (frame_index - previous_frame) / ball_size
        if step > MAX_BALL_STEP:
            self._steps.clear()
            return None, None
        self._steps.append(step)
        speed = sum(self._steps) / len(self._steps)
        # screen y grows downwards
        direction = math.degrees(math.atan2(-dy, dx))
        return speed, direction
