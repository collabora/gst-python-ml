import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "python"
sys.path.insert(0, str(PLUGIN_DIR))

from utils.kick_detector import KickDetector  # noqa: E402

BALL_SIZE = 10
BALL_Y = 300
SPEED_THRESHOLD = 0.8
WINDOW_FRAMES = 10
COOLDOWN_FRAMES = 25
KICKER = 7
SECOND_KICKER = 8
STILL_FRAMES = 5
FLIGHT_FRAMES = 6
# two ball sizes a frame, a firm pass
KICK_STEP = 2 * BALL_SIZE
DRIBBLE_STEP = 3
DRIBBLE_FRAMES = 40


def ball_box(x):
    return (x, BALL_Y, x + BALL_SIZE, BALL_Y + BALL_SIZE)


def still_then_kicked(x, kicker):
    path = [(x, kicker)] * STILL_FRAMES
    path += [(x + KICK_STEP * step, None) for step in range(1, FLIGHT_FRAMES + 1)]
    return path


def kicks(path, start_frame=1):
    detector = KickDetector(SPEED_THRESHOLD, WINDOW_FRAMES, COOLDOWN_FRAMES)
    found = []
    for frame, (x, contact) in enumerate(path, start=start_frame):
        kick = detector.update(frame, ball_box(x), contact)
        if kick is not None:
            found.append(kick)
    return found


def test_a_contact_then_a_fast_ball_is_one_kick_by_that_player():
    found = kicks(still_then_kicked(100, KICKER))
    assert len(found) == 1
    assert found[0].player == KICKER
    assert found[0].speed > SPEED_THRESHOLD
    assert found[0].direction == 0


def test_a_fast_ball_nobody_touched_is_no_kick():
    assert kicks(still_then_kicked(100, None)) == []


def test_a_second_kick_inside_the_cooldown_is_dropped():
    first = still_then_kicked(100, KICKER)
    second = still_then_kicked(first[-1][0], SECOND_KICKER)
    assert len(first + second) < COOLDOWN_FRAMES
    assert [kick.player for kick in kicks(first + second)] == [KICKER]


def test_a_second_kick_after_the_cooldown_counts():
    first = still_then_kicked(100, KICKER)
    resting = [(first[-1][0], None)] * COOLDOWN_FRAMES
    second = still_then_kicked(first[-1][0], SECOND_KICKER)
    found = kicks(first + resting + second)
    assert [kick.player for kick in found] == [KICKER, SECOND_KICKER]


def test_a_slow_dribble_is_no_kick():
    dribble = [(100 + DRIBBLE_STEP * step, KICKER) for step in range(DRIBBLE_FRAMES)]
    assert kicks(dribble) == []
