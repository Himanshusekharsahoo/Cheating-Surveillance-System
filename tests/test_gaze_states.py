import pytest
from eye_movement import (
    GAZE_CENTER,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    GAZE_DOWN,
    GAZE_NO_FACE,
    GAZE_UNCERTAIN,
    COMPLIANT_GAZE_STATES
)
from head_pose import (
    HEAD_SCREEN,
    HEAD_LEFT,
    HEAD_RIGHT,
    HEAD_UP,
    HEAD_DOWN,
    HEAD_TILTED,
    HEAD_NO_FACE,
    COMPLIANT_HEAD_STATES
)

pytestmark = pytest.mark.unit

def test_documented_gaze_constants_distinct():
    """Verify that all gaze states are distinct, non-empty string constants."""
    states = [GAZE_CENTER, GAZE_LEFT, GAZE_RIGHT, GAZE_UP, GAZE_DOWN, GAZE_NO_FACE, GAZE_UNCERTAIN]
    assert len(states) == len(set(states))
    for s in states:
        assert isinstance(s, str) and len(s) > 0

def test_compliant_gaze_states():
    """Verify that center gaze is compliant and off-screen gaze/no-face/uncertain are NOT compliant."""
    assert GAZE_CENTER in COMPLIANT_GAZE_STATES
    assert GAZE_LEFT not in COMPLIANT_GAZE_STATES
    assert GAZE_RIGHT not in COMPLIANT_GAZE_STATES
    assert GAZE_UP not in COMPLIANT_GAZE_STATES
    assert GAZE_DOWN not in COMPLIANT_GAZE_STATES
    assert GAZE_NO_FACE not in COMPLIANT_GAZE_STATES
    assert GAZE_UNCERTAIN not in COMPLIANT_GAZE_STATES

def test_documented_head_constants_distinct():
    """Verify that all head pose states are distinct, non-empty string constants."""
    states = [HEAD_SCREEN, HEAD_LEFT, HEAD_RIGHT, HEAD_UP, HEAD_DOWN, HEAD_TILTED, HEAD_NO_FACE]
    assert len(states) == len(set(states))
    for s in states:
        assert isinstance(s, str) and len(s) > 0

def test_compliant_head_states():
    """Verify that looking at screen is compliant, while deviations and no-face are NOT."""
    assert HEAD_SCREEN in COMPLIANT_HEAD_STATES
    assert HEAD_LEFT not in COMPLIANT_HEAD_STATES
    assert HEAD_RIGHT not in COMPLIANT_HEAD_STATES
    assert HEAD_UP not in COMPLIANT_HEAD_STATES
    assert HEAD_DOWN not in COMPLIANT_HEAD_STATES
    assert HEAD_TILTED not in COMPLIANT_HEAD_STATES
    assert HEAD_NO_FACE not in COMPLIANT_HEAD_STATES
