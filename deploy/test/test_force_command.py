"""The FL end-effector force command watchdog (wam_go2/control/force_command.py).

The force part of the native9 command (fl_ee_fx/fy/fz) reaches the controller on
rambo/ee_force_cmd; WAM-Policy always sends zeros. A stale request must not stay
applied, and a publisher that recovers after a timeout must withdraw (send zeros)
or the robot must stop before a force is passed again. This release ships no foot
contact-force estimate or contact gate for the force command.
"""
import math

import numpy as np
import pytest

from wam_go2.control.force_command import ForceCommandWatchdog, push_motion_phases

F = [14.0, 0.0, 0.0]
ZERO = [0.0, 0.0, 0.0]


def test_fresh_request_passes_straight_through():
    w = ForceCommandWatchdog(command_timeout=0.5)
    for moving in (True, False):
        out = w.step(F, moving, now=1.0, command_time=0.9)
        np.testing.assert_array_equal(out, F)
        assert not w.command_expired


def test_zero_request_is_zero_and_never_expires():
    w = ForceCommandWatchdog(command_timeout=0.5)
    out = w.step(ZERO, True, now=10.0, command_time=None)
    np.testing.assert_array_equal(out, ZERO)
    assert not w.command_expired


def test_stale_request_is_replaced_by_zero():
    w = ForceCommandWatchdog(command_timeout=0.5)
    out = w.step(F, True, now=1.6, command_time=1.0)
    np.testing.assert_array_equal(out, ZERO)
    assert w.command_expired


def test_request_without_any_message_counts_as_stale():
    w = ForceCommandWatchdog(command_timeout=0.5)
    np.testing.assert_array_equal(w.step(F, True, now=0.0, command_time=None), ZERO)
    np.testing.assert_array_equal(w.step(F, True, now=0.0, command_time=math.nan), ZERO)


def test_recovered_publisher_must_withdraw_before_force_returns():
    w = ForceCommandWatchdog(command_timeout=0.5)
    assert w.step(F, True, now=0.0, command_time=0.0)[0] == 14.0
    assert w.step(F, True, now=1.1, command_time=0.49)[0] == 0.0       # timed out
    for i in range(50):                                                 # fresh again, still pushing
        t = 1.11 + 0.01 * i
        assert w.step(F, True, now=t, command_time=t)[0] == 0.0
    assert w.step(ZERO, True, now=1.7, command_time=1.7)[0] == 0.0     # withdrawn
    assert w.step(F, True, now=1.71, command_time=1.71)[0] == 14.0     # passes again


def test_stopping_clears_the_withdrawal_requirement():
    w = ForceCommandWatchdog(command_timeout=0.5)
    w.step(F, True, now=1.0, command_time=0.0)                         # timed out while moving
    assert w.step(F, False, now=1.1, command_time=1.1)[0] == 14.0      # robot stopped, fresh request


def test_zero_timeout_disables_the_watchdog():
    w = ForceCommandWatchdog(command_timeout=0.0)
    np.testing.assert_array_equal(w.step(F, True, now=100.0, command_time=0.0), F)
    assert not w.command_expired


def test_reset_forgets_a_pending_withdrawal():
    w = ForceCommandWatchdog(command_timeout=0.5)
    w.step(F, True, now=1.0, command_time=0.0)
    w.reset()
    assert w.step(F, True, now=1.1, command_time=1.1)[0] == 14.0


def test_output_is_a_copy():
    w = ForceCommandWatchdog(command_timeout=0.5)
    req = np.array(F)
    out = w.step(req, True, now=0.0, command_time=0.0)
    out[0] = 99.0
    assert req[0] == 14.0


@pytest.mark.parametrize("request_", [[1.0, 2.0], [math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0]])
def test_invalid_request_is_rejected(request_):
    with pytest.raises(ValueError):
        ForceCommandWatchdog().step(request_, True, now=0.0, command_time=0.0)


@pytest.mark.parametrize("timeout", [-0.1, math.nan, math.inf])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError):
        ForceCommandWatchdog(command_timeout=timeout)


def test_push_phases_start_with_motion_and_end_with_zeros():
    walk, stop = push_motion_phases(0.15, 0.0, 2.0)
    assert walk.seconds == 2.0 and walk.velocity == (0.15, 0.0, 0.0) and walk.force == (0.0, 0.0, 0.0)
    assert stop.seconds == 0.5 and stop.velocity == (0.0, 0.0, 0.0) and stop.force == (0.0, 0.0, 0.0)
    (open_ended,) = push_motion_phases(0.15, 0.0, 0.0)
    assert open_ended.seconds == math.inf
