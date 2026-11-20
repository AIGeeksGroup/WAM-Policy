"""Fault masks traced to the Unitree Go2 developer documentation (LowState.bit_flag, MotorState.reserve[0])."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_go2.safety.hardware_health import hardware_fault_reason


@pytest.mark.parametrize("bit_flag,expected", [
    (0x80, "motor timeout"),
    (0x40, "MCU timeout"),
    (0x10, "battery timeout"),
    (0x02, "current fault"),
])
def test_each_documented_lowstate_fault_is_reported(bit_flag, expected):
    reason = hardware_fault_reason(bit_flag, np.zeros(12, dtype=np.uint32))
    assert "LowState" in reason
    assert f"0x{bit_flag:02x}" in reason
    assert expected in reason


@pytest.mark.parametrize("bit_flag", [0, 0x01, 0x04, 0x08, 0x0D, 0x20, 0x25])
def test_normal_and_ignored_lowstate_bits_do_not_fault(bit_flag):
    assert hardware_fault_reason(bit_flag, np.zeros(12, dtype=np.uint32)) == ""


def test_remote_timeout_never_hides_a_real_fault():
    no_errors = np.zeros(12, dtype=np.uint32)
    for bit in (0x80, 0x40, 0x10, 0x02):
        assert hardware_fault_reason(0x20 | bit, no_errors)
    errors = no_errors.copy()
    errors[0] = 0x100
    assert hardware_fault_reason(0x20, errors)


@pytest.mark.parametrize("bit,error_type", [
    (0, "overcurrent"),
    (1, "overvoltage"),
    (2, "driver overheat"),
    (3, "bus undervoltage"),
    (4, "winding overheat"),
    (5, "encoder fault"),
    (8, "communication interrupted"),
])
def test_each_documented_motor_error_keeps_joint_identity(bit, error_type):
    errors = np.zeros(12, dtype=np.uint32)
    errors[7] = 1 << bit  # RAMBO joint 7 = RR_thigh.
    reason = hardware_fault_reason(0x04, errors)
    assert "RAMBO[7]" in reason and "RR_thigh" in reason
    assert "reserve[0]" in reason and f"0x{1 << bit:08x}" in reason
    assert error_type in reason


@pytest.mark.parametrize("ignored", [0, 1 << 6, 1 << 7, 0xC0, 0xFFFFFE00])
def test_reserved_motor_bits_do_not_fault(ignored):
    errors = np.full(12, ignored, dtype=np.uint32)
    assert hardware_fault_reason(0x04, errors) == ""


def test_multiple_faults_preserve_full_codes_and_sources():
    errors = np.zeros(12, dtype=np.uint32)
    errors[0] = 0x800001C0  # Unknown upper/reserved bits plus bit 8.
    errors[11] = 0x15  # Official example: current, driver and winding heat.
    original = errors.copy()
    reason = hardware_fault_reason(0xA4, errors)
    for token in ("0xa4", "motor timeout",
                  "RAMBO[0]", "FL_hip", "0x800001c0", "communication interrupted",
                  "RAMBO[11]", "RR_calf", "0x00000015", "overcurrent",
                  "driver overheat", "winding overheat"):
        assert token in reason
    np.testing.assert_array_equal(errors, original)
