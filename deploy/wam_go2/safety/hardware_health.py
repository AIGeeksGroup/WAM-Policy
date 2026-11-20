"""Decode documented Go2 hardware faults without ROS or mutable state.

Source: the Unitree Go2 developer documentation (LowState.bit_flag and
MotorState.reserve[0]). The aggregate DDS stream can remain fresh while a
motor, MCU or battery has timed out.
"""

from ..constants import RAMBO_JOINT_NAMES


_LOWSTATE_FAULTS = (
    (0x80, "motor timeout"),
    (0x40, "MCU timeout"),
    (0x10, "battery timeout"),
    (0x02, "board current fault"),
)

_MOTOR_FAULTS = (
    (1 << 0, "overcurrent"),
    (1 << 1, "overvoltage"),
    (1 << 2, "driver overheat"),
    (1 << 3, "bus undervoltage"),
    (1 << 4, "winding overheat"),
    (1 << 5, "encoder fault"),
    (1 << 8, "communication interrupted"),
)


def hardware_fault_reason(bit_flag, motor_errors):
    """Return documented faults, or an empty string for healthy input.

    ``motor_errors`` contains the twelve uint32 codes in RAMBO joint order,
    as returned by ``parse_lowstate``. Full raw codes are retained in the
    diagnostic, including unknown bits, but only documented fault bits trip.

    LowState 0x04 means *not charging*, 0x08 is reserved, and 0x01 is the
    command timeout commonly seen before taking ownership; none is a fault
    here, and neither is 0x20 (wireless remote timeout: this deployment does
    not use the remote). Motor bits 6/7 and all undocumented upper bits are
    also ignored.
    """
    flags = int(bit_flag)
    reasons = []
    status = [label for mask, label in _LOWSTATE_FAULTS if flags & mask]
    if status:
        reasons.append(f"LowState bit_flag=0x{flags:02x} ({', '.join(status)})")
    for index, raw_code in enumerate(motor_errors):
        code = int(raw_code)
        errors = [label for mask, label in _MOTOR_FAULTS if code & mask]
        if errors:
            reasons.append(
                f"motor RAMBO[{index}] {RAMBO_JOINT_NAMES[index]} "
                f"reserve[0]=0x{code:08x} ({', '.join(errors)})")
    return "; ".join(reasons)
