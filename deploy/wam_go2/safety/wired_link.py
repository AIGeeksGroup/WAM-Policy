"""Is the LowState stream really arriving over the robot cable?

Hard rule: host and robot talk over the Ethernet cable only. If the robot is
also on a WiFi network, an unbound DDS participant lets it stream /lowstate
to the host's WiFi address instead; frames are then lost in 20-250 ms bursts
and the controller runs open loop. Binding DDS to the wired NIC (launch file /
demo scripts) is the fix; this
monitor is an auxiliary check: while LowState frames are flowing, the
wired NIC's receive counter must grow at the stream's rate (500 Hz x
~1.3 kB ~= 0.65 MB/s). The node refuses STAND / POLICY while it
does not. Aggregate NIC bytes cannot establish the source of an individual
topic; DDS must still be bound to the wired interface.

Pure Python: reads /sys/class/net/<nic>/statistics/rx_bytes (or a fake tree
for tests).
"""

import os


class WiredLinkMonitor:
    """update(now, state_count) once per control tick.

    ok is None until the first evaluation with frames flowing, then True /
    False; reason explains a False. A NIC that does not exist is False.
    `nic` empty or "none" disables the check (ok is always True).
    """

    def __init__(self, nic, min_mbps=0.3, period=1.0, min_frames=50,
                 sysfs="/sys/class/net"):
        self.nic = "" if str(nic).strip().lower() in ("", "none", "off") else str(nic).strip()
        self.min_mbps = float(min_mbps)
        self.period = float(period)
        self.min_frames = int(min_frames)
        self._path = os.path.join(sysfs, self.nic, "statistics", "rx_bytes") if self.nic else None
        self._t0 = None
        self._bytes0 = None
        self._count0 = None
        self.ok = True if not self.nic else None
        self.reason = "" if not self.nic else "wired link not verified yet"
        self.rate_mbps = 0.0

    @property
    def enabled(self):
        return bool(self.nic)

    def _read_bytes(self):
        try:
            with open(self._path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def update(self, now, state_count):
        if not self.nic:
            return
        b = self._read_bytes()
        if b is None:
            self.ok = False
            self.reason = f"wired NIC {self.nic} not found ({self._path})"
            self._t0 = None
            return
        if self._t0 is None:
            self._t0, self._bytes0, self._count0 = now, b, state_count
            return
        dt = now - self._t0
        if dt < self.period:
            return
        frames = state_count - self._count0
        self.rate_mbps = (b - self._bytes0) / 1e6 / dt
        self._t0, self._bytes0, self._count0 = now, b, state_count
        if frames < self.min_frames:
            return          # no stream to judge yet (nothing received)
        if self.rate_mbps >= self.min_mbps:
            self.ok = True
            self.reason = ""
        else:
            self.ok = False
            self.reason = (f"LowState frames arrive ({frames / dt:.0f} Hz) but {self.nic} "
                           f"receives only {self.rate_mbps:.2f} MB/s (< {self.min_mbps:.2f}): the "
                           "stream is NOT on the robot cable (WiFi?) - bind DDS to the wired NIC "
                           "(FASTRTPS_DEFAULT_PROFILES_FILE=config/fastdds_wired.xml / "
                           "CYCLONEDDS_URI) or turn the host WiFi off")
