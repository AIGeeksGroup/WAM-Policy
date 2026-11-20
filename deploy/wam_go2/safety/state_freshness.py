"""Freshness of the LowState frame each 100 Hz control tick runs on.

Why this exists: when the default DDS participant also announces the host's
WiFi address, the robot may send /lowstate over WiFi. The control timer then
still runs at a steady 10 ms, but the LowState it consumes is unchanged for
3-17 consecutive ticks (30-170 ms) in bursts - 12-17 % of all ticks on the
real robot in the three-leg stand - i.e. the controller runs open loop for up
to a quarter of a gait cycle at a time, and roll excursions start right after
the longest stale bursts. The communication watchdog (lowstate_timeout) does
not fire because every burst is shorter. Over the cable the stream has 0.0 %
duplicates. DDS is bound to the wired NIC (config/fastdds_wired.xml,
scripts/demo/common.sh); this monitor makes the symptom visible in the log
and in the policy trace instead of silent, and the demo pre-flight checks the
bytes really arrive on the wired NIC.

Pure Python, no ROS: the node feeds it the interface's state_count once per
tick and logs whatever it returns.
"""


class StateFreshnessMonitor:
    """Counts control ticks that ran on an already-used LowState frame.

    update(state_count) once per control tick with the number of LowState
    accepted forward source ticks so far (not callback arrivals); a control
    tick is 'stale' when that count did not change
    since the previous tick. Returns a warning string at the end of every
    `window` seconds in which either the stale fraction reached
    `fraction_warn` or one stale run lasted at least `gap_warn` seconds,
    otherwise None. Totals since reset() are for the end-of-run summary.
    """

    def __init__(self, dt, window=2.0, gap_warn=0.05, fraction_warn=0.10):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.window_ticks = max(1, int(round(window / self.dt)))
        self.gap_warn_ticks = max(1, int(round(gap_warn / self.dt)))
        self.fraction_warn = float(fraction_warn)
        self._last_count = None
        self.last_stale = False
        self._w_ticks = self._w_stale = self._w_longest = self._w_gaps = 0
        self.reset()

    def reset(self):
        """Start a new run summary (POLICY entry). The frame counter and the
        warning window carry on."""
        self.total_ticks = 0
        self.total_stale = 0
        self.longest_run = 0      # [ticks]
        self.gaps = 0             # stale runs of at least gap_warn_ticks
        self._run = 0             # a run straddling the reset belongs to the new session only from here

    def update(self, state_count):
        fresh = self._last_count is None or state_count != self._last_count
        self._last_count = state_count
        self.last_stale = not fresh
        self.total_ticks += 1
        self._w_ticks += 1
        if fresh:
            if self._run >= self.gap_warn_ticks:
                self.gaps += 1
                self._w_gaps += 1
            self._run = 0
        else:
            self._run += 1
            self.total_stale += 1
            self._w_stale += 1
            self.longest_run = max(self.longest_run, self._run)
            self._w_longest = max(self._w_longest, self._run)

        msg = None
        if self._w_ticks >= self.window_ticks:
            frac = self._w_stale / self._w_ticks
            if frac >= self.fraction_warn or self._w_longest >= self.gap_warn_ticks:
                msg = (f"LowState stale on {100.0 * frac:.1f}% of the last "
                       f"{self._w_ticks * self.dt:.1f} s (longest gap "
                       f"{self._w_longest * self.dt * 1000.0:.0f} ms, {self._w_gaps} gaps >= "
                       f"{self.gap_warn_ticks * self.dt * 1000.0:.0f} ms): the controller ran on "
                       "an old frame - /lowstate frames lost or late on the DDS path (is the "
                       "stream on the wired NIC? check per-NIC rx bytes / ip neigh; a WiFi path "
                       "causes this)")
            self._w_ticks = self._w_stale = self._w_longest = self._w_gaps = 0
        return msg

    @property
    def stale_fraction(self):
        return self.total_stale / self.total_ticks if self.total_ticks else 0.0

    @property
    def gaps_including_open(self):
        """Long stale runs so far, counting one still in progress (a run
        that ends the POLICY session never sees the fresh tick that would
        close it)."""
        return self.gaps + (1 if self._run >= self.gap_warn_ticks else 0)

    def summary(self):
        if self.total_ticks == 0:
            return "no control ticks"
        return (f"{self.total_ticks} ticks, {self.total_stale} "
                f"({100.0 * self.stale_fraction:.1f}%) on a stale LowState, longest gap "
                f"{self.longest_run * self.dt * 1000.0:.0f} ms, {self.gaps_including_open} gaps >= "
                f"{self.gap_warn_ticks * self.dt * 1000.0:.0f} ms")
