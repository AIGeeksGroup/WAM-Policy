"""Gait / contact schedule generator, single robot (numpy).

Produces the same contact schedule as the RAMBO training environment
(rambo_quadruped/modules/contact_generator.py, batched torch). Differences:
- single robot (no env batch);
- the last segment of each foot's sequence is extended indefinitely so the
  controller can run longer than the training episode (10 s): the mode index
  is clipped to the last entry, which keeps FL in permanent swing and the
  other legs cycling in phase mode, exactly as at the end of an episode.

Foot order everywhere: [FL, FR, RL, RR].
Sequence entries: [mode ("stance"|"phase"|"swing"), duration_s,
                   init_phase, gait_duration_s, stance_ratio]
(the last three are extra[0:3]; only used in "phase" mode).
"""

import numpy as np

MODE_STANCE = 1.0
MODE_PHASE = 0.0
MODE_SWING = -1.0

_MODE_VALUES = {"stance": MODE_STANCE, "phase": MODE_PHASE, "swing": MODE_SWING}


class ContactGenerator:
    def __init__(self, contact_sequence):
        self.feet = list(contact_sequence.keys())
        assert self.feet == ["FL", "FR", "RL", "RR"], (
            "contact sequence must be given in RAMBO foot order FL, FR, RL, RR")
        self._mode = []
        self._timing = []
        self._extra = []
        self._max_index = []
        for foot in self.feet:
            entries = contact_sequence[foot]
            self._mode.append([_MODE_VALUES[e[0]] for e in entries])
            self._timing.append([float(e[1]) for e in entries])
            self._extra.append([[float(v) for v in e[2:5]] for e in entries])
            self._max_index.append(len(entries))
        # pad rows to equal length for array storage
        max_len = max(self._max_index)
        for i in range(4):
            pad = max_len - self._max_index[i]
            self._mode[i] += [MODE_STANCE] * pad
            self._timing[i] += [0.0] * pad
            self._extra[i] += [[0.0, 0.0, 0.0]] * pad
        self._mode = np.array(self._mode)            # (4, K)
        self._timing = np.array(self._timing)        # (4, K)
        self._extra = np.array(self._extra)          # (4, K, 3)
        self._timing_cumsum = np.cumsum(self._timing, axis=1)
        self._max_index = np.array(self._max_index)

        # buffers
        self.current_mode = np.ones(4)
        self.current_timing = np.zeros(4)
        self.current_extra = np.zeros((4, 3))
        self.current_phase = np.zeros(4)
        self.current_state = np.ones(4, dtype=bool)
        self._time_since_reset = 0.0

    def reset(self, time_since_reset=0.0):
        self._time_since_reset = time_since_reset
        self._compute()

    def update(self, time_since_reset):
        self._time_since_reset = time_since_reset
        self._compute()

    def _compute(self):
        t = self._time_since_reset
        for i in range(4):
            # index of the active segment (contact_generator.py:153-158),
            # clipped to the last real entry so t beyond the plan total holds
            # the final segment forever
            idx = int(np.sum(t >= (self._timing_cumsum[i, :self._max_index[i]] + 1e-4)))
            idx = min(idx, self._max_index[i] - 1)
            prev = idx - 1

            self.current_mode[i] = self._mode[i, idx]
            self.current_timing[i] = self._timing[i, idx]
            self.current_extra[i] = self._extra[i, idx]
            after_current = t if prev < 0 else t - self._timing_cumsum[i, prev]

            mode = self.current_mode[i]
            if mode > 0.5:  # stance
                self.current_phase[i] = 1.0
                self.current_state[i] = True
            elif mode < -0.5:  # swing
                self.current_phase[i] = 0.0
                self.current_state[i] = False
            else:  # phase
                init_phase, gait_duration, stance_ratio = self.current_extra[i]
                self.current_phase[i] = (
                    np.remainder(after_current + init_phase * gait_duration,
                                 gait_duration) / gait_duration)
                self.current_state[i] = bool(self.current_phase[i] < stance_ratio)

    # ------------------------------------------------------------------
    @property
    def desired_contact_state(self):
        return self.current_state.copy()

    @property
    def desired_contact_phase(self):
        return self.current_phase.copy()

    @property
    def desired_contact_mode(self):
        return self.current_mode.copy()

    @property
    def normalized_phase(self):
        """Phase normalized to [0,1) separately for the stance and swing
        segments of "phase" mode (contact_generator.py:239-260)."""
        phase = self.current_phase.copy()
        for i in range(4):
            if -0.5 < self.current_mode[i] < 0.5:
                stance_ratio = self.current_extra[i, 2]
                if self.current_phase[i] < stance_ratio:
                    phase[i] = self.current_phase[i] / stance_ratio
                else:
                    phase[i] = ((self.current_phase[i] - stance_ratio)
                                / (1.0 - stance_ratio))
        return phase

    @property
    def stance_duration(self):
        """Per-foot stance duration (contact_generator.py:262-275)."""
        duration = self.current_timing.copy()
        for i in range(4):
            if self.current_mode[i] <= -0.5:  # swing
                duration[i] = 0.0
            elif -0.5 < self.current_mode[i] < 0.5:  # phase
                duration[i] = self.current_extra[i, 1] * self.current_extra[i, 2]
        return duration
