"""Flow-matching settings shared by training and sampling (SNR shifts, noisy conditioning, windows)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowConfig:
    video_snr_shift: float = 5.0
    action_snr_shift: float = 1.0
    video_noisy_conditioning: float = 0.5
    action_noisy_conditioning: float = 0.0
    chunk_min: int = 1
    chunk_max: int = 4
    window_min: int = 4
    window_max: int = 64
