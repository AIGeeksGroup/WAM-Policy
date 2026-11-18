"""Stage 1.1 ablation: RGB + proprioception teacher.

Same whole-episode SFT recipe as the Stage 1.1 teacher (5000 steps, frozen VAE/T5 cache, native 9D
actions), with an additional 10D proprioceptive state stream predicted and conditioned jointly with
video: base height above the ground plane (1), gravity direction in the body frame (3), body-frame
linear velocity (3) and body-frame angular velocity (3). One state is taken at each 12.5 Hz RGB
frame, so a video latent group carries 1 (initial) or 4 states.

The model is ``WanTransformer3DModel(..., proprioception_dim=10)``: two extra linear layers
(``proprioception_embedder`` / ``proprioception_proj_out``) and six token roles in the causal mask.
Online it needs QLM live observations that carry the measured root-link state.

Entry point: ``python -m wam_policy.ablations.proprioception {convert,data,train,export,serve}``.
"""
