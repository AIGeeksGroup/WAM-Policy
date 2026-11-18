"""Stage 1.2 OPSD: on-policy self-distillation of the low-NFE (1/2) student from the Stage 1.1 teacher.

This package is the distillation engine shared with Stage 2.2 M-OPD (:mod:`wam_policy.mopd`):
config, teacher/student LoRA roles, sampling, same-noise regression targets and the teacher target
cache, history KV replay, the streaming coordinator serving the QLM-Bench training-environment
workers, the learner, losses, checkpoints, export in the release student format, the single-node
launcher, metrics, optional visualization, and the deployment sampler (:mod:`.deploy`).
"""
