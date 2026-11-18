"""Stage 2.2 M-OPD: multi-teacher on-policy distillation with task-routed specialist teachers.

The distillation engine is :mod:`wam_policy.opsd`; this package adds the routed teacher
(:mod:`.experts`: Stage 2.1 specialists on the Stage 1.1 teacher, selected by each context's task)
and the specialist teacher cache (:mod:`.expert_cache`: collect only the expert tasks and merge them
into the OPSD teacher cache).
"""
