"""QLM-Bench Standard100 evaluation, policy side.

* :mod:`.standard100` -- default: one ``wam_policy.inference.serve`` process and one QLM-Bench
  client per policy seed, episode by episode (not accelerated; the VAE may stay on the CPU).
* :mod:`.batched_service`, :mod:`.multi_session_service`, :mod:`.run_parallel` -- accelerated:
  a resident multi-session service (VAE on the GPU) with several concurrent clients.
* :mod:`.trials`, :mod:`.compare` -- trial tables, success rates and paired McNemar tests over
  (case, policy seed) pairs (seeds 42/43/44 give the 300-episode protocol).
"""
