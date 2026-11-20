"""Test configuration: the QLM-Bench live package for the bridge tests.

The bridge speaks QLM-Bench's qlm-live-v1 protocol through ``qlm_bench.live``. Point
``QLM_BENCH_SRC`` at the ``src/`` of a QLM-Bench checkout (the same variable as the
repository's configs/site.env) to run the tests that need it; they skip otherwise.
"""
import os
import sys

DEPLOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY not in sys.path:
    sys.path.insert(0, DEPLOY)

QLM_SRC = None
_src = os.environ.get("QLM_BENCH_SRC", "")
if _src and os.path.isdir(os.path.join(_src, "qlm_bench")):
    QLM_SRC = os.path.abspath(_src)
    if QLM_SRC not in sys.path:
        sys.path.insert(0, QLM_SRC)
