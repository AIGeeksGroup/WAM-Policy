"""Keep Python's cyclic garbage collector off the 100 Hz control path.

The first full collection of the start-up heap (torch, rclpy, numpy, the policy: about 190 000 objects,
gc.collect() 20-26 ms in a bare process) otherwise lands in the first busy phase of the first POLICY run as one
~40 ms control gap; with the policy server running on the same machine that can exceed the controller's 50 ms
control_gap_timeout and latch DAMP. Collecting once while the node is idle and then freezing every live object
(gc.freeze: moved to the permanent generation, never scanned again) leaves later collections only the few
objects created after start-up.
"""
import gc
import time


def freeze_startup_heap():
    """Full collection now, then gc.freeze(). Returns (objects frozen, collection time in ms)."""
    t = time.perf_counter()
    gc.collect()
    collect_ms = (time.perf_counter() - t) * 1000.0
    gc.freeze()
    return gc.get_freeze_count(), collect_ms
