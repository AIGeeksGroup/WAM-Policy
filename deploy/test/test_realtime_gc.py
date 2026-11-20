"""wam_go2/realtime.py: the start-up heap is frozen out of Python's cyclic GC in the controller and the bridge.

Without the freeze, a fresh controller's first POLICY run (mock_robot_test) had one 42.6 ms control gap (every
other tick 10 ms) - the first full collection of the start-up heap (torch + rclpy, ~190 000 objects: gc.collect()
20-26 ms in a bare process). With the policy server running on the same machine the gap could exceed the 50 ms
control_gap_timeout and latch DAMP. What is pinned:
  * freeze_startup_heap collects once, then freezes every live object (a later full collection is cheap);
  * the controller and the bridge call it after the node is built (policy loaded) and before they spin.
"""
import gc
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_freeze_startup_heap_freezes_the_live_objects_and_makes_a_full_collection_cheap():
    from wam_go2 import realtime
    junk = [{"i": i, "l": [i]} for i in range(200_000)]          # a heap like the controller's
    try:
        gc.collect()
        t = time.perf_counter()
        gc.collect()                                                 # a full scan of that heap, for comparison
        unfrozen_ms = (time.perf_counter() - t) * 1000
        frozen, collect_ms = realtime.freeze_startup_heap()
        # every live object is frozen; other threads of the test run may free a few frozen ones afterwards
        assert frozen >= 200_000 and gc.get_freeze_count() >= 200_000
        assert collect_ms >= 0.0
        durations = []
        for _ in range(5):                  # the fastest of five: other threads of the test run may take the GIL
            t = time.perf_counter()
            gc.collect()
            durations.append((time.perf_counter() - t) * 1000)
        # absolute bound, or (on a loaded machine, where every timing stretches) far below the full scan
        fastest = min(durations)
        assert fastest < 5.0 or fastest < unfrozen_ms / 4, \
            f"a full collection still scans the frozen heap ({fastest:.1f} ms vs {unfrozen_ms:.1f} ms unfrozen)"
    finally:
        gc.unfreeze()
        del junk


def _main_body(path):
    text = (REPO / path).read_text()
    return text[text.index("\ndef main("):]


def test_the_controller_freezes_after_building_the_node_and_before_spinning():
    body = _main_body("wam_go2/nodes/controller_node.py")
    build, freeze, spin = body.index("node = RamboGo2Controller()"), body.index("freeze_startup_heap()"), body.index("spin_once")
    assert build < freeze < spin
    assert re.search(r"from \.\.realtime import freeze_startup_heap|from \.\. import realtime", (REPO / "wam_go2/nodes/controller_node.py").read_text())


def test_the_bridge_freezes_after_building_the_node_and_before_spinning():
    body = _main_body("wam_go2/nodes/wam_bridge_node.py")
    freeze, spin = body.index("freeze_startup_heap()"), body.index("spinner.start()")
    assert freeze < spin
