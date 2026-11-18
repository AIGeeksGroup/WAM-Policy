"""Global confirmed-start scheduling, independent of model and simulator runtimes.

One node-local SQLite ledger serves the eight ranks. Only short ledger operations
hold a transaction; reset, rendering, GPU work and collectives run outside it.
Checkpoints contain portable logical state, never the database pathname.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import sqlite3

from arm_opsd.config import canonical_hash

SCHEMA = "arm-libero-global-schedule-v1"


class EpisodeSchedule:
    def __init__(self, path, manifest_key, seed=42):
        self.path = Path(path)
        self.manifest_key, self.seed = manifest_key, seed

    @classmethod
    def create(cls, path, manifest, segment, seed=42):
        obj = cls(path, manifest["content_key"], seed)
        decks = {str(t): sorted(c["case_id"] for c in manifest["cases"]
                               if c["split"] == "train" and c["task_id"] == t) for t in range(10)}
        if any(not v or len(v) != len(set(v)) for v in decks.values()):
            raise ValueError("Every task requires unique train reset identities")
        for t in decks:
            obj.shuffle(decks[t], t, 0)
        state = {"schema": SCHEMA, "manifest": obj.manifest_key, "seed": seed,
                 "segment": segment, "cursor": 0, "stopped": False,
                 "confirmed": {t: 0 for t in decks}, "completed": {t: 0 for t in decks},
                 "cycles": {t: 0 for t in decks}, "used": {t: [] for t in decks},
                 "decks": decks, "tickets": {}}
        obj.validate(state)
        obj.path.parent.mkdir(parents=True, exist_ok=True)
        with obj.path.open("xb"):
            pass
        with sqlite3.connect(obj.path) as db:
            db.execute("CREATE TABLE ledger (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
            db.execute("INSERT INTO ledger VALUES (1,?)", (json.dumps(state),))
        return obj

    def shuffle(self, deck, task, cycle):
        deck.sort()
        random.Random(int(canonical_hash([self.seed, task, cycle]), 16)).shuffle(deck)

    def validate(self, state):
        if (state.get("schema") != SCHEMA or state.get("manifest") != self.manifest_key
                or state.get("seed") != self.seed or not state.get("segment")):
            raise ValueError("Scheduler state belongs to another reset manifest, seed or run")
        tasks = {str(t) for t in range(10)}
        if type(state.get("cursor")) is not int or not 0 <= state["cursor"] < 10:
            raise ValueError("Invalid scheduler cursor")
        for field in ("decks", "used", "cycles", "confirmed", "completed"):
            if set(state[field]) != tasks:
                raise ValueError("Incomplete scheduler tasks")
        for task in tasks:
            deck, used = state["decks"][task], state["used"][task]
            tickets = [v for v in state["tickets"].values() if v["task"] == task]
            active = [v["case_id"] for v in tickets if v["status"] in ("reserved", "active")]
            if (not deck or len(deck) != len(set(deck)) or len(used) != len(set(used))
                    or not set(used) <= set(deck) or len(active) > 2 or len(active) != len(set(active))):
                raise ValueError("Invalid deck or active/reserved concurrency")
            if any(v["case_id"] not in deck or v["status"] not in
                   ("reserved", "active", "finished", "cancelled", "interrupted") for v in tickets):
                raise ValueError("Invalid scheduler ticket")
            if state["confirmed"][task] != sum(v["status"] in ("active", "finished", "interrupted") for v in tickets):
                raise ValueError("Confirmed-start count differs from tickets")
            if state["completed"][task] != sum(v["status"] == "finished" for v in tickets):
                raise ValueError("Completed count differs from tickets")
            if state["confirmed"][task] != state["cycles"][task] * len(deck) + len(used):
                raise ValueError("Reset deck consumption differs from confirmed starts")

    @contextmanager
    def transaction(self):
        if not self.path.is_file():
            raise ValueError("Scheduler has not been initialized")
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            state = json.loads(db.execute("SELECT payload FROM ledger WHERE id=1").fetchone()[0])
            self.validate(state)
            yield state
            self.validate(state)
            db.execute("UPDATE ledger SET payload=? WHERE id=1", (json.dumps(state, allow_nan=False),))
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def reserve(self, segment, rank, episode_index):
        if type(rank) is not int or rank < 0 or type(episode_index) is not int or episode_index < 0:
            raise ValueError("Invalid stream segment/rank/episode index")
        key = canonical_hash([segment, rank, episode_index])
        with self.transaction() as state:
            if state["segment"] != segment or state["stopped"]:
                raise ValueError("Stale or stopped scheduler segment")
            if key in state["tickets"]:
                return deepcopy(state["tickets"][key])
            if any(v["rank"] == rank and v["status"] in ("reserved", "active") for v in state["tickets"].values()):
                raise ValueError("Stream still owns an active reservation")
            candidates = {}
            for task, deck in state["decks"].items():
                active = [v["case_id"] for v in state["tickets"].values()
                          if v["task"] == task and v["status"] in ("active", "reserved")]
                used = [] if len(state["used"][task]) == len(deck) else state["used"][task]
                free = [case for case in deck if case not in used and case not in active]
                if len(active) < 2 and free:
                    candidates[task] = free
            if not candidates:
                raise RuntimeError("No available reset ticket")
            task = min(candidates, key=lambda t: (state["confirmed"][t], (int(t) - state["cursor"]) % 10))
            if len(state["used"][task]) == len(state["decks"][task]):
                state["used"][task] = []
                state["cycles"][task] += 1
                self.shuffle(state["decks"][task], task, state["cycles"][task])
                candidates[task].sort(key=state["decks"][task].index)
            ticket = {"id": key, "segment": segment, "rank": rank, "episode_index": episode_index,
                      "task": task, "case_id": candidates[task][0], "status": "reserved"}
            state["tickets"][key] = ticket
            state["cursor"] = (int(task) + 1) % 10
            return deepcopy(ticket)

    def confirm(self, ticket_id):
        with self.transaction() as state:
            ticket = state["tickets"][ticket_id]
            if ticket["status"] in ("active", "finished"):
                return
            if ticket["status"] != "reserved" or state["stopped"]:
                raise ValueError("Cannot confirm retired ticket")
            state["confirmed"][ticket["task"]] += 1
            state["used"][ticket["task"]].append(ticket["case_id"])
            ticket["status"] = "active"

    def finish(self, ticket_id, reason):
        if reason not in ("task_success", "training_timeout", "budget_stop", "environment_error"):
            raise ValueError("Unknown terminal reason")
        with self.transaction() as state:
            ticket = state["tickets"][ticket_id]
            if ticket["status"] in ("finished", "interrupted"):
                if ticket.get("reason") != reason:
                    raise ValueError("Conflicting terminal acknowledgement")
                return
            if ticket["status"] != "active":
                raise ValueError("Terminal requires a confirmed episode")
            natural = reason in ("task_success", "training_timeout")
            ticket.update(status="finished" if natural else "interrupted", reason=reason)
            state["completed"][ticket["task"]] += int(natural)

    def cancel(self, ticket_id):
        with self.transaction() as state:
            ticket = state["tickets"][ticket_id]
            if ticket["status"] == "reserved":
                ticket["status"] = "cancelled"

    def snapshot(self):
        with self.transaction() as state:
            return deepcopy(state)

    def restore(self, snapshot, segment):
        self.validate(snapshot)
        with self.transaction() as state:
            if state["tickets"] or any(state["confirmed"].values()):
                raise ValueError("Restore requires a fresh scheduler")
            if any(set(state["decks"][t]) != set(snapshot["decks"][t]) for t in state["decks"]):
                raise ValueError("Reset identities changed")
            restored = deepcopy(snapshot)
            restored.update(segment=segment, stopped=False)
            for ticket in restored["tickets"].values():
                if ticket["status"] == "reserved":
                    ticket["status"] = "cancelled"
                elif ticket["status"] == "active":
                    ticket.update(status="interrupted", reason="continuity_break")
            state.clear()
            state.update(restored)

    def stop(self):
        with self.transaction() as state:
            state["stopped"] = True
