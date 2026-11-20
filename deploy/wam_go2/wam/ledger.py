"""Execution ledger of one 16-command chunk, on QLM-Bench's public command contract.

Each native9 command of a chunk is prepared with ``qlm_bench.contracts.command.runtime.prepare_command``
(validated, force zeroed). The robot confirms every 10 ms controller step (5 logical ticks); a command is
complete after two confirmed steps. ``acknowledgement`` is the body of the ExecutionAck the bridge sends
back to the policy server: completed commands carry what was executed, the rest stay ``not_executed`` /
``partial``.
"""
from __future__ import annotations

import copy

from qlm_bench.contracts.command.runtime import prepare_command
from qlm_bench.contracts.command.validation import require, validate_command, validate_record


class CommandLedger:
    """Acknowledges backend-confirmed 10 ms steps; a full command needs two steps."""

    def __init__(self, actions, chunk_id, start_tick):
        require(type(chunk_id) is str and bool(chunk_id), "action", "chunk ID required")
        require(type(actions) is list and len(actions) > 0 and len(actions) % 16 == 0,
                "action", "K must be a positive multiple of 16")
        self.commands = [prepare_command(a, f"{chunk_id}:{i}", start_tick + 10 * i) for i, a in enumerate(actions)]
        self.chunk_id = chunk_id
        self.start_tick = start_tick
        self.tick = start_tick
        self.trace = []

    @property
    def complete(self):
        return self.tick == self.commands[-1]["end_tick"]

    def next_step(self):
        require(not self.complete, "state", "chunk completed")
        row = self.commands[(self.tick - self.start_tick) // 10]
        return dict(start_tick=self.tick, end_tick=self.tick + 5, command_id=row["command_id"],
                    held=copy.deepcopy(row["filtered"]))

    def confirm(self, record):
        expected = self.next_step()
        validate_record("control_step", record)
        require(all(record[k] == v for k, v in expected.items()), "execution",
                "backend confirmation differs from scheduled step")
        require(record["fl_ik_computed_tick"] <= self.tick, "timing", "future IK computation")
        row = self.commands[(self.tick - self.start_tick) // 10]
        self.trace.append(copy.deepcopy(record))
        self.tick = record["end_tick"]
        row["executed_until_tick"] = self.tick
        row["status"] = "completed" if self.tick == row["end_tick"] else "partial"
        row["executed"] = copy.deepcopy(row["filtered"]) if row["status"] == "completed" else None
        validate_command(row)

    def acknowledgement(self, reason):
        return dict(chunk_id=self.chunk_id, start_tick=self.start_tick, end_tick=self.tick,
                    completed_count=sum(x["status"] == "completed" for x in self.commands),
                    commands=copy.deepcopy(self.commands), reason=reason)
