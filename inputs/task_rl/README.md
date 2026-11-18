# Task RL inputs (Stage 2.1)

Scene and case inputs of the two task-specific RL specialists, `push_box` and `open_room_door`. The RL
algorithm, reward and launchers are in WAM-Policy (`wam_policy.task_rl`); its `prepare` step unpacks these inputs
into a data root, fetches the missing large files by sha256 from the QLM-Bench dataset and starts the QLM scene
worker (`python -m qlm_bench.task_rl.scene_worker`, render mode `sampled` in the paper).

Layout below the data root (recorded paths start with `@WORKSPACE@`, i.e. the data root):

```
task_rl/push_box/scenes/<scene>/        81 push_box train scene packs (in the push_box archive)
task_rl/push_box/cases/, cases-index.json   the 80 push_box train EvalCases (in the push_box archive)
task_rl/open_room_door/scenes/<scene>/  80 open_room_door train scene packs (in the open_room_door archive)
assets/rambo-go2-controller/iteration-2000/runtime/   RAMBO controller (fetched from the dataset)
standard100/benchmarks/mujoco-standard-test100-v1/    Standard100 suite (optional; named in the cases indexes)
```

| file | content |
|---|---|
| `*/workspace-files.tar.gz` (+ `.sha256`) | the small files of the scene packs (xml, json, small obj/npz; push also the cases): push 6,081 files, door 560 |
| `open_room_door/cases/`, `cases-index.json` | the 80 door train EvalCases (16 Clean, 64 Randomized; horizon 60 s) and their index |
| `*/scene-files.json` | sha256 of every file a run reads by data-root-relative path: the archive's files, the large meshes (fetched by sha256) and the controller runtime files |
| `*/scene-check-picks.json`, `*/commands/` | three stored episode starts per task and their executed chunks; example inputs for `scripts/smoke/task_rl_scene_worker.py` |

The released inputs contain training cases. See the [Task RL input contract](../../docs/interfaces.md) for the expected layout.
