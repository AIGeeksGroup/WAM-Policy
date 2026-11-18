"""push_box window reward (numpy only).

r = -(e_end - e_start) + 0.5 * min(ds, cap). s = box centre x minus its 2.0 s settled value (world +x); e = mean 3D
distance of the 4 bottom corners to the reference pose (settled pose translated along +x by min(max(s,0), u_stop), yaw
unchanged); cap = progress of the median ideal progress-time curve (80 scripted push_box expert runs from reset) over
the same window length starting where it reaches s_start, also limited by u_stop (= distance of the settled box's
rear-most corner to the goal's near edge).

The median ideal curve is the package data file ``data/ideal_progress.npy``.

A trajectory is the teacher's own episode rows from reset up to the start tick, then the candidate's rows after it
(so the 2.0 s settled row is the real one). Early terminal (success inside the window): last row held to the window end.
"""
from pathlib import Path

import numpy as np

CH = 16
BOTTOM = (0, 2, 4, 6)
SPEED = Path(__file__).resolve().parent / 'data' / 'ideal_progress.npy'


def fill(rows, n):
    rows = np.asarray(rows, np.float64)
    if len(rows) >= n:
        return rows, False
    return np.concatenate([rows, np.repeat(rows[-1:], n - len(rows), 0)]), True


def D_from_rows(priv, cols):
    """Task distance D: max over the 8 corners of the XY distance to the goal rectangle."""
    cols = [str(c) for c in cols]; g = lambda n: priv[:, cols.index(n)]
    cx = np.stack([g(f'corner{k}_x') for k in range(8)], 1); cy = np.stack([g(f'corner{k}_y') for k in range(8)], 1)
    q = priv[0, [cols.index(n) for n in ('goal_x0', 'goal_x1', 'goal_y0', 'goal_y1')]]
    return np.hypot(np.maximum(0, np.maximum(q[0] - cx, cx - q[1])), np.maximum(0, np.maximum(q[2] - cy, cy - q[3]))).max(1)


class PushReward:
    BETA = 0.5; FLOOR = 5e-4; I0 = 100   # 2.0 s settled row

    def __init__(self, window_chunks, speed=None):
        self.H = window_chunks
        self.speed = np.load(SPEED) if speed is None else np.asarray(speed, np.float64)

    def cap(self, s0, rows):
        s0 = max(float(s0), self.FLOOR); j = int(np.searchsorted(self.speed, s0, side='right'))
        if j >= len(self.speed): return 0.0
        i = max(j - 1, 0); k = min(i + rows, len(self.speed) - 1)
        return float(self.speed[k] - self.speed[i])

    def geometry(self, priv, cols):
        g = lambda n: priv[:, cols.index(n)]
        cor = np.stack([np.stack([g(f'corner{k}_x'), g(f'corner{k}_y'), g(f'corner{k}_z')], 1) for k in BOTTOM], 1)
        cen = np.stack([g('object_center_x'), g('object_center_y')], 1)
        yaw = np.unwrap(2 * np.arctan2(g('object_qz'), g('object_qw')))
        goal = priv[0, [cols.index(n) for n in ('goal_x0', 'goal_x1', 'goal_y0', 'goal_y1')]]
        return cor, cen, yaw, goal

    def series(self, priv, cols):
        cols = [str(c) for c in cols]
        cor, cen, yaw, goal = self.geometry(priv, cols)
        c0 = cor[self.I0]; s = cen[:, 0] - cen[self.I0, 0]
        u_stop = float(goal[0] - c0[:, 0].min())
        y_inside = bool((c0[:, 1] >= goal[2]).all() and (c0[:, 1] <= goal[3]).all())
        u = np.clip(s, 0, u_stop)
        ref = c0[None] + np.stack([u, np.zeros_like(u), np.zeros_like(u)], 1)[:, None, :]
        e = np.linalg.norm(cor - ref, axis=2).mean(1)
        lat = cen[:, 1] - cen[self.I0, 1]; dyaw = yaw - yaw[self.I0]
        return dict(s=s, e=e, lat=lat, dyaw=dyaw, u_stop=u_stop, y_inside=y_inside)

    def window(self, priv, cols, k0):
        z = self.series(priv, cols); k1 = k0 + self.H * CH
        ds = z['s'][k1] - z['s'][k0]
        cap = min(self.cap(z['s'][k0], self.H * CH), max(z['u_stop'] - max(z['s'][k0], 0), 0.0))
        de = z['e'][k1] - z['e'][k0]; prog = self.BETA * min(ds, cap)
        return dict(r=-de + prog, dev=-de, prog=prog, ds=ds, cap=cap, e0=z['e'][k0], e1=z['e'][k1],
                    lat1=z['lat'][k1], dlat=z['lat'][k1] - z['lat'][k0], dyaw=z['dyaw'][k1] - z['dyaw'][k0],
                    yaw1=z['dyaw'][k1], y_inside=z['y_inside'])


def candidate_reward(R, base_rows, cand_rows, cols):
    """Window reward of one candidate: base episode rows + candidate rows (restore row first), early end held."""
    from wam_policy.task_rl.tasks.base import splice
    rows, padded = fill(cand_rows, R.H * CH + 1)
    priv, i0 = splice(base_rows, rows)
    w = R.window(priv, cols, i0)
    w['padded'] = bool(padded)
    return w
