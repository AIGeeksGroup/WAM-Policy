#!/usr/bin/env python3
"""MuJoCo closed loop running the same controller code as the robot.

    python3 sim/mj_sim.py SCENARIO [options]          # --help lists every option

Everything above the motors is the code the ROS node runs: the state estimator and RamboController with
the RAMBO checkpoint (policies/quadruped/model_2000.pt or --checkpoint); the motor PD runs at 500 Hz
on the commanded (q, dq, tau_ff) with the node's gains. Model: MuJoCo Menagerie's Unitree Go2
(third_party/mujoco_menagerie/unitree_go2, BSD-3; scripts/fetch_mujoco_menagerie.sh), 0.002 s physics as in training,
with a knee-end contact sphere (r = 0.013), stiff foot contact, friction on all collision geoms and
mild passive joint damping / friction (0.5 Nm s/rad, 0.2 Nm; --joint-damping / --joint-friction).

SCENARIO
  check-fk    compare wam_go2 kinematics with the MuJoCo model (joint order, signs, foot positions);
              run this first
  quad        the quadruped policy from a level stand
  wam_chunks  the WAM-Policy bridge's execution model: STAND -> POLICY -> the bridge's episode driver
              with a scripted mock policy and --infer-delay seconds of simulated latency per request
              (needs the QLM-Bench live package: QLM_BENCH_SRC)

    python3 sim/mj_sim.py wam_chunks --chunks 10 --infer-delay 1.0

The simulation is a design check, not a validation: what fails here is not worth trying on the
robot, and what works here still needs the leash.
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # noqa: E402

from wam_go2.adapter.hardware_limits import clip_joint_positions  # noqa: E402
from wam_go2.constants import (  # noqa: E402
    DEFAULT_JOINT_POS,
    JOINT_TORQUE_LIMIT,
)
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.math_utils import (  # noqa: E402
    quat_rotate,
    quat_rotate_inverse,
    rp_rotation_from_gravity_b,
)
from wam_go2.control.policy import RamboPolicy  # noqa: E402
from wam_go2.control.rambo_controller import RamboController  # noqa: E402
from wam_go2.control.robot_state import RobotState  # noqa: E402
from wam_go2.estimator.state_estimator import StateEstimator  # noqa: E402
from wam_go2.tasks import QUADRUPED  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCENE = os.path.join(ROOT, "third_party", "mujoco_menagerie", "unitree_go2", "scene.xml")
CHECKPOINT = os.path.join(ROOT, "policies", QUADRUPED.checkpoint_relpath)

# MuJoCo joint/actuator order: per leg FL, FR, RL, RR x (hip, thigh, calf).
# RAMBO order: [hip x4, thigh x4, calf x4] with legs FL, FR, RL, RR.
RAMBO_TO_MJ = np.array([3 * leg + j for j in range(3) for leg in range(4)])
FEET = ["FL", "FR", "RL", "RR"]
STAND_KP, STAND_KD = 60.0, 5.0
POLICY_KP, POLICY_KD = 40.0, 1.0
PHYS_DT = 0.002
CTRL_DT = 0.01
np.set_printoptions(precision=3, suppress=True, linewidth=200)


# ---------------------------------------------------------------------------
# Rounded knee end of the calf link: a contact sphere at the knee axis (the
# menagerie calf's shin cylinder stops 4.7 cm short of it), so a knee that
# touches the floor rests on it instead of passing through. 0 = no sphere.
KNEE_RADIUS = 0.013

# Pushable box (wam_chunks). dx is the DEPTH along the push
# direction, dz the height; x/y place its centre in world coordinates.
BoxSpec = collections.namedtuple(
    "BoxSpec", "mass dx dy dz mu_bottom mu_shell x y dof")
BOX_SHEET_T = 0.004      # half-thickness of the slippery bottom sheet [m]


def load_model(knee_radius=KNEE_RADIUS, joint_damping=None, joint_friction=None,
               foot_friction=None, com_shift=0.0, box=None):
    """box: None (default, the model is untouched) or a BoxSpec - a free-floating
    pushable box built as TWO geoms, which is how the real article is meant to be
    prepared:

      box_shell   the carton itself, priority 0, so the PAW's contact keeps the
                  foot's own parameters (the menagerie foot is priority 1:
                  condim 6, friction 0.8, go2.xml:31-34). Tuning the box cannot
                  change how the paw grips it.
      box_sheet   a thin slab under the carton, priority 2, so IT decides the
                  box/floor friction. Without this the contact friction is
                  max(box, floor) = the floor's 1.0 and the box cannot be made
                  to slide at all.

    Why two geoms and not one: on the high-friction mat a 4 kg carton needs
    mu*m*g ~27 N to slide (~35 N to break away). The policy's 30 N clip bounds
    the FEEDFORWARD force command only, not the contact force from position
    stiffness, so it does not by itself say the carton cannot be moved. Static
    tipping precedes sliding only for a contact higher than (depth/2)/mu =
    0.21 m at mu 0.7 (0.50 m at mu 0.3); the 0.16 m paw is below both, so a low
    push slides first. A slippery bottom sheet at mu 0.30 drops the
    push to ~12 N, a force the gait absorbs gently, and is what the operator
    tapes under the real box.

    Both geoms are group 0 on purpose. Group 3 would be caught by the two
    post-compile loops below (which would overwrite the friction with 0.6 and
    again under --foot-friction) and, because MjvOption's default geomgroup is
    [1 1 1 0 0 0], would also be invisible in --record.
    """
    spec = mujoco.MjSpec.from_file(SCENE)
    for name, kind in (("gyro", mujoco.mjtSensor.mjSENS_GYRO),
                       ("acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
                       ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname="imu")
    if knee_radius > 0:
        for f in FEET:
            body = spec.body(f"{f}_calf")
            g = body.add_geom()
            g.name = f"{f}_knee"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[0] = knee_radius
            g.pos[:] = [0.0, 0.0, 0.0]
            g.group = 3
            g.condim = 3
            g.friction[:] = [0.6, 0.005, 0.0001]
            g.rgba[:] = [0.8, 0.2, 0.2, 0.4]
    if box is not None:
        # added LAST so the box free joint lands after the robot's: qpos[7:] /
        # qvel[6:] and the RAMBO_TO_MJ actuator indexing all stay valid.
        # centre height so the BOTTOM OF THE SHEET rests exactly on the floor.
        # Getting this wrong buries the sheet in the floor, which for a planar
        # box (no z dof) welds it down: measured -0.05 m of box travel and the
        # robot pushed backwards.
        b = spec.worldbody.add_body(
            name="box", pos=[box.x, box.y, box.dz / 2 + 2 * BOX_SHEET_T])
        if box.dof == "planar":
            # A box the OPERATOR has constrained against rotation - guides, a
            # rail, a shallow tray. Two slide joints: it translates on the floor
            # and cannot yaw or tip. Use this to measure the push itself; use
            # the default free box to see what an unconstrained one really does
            # (it spins out from under a single point contact within a few
            # seconds - see --box-dof in the help).
            for axis, name in (([1, 0, 0], "box_x"), ([0, 1, 0], "box_y")):
                j = b.add_joint()
                j.name = name
                j.type = mujoco.mjtJoint.mjJNT_SLIDE
                j.axis[:] = axis
        else:
            b.add_freejoint()
        shell = b.add_geom()
        shell.name = "box_shell"
        shell.type = mujoco.mjtGeom.mjGEOM_BOX
        shell.size[:] = [box.dx / 2, box.dy / 2, box.dz / 2]
        shell.mass = box.mass          # explicit: the default density 1000 would
        shell.group = 0                # make a 0.3x0.35x0.35 carton weigh 36 kg
        shell.condim = 3
        shell.friction[:] = [box.mu_shell, 0.005, 0.0001]
        shell.priority = 2             # beats the foot (priority 1) so mu_shell
        shell.rgba[:] = [0.78, 0.57, 0.33, 1.0]   # decides the PAW contact too
        sheet = b.add_geom()
        sheet.name = "box_sheet"
        sheet.type = mujoco.mjtGeom.mjGEOM_BOX
        sheet.size[:] = [box.dx / 2, box.dy / 2, BOX_SHEET_T]
        sheet.pos[:] = [0.0, 0.0, -box.dz / 2 - BOX_SHEET_T]
        sheet.mass = 0.01
        sheet.group = 0
        sheet.condim = 3
        sheet.friction[:] = [box.mu_bottom, 0.005, 0.0001]
        sheet.priority = 2             # beats the floor -> the box's mu wins
        sheet.rgba[:] = [0.25, 0.45, 0.85, 1.0]
    model = spec.compile()
    # the menagerie go2 class makes every collision geom but the feet
    # frictionless (condim 1); give the knees/shins ordinary friction for
    # when they touch the floor
    foot_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FEET}
    for g in range(model.ngeom):
        if model.geom_group[g] == 3 and g not in foot_ids:
            model.geom_condim[g] = 3
            model.geom_friction[g] = [0.6, 0.005, 0.0001]
    # the menagerie foot is very soft (solimp 0.015 ..: it sinks ~1.3 cm under
    # 70 N). Use MuJoCo's default (stiff) contact for the feet.
    for g in foot_ids:
        model.geom_solimp[g] = [0.9, 0.95, 0.001, 0.5, 2.0]
        model.geom_solref[g] = [0.02, 1.0]
    # training actuators (rambo_quadruped qp_env.py:395-421): implicit PD kp 40 /
    # kd 1, friction 0, no extra passive joint damping; the menagerie model
    # adds damping 2 + frictionloss 0.2 per joint. Override when asked
    # (dofs 0-5 are the free joint).
    if joint_damping is not None:
        model.dof_damping[6:] = joint_damping
    if joint_friction is not None:
        model.dof_frictionloss[6:] = joint_friction
    # floor sweep (--foot-friction): sliding friction of every robot/floor
    # contact. Default (None) keeps the menagerie values: foot 0.8 (priority
    # 1, so it wins over the floor), knees 0.6 but combined with the floor's
    # 1.0 by max -> 1.0. With a value, floor + feet + knees all get mu, so
    # every contact slides at mu (the RAMBO paper's rubber mat vs parquet).
    if foot_friction is not None:
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_friction[floor, 0] = foot_friction
        for g in range(model.ngeom):
            if model.geom_group[g] == 3:
                model.geom_friction[g, 0] = foot_friction
    # --com-shift: move the trunk's centre of mass toward the BELLY (body -z)
    # to test how sensitive a scenario is to a CoM that differs from the
    # menagerie model.
    if com_shift:
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        model.body_ipos[base, 2] -= float(com_shift)
    return model


class Sim:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.foot_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FEET]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.calf_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{f}_calf") for f in FEET]
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.kp, self.kd = STAND_KP, STAND_KD      # scalars or (12,) RAMBO-order arrays
        self.q_des = np.zeros(12)
        self.dq_des = np.zeros(12)
        self.tau_ff = np.zeros(12)
        self.t = 0.0
        # pushable box (None unless load_model was given a BoxSpec)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
        if bid < 0:
            self.box_body = self.box_qadr = self.box_dofadr = None
            self.box_geoms = None
            self.box_rest_z = 0.0
            self.box_free = False
            self.box_nq = self.box_nv = 0
        else:
            self.box_body = bid
            jnt = model.body_jntadr[bid]
            self.box_qadr = int(model.jnt_qposadr[jnt])
            self.box_dofadr = int(model.jnt_dofadr[jnt])
            # free box: one 7-qpos free joint. planar box: two 1-qpos slides.
            self.box_free = bool(model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE)
            self.box_nq = 7 if self.box_free else int(model.body_jntnum[bid])
            self.box_nv = 6 if self.box_free else int(model.body_dofnum[bid])
            self.box_geoms = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in ("box_shell", "box_sheet")}
            self._box_home = model.body_pos[bid].copy()
            shell = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_shell")
            # centre height at rest = half height + the bottom sheet
            self.box_rest_z = float(model.geom_size[shell][2] + 2 * BOX_SHEET_T)

    # -- state (RAMBO order) -------------------------------------------------
    @property
    def q(self):
        return self.data.qpos[7:][RAMBO_TO_MJ].copy()

    @property
    def dq(self):
        return self.data.qvel[6:][RAMBO_TO_MJ].copy()

    @property
    def quat(self):
        return self.data.sensor("quat").data.copy()

    @property
    def gyro(self):
        return self.data.sensor("gyro").data.copy()

    @property
    def acc(self):
        return self.data.sensor("acc").data.copy()

    @property
    def tau_applied(self):
        return self.data.actuator_force[RAMBO_TO_MJ].copy()

    def set_pose(self, q_rambo, quat, z):
        # qpos[:] = 0 would also teleport a box to the origin with a zero
        # (invalid) quaternion, so its free joint is saved and restored.
        box_q = None if self.box_qadr is None else self.data.qpos[
            self.box_qadr:self.box_qadr + self.box_nq].copy()
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = z
        self.data.qpos[3:7] = quat
        self.data.qpos[7:][RAMBO_TO_MJ] = q_rambo
        if box_q is not None:
            self.data.qpos[self.box_qadr:self.box_qadr + self.box_nq] = box_q
        mujoco.mj_forward(self.model, self.data)

    def place_box(self, x, y=None, yaw=0.0):
        """Put the box down at (x, y), level and at rest. Must be called AFTER
        set_pose / drop_to_floor, which both move or ignore it."""
        if self.box_qadr is None:
            return
        a = self.box_qadr
        if self.box_free:
            self.data.qpos[a + 0] = x
            if y is not None:
                self.data.qpos[a + 1] = y
            self.data.qpos[a + 2] = self.box_rest_z
            self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        else:
            # planar: slide x, slide y, measured from the body's spec position
            self.data.qpos[a + 0] = x - self._box_home[0]
            if y is not None:
                self.data.qpos[a + 1] = y - self._box_home[1]
        self.data.qvel[self.box_dofadr:self.box_dofadr + self.box_nv] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def box_state(self):
        """(position (3,), yaw [rad]) of the box, or (None, 0.0)."""
        if self.box_qadr is None:
            return None, 0.0
        # xpos is correct for both joint layouts; qpos is not
        pos = self.data.xpos[self.box_body].copy()
        if not self.box_free:
            return pos, 0.0
        a = self.box_qadr
        w, x, y, z = self.data.qpos[a + 3:a + 7]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return pos, yaw

    def drop_to_floor(self, clearance=0.002):
        """Lower the base until the first floor contact, then back off.

        Only ROBOT geoms count: a box resting on the floor is itself a
        floor contact with dist < 0 and would stop the loop at once, leaving
        the base 0.3 m in the air.
        """
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        d.qpos[2] += 0.3
        for _ in range(2000):
            mujoco.mj_forward(m, d)
            touching = any(
                self.floor in (d.contact[i].geom1, d.contact[i].geom2)
                and d.contact[i].dist < 0.0
                and not self._is_box_contact(d.contact[i])
                for i in range(d.ncon))
            if touching:
                break
            d.qpos[2] -= 0.001
        d.qpos[2] += clearance
        mujoco.mj_forward(m, d)

    def _is_box_contact(self, con):
        if self.box_geoms is None:
            return False
        return con.geom1 in self.box_geoms or con.geom2 in self.box_geoms

    def com_vs_feet(self):
        """Whole-body CoM relative to the hind feet (heading frame fwd, and
        world up), mean over RL/RR."""
        # subtree_com[base_body], not subtree_com[0]: the world subtree would
        # include a pushable box (measured: a 4 kg box shifts it 5 cm and the
        # printed 'com-foot fwd' with it).
        com = self.data.subtree_com[self.base_body]
        zb_w = quat_rotate(self.quat, np.array([0, 0, 1.0]))
        fwd = -zb_w.copy(); fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0, 0])
        feet = np.mean([self.data.geom_xpos[g] for g in self.foot_geom[2:]], axis=0)
        r = com - feet
        return float(r @ fwd), float(r[2])

    def command(self, q_des, dq_des=None, tau_ff=None, kp=None, kd=None):
        # Same final hardware position fence as the ROS LowCmd adapter.
        self.q_des = clip_joint_positions(q_des)
        self.dq_des = np.zeros(12) if dq_des is None else np.asarray(dq_des, dtype=float)
        self.tau_ff = np.zeros(12) if tau_ff is None else np.asarray(tau_ff, dtype=float)
        # gains are kept when not given
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd

    def physics_step(self):
        q, dq = self.q, self.dq
        tau = self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq) + self.tau_ff
        tau = np.clip(tau, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
        self.data.ctrl[RAMBO_TO_MJ] = tau
        mujoco.mj_step(self.model, self.data)
        self.t += PHYS_DT

    # -- ground truth diagnostics --------------------------------------------
    def foot_contacts(self):
        c = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for k, g in enumerate(self.foot_geom):
                if g in (con.geom1, con.geom2):
                    c[k] = True
        return c

    def contact_normal_force_on(self, body_ids):
        """Total normal contact force [N] on the given bodies' geoms (floor contacts)."""
        total = 0.0
        f = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (b1 in body_ids or b2 in body_ids) and self.floor in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, f)
                total += abs(f[0])
        return total

    def knee_load(self):
        """Floor normal force carried by the hind calves minus the hind feet
        (= what the shins/knees carry)."""
        calves = set(self.calf_body[2:])
        total = self.contact_normal_force_on(calves)
        feet = 0.0
        f = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 in self.foot_geom[2:] or con.geom2 in self.foot_geom[2:]) \
                    and self.floor in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, f)
                feet += abs(f[0])
        return max(0.0, total - feet), feet

    def posture(self):
        g = quat_rotate_inverse(self.quat, np.array([0.0, 0.0, -1.0]))
        pitch_back = np.degrees(np.arctan2(g[2], -g[0]))
        roll = np.degrees(np.arctan2(g[1], -g[0]))
        return g, pitch_back, roll

    def base_vs_feet(self):
        """Base position relative to the hind feet in the heading frame
        (fwd = horizontal belly direction): (RL fwd, RL up), (RR fwd, RR up)."""
        base = self.data.xpos[self.base_body]
        zb_w = quat_rotate(self.quat, np.array([0, 0, 1.0]))
        fwd = -zb_w.copy(); fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0, 0])
        out = []
        for g in self.foot_geom[2:]:
            r = base - self.data.geom_xpos[g]
            out.append((float(r @ fwd), float(r[2])))
        return out


# ---------------------------------------------------------------------------
class Harness:
    """Runs the node's control stack against the sim at 100 Hz."""

    def __init__(self, sim, task=QUADRUPED, policy_path=None, policy=None):
        self.sim = sim
        self.task = task
        policy_path = policy_path or os.path.join(ROOT, "policies", task.checkpoint_relpath)
        self.est = StateEstimator(nominal_height=task.nominal_base_height)
        self.est.reset()
        # `policy`: an already-loaded RamboPolicy (stateless: the obs history
        # lives in the controller), so repeated runs do not reload the net
        self.policy = policy if policy is not None else RamboPolicy(policy_path, num_obs=task.num_obs)
        self.controller = RamboController(self.policy, task=task)
        self.mode = "HOLD"
        self.policy_time = 0.0
        self.last_contact = np.ones(4, dtype=bool)
        self.log = []
        self.true_vel = False       # feed the sim's true base velocity instead of the KF
        self.recorder = None        # Recorder: captures a frame per video tick (see --record)
        self.phase = ""             # caption shown in the recording
        self.fall_check = True      # run() stops on a fall (gravity error > 1.2 or base < 0.12 m)
        # motor gains while in POLICY: the node's 40/1
        self.policy_kp = POLICY_KP
        self.policy_kd = POLICY_KD

    def kf_update(self, contact):
        s = self.sim
        self.est.update(PHYS_DT, s.quat, s.gyro, s.acc, s.q, s.dq, contact)

    def true_base_lin_vel_b(self):
        s = self.sim
        v_w = s.data.qvel[0:3].copy()          # free joint: world-frame linear velocity
        return quat_rotate_inverse(s.quat, v_w)

    def robot_state(self, contact):
        s = self.sim
        v_b = self.true_base_lin_vel_b() if self.true_vel else self.est.base_lin_vel_b(s.quat)
        h = float(s.data.xpos[s.base_body][2]) if self.true_vel else self.est.base_height
        return RobotState(joint_pos=s.q, joint_vel=s.dq, base_quat=s.quat,
                          base_ang_vel_b=s.gyro, base_lin_vel_b=v_b,
                          base_height=h, foot_contacts=contact).finalize()

    def enter_policy(self):
        s = self.sim
        self.est.reset()
        # KF restart on four planted feet: POLICY starts from the level stand
        contact = np.ones(4, dtype=bool)
        self.est.update(PHYS_DT, s.quat, s.gyro, s.acc, s.q, s.dq, contact)
        self.last_contact = contact
        rs = self.robot_state(contact)
        self.controller.set_commands(vel_cmd=[0.0, 0.0, 0.0])
        for slot in range(self.task.num_ee):
            self.controller.set_commands(ee_pos_cmd=self.task.ee_pos_default[slot],
                                         ee_force_cmd=[0.0, 0.0, 0.0], leg_slot=slot)
        self.controller.reset(rs)
        self.policy_time = 0.0
        self.mode = "POLICY"

    def run(self, duration, tick_fn=None, print_every=0.05, label=""):
        """Advance `duration` seconds. tick_fn(harness, t) is called at 100 Hz
        BEFORE the controller/PD command of that tick and may change modes."""
        s = self.sim
        t_end = s.t + duration
        next_print = s.t
        while s.t < t_end - 1e-9:
            if tick_fn is not None:
                tick_fn(self, s.t)
            if self.mode == "POLICY":
                contact = self.controller.contact_generator.desired_contact_state
                self.last_contact = contact
                rs = self.robot_state(contact)
                q_des, dq_des, tau_ff = self.controller.step(self.policy_time, rs)
                self.policy_time += CTRL_DT
                s.command(q_des, dq_des, tau_ff, self.policy_kp, self.policy_kd)
            if s.t >= next_print - 1e-9:
                self.report(label)
                next_print += print_every
            if self.recorder is not None:
                self.recorder.capture(s, self.phase or label)
            for _ in range(int(round(CTRL_DT / PHYS_DT))):
                s.physics_step()
                # KF like the node: desired contact in POLICY, true feet otherwise
                self.kf_update(self.last_contact if self.mode == "POLICY" else s.foot_contacts())
            g, _, _ = s.posture()
            if self.fall_check and (self.task.gravity_error(g) > 1.2 or s.data.xpos[s.base_body][2] < 0.12):
                self.report(label + " FALL")
                return False
        return True

    def report(self, label=""):
        s = self.sim
        g, pb, ro = s.posture()
        (rl, rr) = s.base_vs_feet()
        knee, feet = s.knee_load()
        tau = s.tau_applied
        q = s.q
        extra = ""
        if self.mode == "POLICY":
            c = self.controller
            v_kf = self.est.base_lin_vel_b(s.quat)
            v_true = self.true_base_lin_vel_b()
            extra = (f" v_kf {v_kf} v_true {v_true}"
                     f" s={c.entry_fraction:4.2f} |a|={np.abs(c.last_raw_action).max():4.2f}"
                     f" hdes={c.qp.desired_base_position[2]:.2f}"
                     f" contact={c.contact_generator.desired_contact_state[2:].astype(int)}")
        com = s.com_vs_feet()
        row = dict(t=s.t, mode=self.mode, pitch_back=pb, roll=ro, gyro=s.gyro.copy(),
                   base_z=float(s.data.xpos[s.base_body][2]), rl=rl, rr=rr, knee=knee,
                   feet=feet, q=q.copy(), tau=tau.copy(), kf_h=self.est.base_height, com=com)
        self.log.append(row)
        print(f"{s.t:6.2f} {self.mode:6s} pitch_back {pb:+6.1f} roll {ro:+5.1f} "
              f"gyro_y {s.gyro[1]:+5.2f} z {row['base_z']:.3f} (kf {self.est.base_height:.3f}) "
              f"base-foot fwd RL {rl[0]:+.3f} RR {rr[0]:+.3f} com-foot fwd {com[0]:+.3f} "
              f"knee {knee:5.1f}N feet {feet:5.1f}N "
              f"hind thigh/calf {q[6]:.2f}/{q[10]:.2f} {q[7]:.2f}/{q[11]:.2f} "
              f"tau calf {tau[10]:+5.1f}/{tau[11]:+5.1f}{extra} {label}")


# ---------------------------------------------------------------------------
class Recorder:
    """Off-screen video of the run (MUJOCO_GL=egl on a headless GPU box):
    a camera tracking the base, a caption with the phase and the sim time,
    frames streamed to an mp4 through imageio-ffmpeg."""

    def __init__(self, model, path, fps=50, width=960, height=540, azimuth=135.0,
                 elevation=-12.0, distance=1.9, font=None):
        import imageio
        self.model = model
        # the offscreen framebuffer defaults to 640x480 (<visual><global offwidth/offheight>)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.lookat[:] = [0.0, 0.0, 0.3]
        self.opt = mujoco.MjvOption()
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                         macro_block_size=1, pixelformat="yuv420p")
        self.dt = 1.0 / fps
        self.next_t = 0.0
        self.frames = 0
        self.path = path
        self._font = None
        if font:
            try:
                from PIL import ImageFont
                self._font = ImageFont.truetype(font, max(18, height // 24))
            except Exception as e:  # pragma: no cover - font is cosmetic
                print(f"[record] font {font} not usable ({e}); using PIL default")

    def capture(self, sim, caption):
        if sim.t < self.next_t - 1e-9:
            return
        self.next_t += self.dt
        # keep the camera's look-at slightly above the tracked base so the
        # whole body stays in view when it stands up
        self.renderer.update_scene(sim.data, camera=self.cam, scene_option=self.opt)
        frame = self.renderer.render()
        if caption:
            frame = self._annotate(frame, f"{caption}   t = {sim.t:5.2f} s")
        self.writer.append_data(frame)
        self.frames += 1

    def _annotate(self, frame, text):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return frame
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = self._font
        pad = 10
        if font is not None:
            box = draw.textbbox((pad, pad), text, font=font)
        else:
            box = draw.textbbox((pad, pad), text)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
        draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
        return np.asarray(img)

    def close(self):
        self.writer.close()
        print(f"[record] {self.frames} frames -> {self.path}")


# ---------------------------------------------------------------------------
def check_fk(model):
    s = Sim(model)
    ok = True
    for name, q in (("nominal", DEFAULT_JOINT_POS),
                    ("random", np.array([0.1, -0.2, 0.15, -0.1, 1.2, 2.0, 2.3, 1.0,
                                         -1.4, -2.0, -2.5, -0.9]))):
        s.set_pose(q, np.array([1.0, 0, 0, 0]), 1.0)
        base = s.data.xpos[s.base_body]
        feet_mj = np.array([s.data.geom_xpos[g] - base for g in s.foot_geom])
        feet_fk = kinematics.foot_positions_b(q)
        err = np.abs(feet_mj - feet_fk).max()
        ok &= err < 3e-3
        print(f"{name:8s} max |foot_mj - foot_fk| = {err * 1000:.2f} mm")
        if err >= 3e-3:
            print("  mj:", feet_mj, "\n  fk:", feet_fk)
    # IMU conventions: level robot -> accelerometer reads +g on body z
    s.set_pose(DEFAULT_JOINT_POS, np.array([1.0, 0, 0, 0]), 1.0)
    print("acc (level, at rest):", s.acc, " quat:", s.quat)
    print("FK check", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# FL push pose and ground-truth metrics
# ---------------------------------------------------------------------------
# FL braced low against a box face (base height 0.295 m):
#   det(J_FL) = 0 exactly when the commanded ee z equals the base height (the
#   leg horizontal). At z 0.16 the margin is 13.5 cm: |det| 0.0061 (61x the
#   estimator's 1e-4 skip threshold), cond 2.8, and the force estimate the
#   contact gate depends on stays alive.
#   Torque is never the binding constraint (158 N of +x authority here, 5x the
#   30 N command clip; a 30 N push costs 19 % of the thigh budget, 5 % of the
#   calf), and the low contact keeps the box far from tipping.
#   y stays at the hip line 0.142: pulling the paw inboard to the robot's
#   centreline is reachable only below z ~0.15 and costs 44 deg of hip
#   abduction (16 deg from the limit). The box is offset instead - it is free
#   to place, the leg is not.
FL_PUSH_EE = (0.44, 0.142, 0.16)     # projected com frame [m]


def base_rpy(quat):
    """(roll, pitch, yaw) [rad] of a (w, x, y, z) body->world quaternion (ZYX)."""
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def projected_axes(quat):
    """World-frame unit forward (+x) and left (+y) axes of the JPC's
    gravity-aligned 'projected com' frame: robot_state.base_rot_mat_rp, i.e.
    rp_rotation_from_gravity_b(g_b) with g_b = quat_rotate_inverse(quat,
    [0, 0, -1]) (joint_position_controller.py:106-150). Its x axis is the
    base x axis levelled about the base y axis - NOT the yaw-only frame:
    the two differ once the base rolls."""
    g_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    rp_t = rp_rotation_from_gravity_b(g_b)          # p_base = rp_t @ p_projected
    return quat_rotate(quat, rp_t[:, 0]), quat_rotate(quat, rp_t[:, 1])


class DemoMetrics:
    """100 Hz ground-truth samples of a run, tagged with the current phase:
    base roll/pitch, heading speed and base height."""

    def __init__(self):
        self.rows = []
        self.phase_id = -1
        self.phase_name = ""

    def begin_phase(self, name):
        self.phase_id += 1
        self.phase_name = name

    def sample(self, h):
        s = h.sim
        roll, pitch, _ = base_rpy(s.quat)
        fwd, _ = projected_axes(s.quat)
        self.rows.append(dict(
            t=s.t, phase=self.phase_id, roll=roll, pitch=pitch,
            vx=float(s.data.qvel[0:3] @ fwd), z=float(s.data.xpos[s.base_body][2])))


# ---------------------------------------------------------------------------
# wam_chunks: the WAM bridge's execution model in MuJoCo
class MujocoRuntime:
    """wam_go2.wam.episode.Runtime over the harness: sim time is the clock, the
    cameras are synthetic frames at the real rates (the mock policy ignores
    pixels), the state is what the node publishes on rambo/state (KF height and
    body velocity, IMU quaternion and gyro) and commands reach the controller the
    way the node applies them (velocity every tick, FL target when it changes).
    Between chunks the policy's latency is real sim time with the hold command."""

    def __init__(self, h, metrics, print_every, ego_period=1.0 / 60.0, d435i_period=1.0 / 60.0):
        # both views as deployed (ego + D435i) at ~60 fps, 640x360 (the policy's dual-rgb-360p-v1 input)
        from wam_go2.wam.observation import FrameRecord, ObservationBuffers, StateRecord
        self.FrameRecord, self.StateRecord = FrameRecord, StateRecord
        self.h, self.s, self.m = h, h.sim, metrics
        self.print_every = print_every
        self.buffers = ObservationBuffers(keep_s=3.0)
        self.vel = [0.0, 0.0, 0.0]
        self.ee = None
        self.fell = False
        self.events = []
        self.commands = []                      # (sim time, native9 or None)
        self._next = {"ego": self.s.t, "task_centric": self.s.t}
        self._period = {"ego": ego_period, "task_centric": d435i_period}
        self._ids = {"ego": 0, "task_centric": 0}
        self._ego_img = np.full((360, 640, 3), 90, np.uint8)
        self._d_img = np.full((360, 640, 3), 150, np.uint8)
        self.label = "wam"

    def _tick(self, hh, t):
        hh.controller.set_commands(vel_cmd=list(self.vel))
        for cam in ("ego", "task_centric"):
            while self._next[cam] <= t + 1e-9:
                self._ids[cam] += 1
                img = self._ego_img if cam == "ego" else self._d_img
                self.buffers.add_frame(cam, self.FrameRecord(self._next[cam], self._ids[cam], img, self._next[cam] + 0.08))
                self._next[cam] += self._period[cam]
        s = self.s
        action = getattr(hh.controller, "_last_action", None)
        self.buffers.add_state(self.StateRecord(
            time=t, height=float(hh.est.base_height), quat_wxyz=tuple(float(v) for v in s.quat),
            lin_vel_b=tuple(float(v) for v in hh.est.base_lin_vel_b(s.quat)),
            ang_vel_b=tuple(float(v) for v in s.gyro), mode=hh.mode, fresh=True,
            action18=None if action is None else [float(v) for v in np.asarray(action).reshape(-1)]))
        self.m.sample(hh)

    # Runtime protocol ----------------------------------------------------------
    def now(self):
        return self.s.t

    def sleep_until(self, t):
        duration = t - self.s.t
        if duration > 1e-9 and not self.fell:
            if not self.h.run(duration, self._tick, self.print_every, self.label):
                self.fell = True

    def command(self, native9):
        if native9 is None:
            self.vel = [0.0, 0.0, 0.0]
        else:
            self.vel = [float(native9[0]), float(native9[1]), float(native9[2])]
            ee = [float(native9[3]), float(native9[4]), float(native9[5])]
            if ee != self.ee:
                self.ee = ee
                self.h.controller.set_commands(ee_pos_cmd=ee, leg_slot=0)
        self.commands.append((self.s.t, None if native9 is None else [float(v) for v in native9]))

    def step_ok(self):
        if self.fell:
            return False, "fell"
        return (self.h.mode == "POLICY", "" if self.h.mode == "POLICY" else f"mode {self.h.mode}")

    def ready(self, grid_wall, tolerances):
        return self.buffers.ready(grid_wall, tolerances)

    def resolve(self, **kw):
        return self.buffers.resolve(**kw)

    def last_action18(self):
        action = getattr(self.h.controller, "_last_action", None)
        return None if action is None else [float(v) for v in np.asarray(action).reshape(-1)]

    def latest_ego_time(self):
        return self.buffers.latest_ego_time()

    def record(self, event, payload):
        self.events.append((self.s.t, event, payload))
        if event in ("hold", "executed", "rejected", "error", "feedback_terminal"):
            print(f"   [wam] t {self.s.t:6.2f} {event}: " + ", ".join(f"{k}={v}" for k, v in payload.items()
                                                                  if k not in ("actions", "reply"))[:160])

    def interrupted(self):
        return False


def run_wam_chunks(h, args):
    """STAND -> POLICY with the FL push pose and a box ahead of the paw -> the WAM
    bridge's episode driver with a mock policy (constant vx / FL target) and
    --infer-delay seconds of simulated latency per RPC. Returns a result dict."""
    import sys as _sys
    qlm_src = os.environ.get("QLM_BENCH_SRC", "")        # the src/ of a QLM-Bench checkout
    if qlm_src and os.path.isdir(qlm_src) and qlm_src not in _sys.path:
        _sys.path.insert(0, qlm_src)
    from wam_go2.wam import mock_service as ms
    from wam_go2.wam.episode import EpisodeDriver
    s = h.sim
    m = DemoMetrics()
    ee = np.asarray(args.ee if args.ee is not None else FL_PUSH_EE, dtype=float)
    print(f"=== wam_chunks: {args.chunks} chunks x 16 commands, mock policy vx {args.vx} ee {ee}, "
          f"inference latency {args.infer_delay} s per RPC (hold), settle {args.settle} s gap {args.gap} s, "
          f"box at {args.box_distance} m")
    s.set_pose(DEFAULT_JOINT_POS, np.array([1.0, 0.0, 0.0, 0.0]), 0.4)
    s.drop_to_floor(0.005)
    s.command(DEFAULT_JOINT_POS, kp=STAND_KP, kd=STAND_KD)
    h.mode = "HOLD"
    h.phase = "STAND"
    m.begin_phase("stand")
    ok = h.run(1.0, lambda hh, t: m.sample(hh), args.print_every, "stand")
    s.place_box(s.data.xpos[s.base_body][0] + args.box_distance, y=args.box_y, yaw=0.0)
    h.enter_policy()
    h.controller.set_commands(ee_pos_cmd=ee, leg_slot=0)
    h.phase = "POLICY settle"
    m.begin_phase("policy settle")
    ok = ok and h.run(args.settle + args.gap, lambda hh, t: m.sample(hh), args.print_every, "settle")
    box_start, _ = s.box_state()
    base_start = s.data.xpos[s.base_body].copy()
    runtime = MujocoRuntime(h, m, args.print_every)
    m.begin_phase("wam chunks")
    h.phase = "WAM chunks"
    from wam_go2.wam import settings
    service = ms.MockPolicyService(ms.ConstantPolicy(vx=args.vx, ee=tuple(ee)), instruction=settings.INSTRUCTION)
    client = ms.LocalClient(service, latency_s=args.infer_delay,
                            advance_clock=lambda dt: runtime.sleep_until(runtime.now() + dt))
    driver = EpisodeDriver(runtime, client, session_id="mujoco", episode_id="mujoco:1", reset_epoch=1,
                           instruction=settings.INSTRUCTION, sample_wait_s=args.sample_wait,
                           max_chunks=args.chunks)
    result = None
    error = None
    if ok:
        try:
            driver.reset_episode()
            result = driver.run()
        except Exception as exc:          # a protocol/driver failure is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"
    box_end, box_yaw = s.box_state()
    base_end = s.data.xpos[s.base_body].copy()
    rows = [r for r in m.rows if r["phase"] == m.phase_id]
    vx = np.array([r["vx"] for r in rows]) if rows else np.zeros(0)
    moving = [c for c in runtime.commands if c[1] is not None]
    holds = [e[2]["duration_s"] for e in runtime.events if e[1] == "hold"]
    fb = [e for e in service.events if e["event"] == "feedback_commit"]
    return dict(ok=bool(ok and not runtime.fell and error is None and result is not None and result.exit_code == 0),
                fell=runtime.fell, error=error, result=None if result is None else result.__dict__,
                chunks_fed_back=len(fb), commands=len(moving), holds_s=holds,
                box_travel_m=None if box_start is None else float(box_end[0] - box_start[0]),
                box_side_m=None if box_start is None else float(box_end[1] - box_start[1]),
                box_yaw_deg=None if box_yaw is None else float(np.degrees(box_yaw)),
                robot_travel_m=float(base_end[0] - base_start[0]), sim_time_s=float(s.t),
                vx_mean=float(vx.mean()) if vx.size else 0.0, vx_max=float(vx.max()) if vx.size else 0.0,
                roll_max_deg=float(np.degrees(max(abs(r["roll"]) for r in rows))) if rows else 0.0,
                pitch_max_deg=float(np.degrees(max(abs(r["pitch"]) for r in rows))) if rows else 0.0,
                z_min=float(min(r["z"] for r in rows)) if rows else 0.0,
                samples=[e[2] for e in runtime.events if e[1] == "sample"][:3])


def print_wam_chunks_result(res):
    print("\n=== wam_chunks RESULT: " + ("OK" if res["ok"] else "FAILED"))
    r = res["result"] or {}
    print(f"  episode: {r.get('reason')} ({r.get('detail')}), exit {r.get('exit_code')}, chunks fed back {res['chunks_fed_back']}, "
          f"commands {res['commands']}, ticks {r.get('ticks')}; fell {res['fell']}; error {res['error']}")
    if res["holds_s"]:
        print(f"  holds between chunks: n {len(res['holds_s'])}, mean {np.mean(res['holds_s']):.2f} s, max {max(res['holds_s']):.2f} s")
    print(f"  robot travel {res['robot_travel_m']:+.2f} m, box travel {res['box_travel_m']} m (side {res['box_side_m']}, "
          f"yaw {res['box_yaw_deg']} deg), vx mean {res['vx_mean']:+.3f} max {res['vx_max']:+.3f} m/s over {res['sim_time_s']:.1f} s sim")
    print(f"  posture: |roll| max {res['roll_max_deg']:.1f} deg, |pitch| max {res['pitch_max_deg']:.1f} deg, z min {res['z_min']:.3f} m")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["check-fk", "quad", "wam_chunks"])
    # wam_chunks: the WAM bridge's chunked, stop-and-wait execution through
    # the real controller code, with a mock policy and simulated inference latency
    ap.add_argument("--chunks", type=int, default=10, help="wam_chunks: chunks to execute (16 commands each)")
    ap.add_argument("--infer-delay", type=float, default=1.0,
                    help="wam_chunks: simulated policy latency per RPC [s] (predict and feedback each); "
                         "the robot holds zero velocity meanwhile, sim time keeps running")
    ap.add_argument("--sample-wait", type=float, default=0.3, help="wam_chunks: bounded wait for sample frames [s]")
    ap.add_argument("--ee", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="wam_chunks: FL ee target of the mock policy, projected com frame [m] "
                         "(default 0.44 0.142 0.16 = FL braced low against the box face)")
    ap.add_argument("--vx", type=float, default=0.15, help="wam_chunks: mock policy walk speed [m/s]")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="wam_chunks: extra POLICY time after --settle before the first chunk [s]")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="wam_chunks: seconds in POLICY before --gap; the ee target is in force "
                         "from entry and FL lifts to it when freed at 1.0 s (so >= 1.0)")
    # The box. Defaults describe the article the operator is asked to prepare:
    # a ~4 kg carton with a smooth sheet taped underneath. On the bare
    # high-friction mat (mu ~0.7) it needs ~27 N to slide and ~35 N to break
    # away (the 30 N figure is only the policy's feedforward command clip, not a
    # bound on position-stiffness contact force), and static tipping would need
    # a contact above 0.21 m - the 0.16 m paw is below that. At mu 0.30 it needs
    # ~12 N. MuJoCo agrees with the arithmetic: 11 N does not move it, 12 N does.
    ap.add_argument("--box-mass", type=float, default=4.0, help="wam_chunks: box mass [kg]")
    ap.add_argument("--box-size", type=float, nargs=3, default=(0.30, 0.35, 0.35),
                    metavar=("DEPTH", "WIDTH", "HEIGHT"), help="wam_chunks: box size [m]")
    ap.add_argument("--box-friction", type=float, default=0.30,
                    help="wam_chunks: friction of the box's BOTTOM against the floor (the taped-on "
                         "sheet). This is what decides the push force: mu*m*g (default 0.30)")
    ap.add_argument("--box-shell-friction", type=float, default=0.8,
                    help="wam_chunks: friction of the box's sides. The paw contact actually uses the "
                         "FOOT's parameters (priority 1), so this only matters for other contacts")
    ap.add_argument("--box-distance", type=float, default=0.62,
                    help="wam_chunks: box centre ahead of the base at the start [m] (default 0.62 = "
                         "the paw at x 0.44 just clear of a 0.30 m deep box's near face)")
    ap.add_argument("--box-y", type=float, default=0.142,
                    help="wam_chunks: box centre offset to the robot's left [m]. Defaults to the FL "
                         "hip line so the paw lands on the box's CENTRE - the paw cannot reach the "
                         "robot's centreline at this pose, so the BOX is offset instead. Pushing "
                         "off-centre yaws the box out from under the paw (measured: 68 deg)")
    ap.add_argument("--box-dof", choices=("free", "planar"), default="free",
                    help="wam_chunks: 'free' (default) is an unconstrained box - honest, and it spins "
                         "out from under the paw in a few seconds, because a single point contact "
                         "cannot control a sliding box's heading (the push and the box's bottom "
                         "friction form a couple as soon as the contact is ~1 cm off the box's "
                         "centre line, and a sliding box has almost no yaw resistance left). "
                         "'planar' models a box the OPERATOR has constrained against rotation "
                         "(guides / a rail / a tray): it slides in x and y but cannot yaw or tip")
    ap.add_argument("--record", type=str, default=None,
                    help="write an mp4 of the run (needs MUJOCO_GL=egl or osmesa and imageio-ffmpeg)")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--size", type=str, default="960x540", help="video WxH")
    ap.add_argument("--font", type=str, default=None, help="TTF font for the caption")
    ap.add_argument("--cam-azimuth", type=float, default=135.0)
    ap.add_argument("--cam-elevation", type=float, default=-12.0)
    ap.add_argument("--cam-distance", type=float, default=1.9)
    ap.add_argument("--duration", type=float, default=4.0, help="quad: seconds of POLICY")
    ap.add_argument("--print-every", type=float, default=0.05)
    ap.add_argument("--knee-radius", type=float, default=KNEE_RADIUS)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="policy checkpoint to load instead of the bundled one for the task "
                         "(a .pt file)")
    ap.add_argument("--joint-damping", type=float, default=0.5,
                    help="passive joint damping [Nm s/rad]; training had none, the menagerie file "
                         "2 (slows balance corrections 3x); 0.5 = mild gear/back-EMF damping")
    ap.add_argument("--true-vel", action="store_true",
                    help="feed the controller the sim's true base velocity/height (bypass the KF)")
    ap.add_argument("--joint-friction", type=float, default=0.2,
                    help="joint frictionloss [Nm]; training 0, menagerie 0.2 (default)")
    ap.add_argument("--com-shift", type=float, default=0.0, metavar="M",
                    help="move the trunk centre of mass this far toward the belly (body -z) "
                         "(model sensitivity check)")
    ap.add_argument("--foot-friction", type=float, default=None,
                    help="sliding friction of floor + feet + knees (default: menagerie foot 0.8 / knee 1.0); "
                         "0.6 rubber-ish, 0.3-0.4 smooth wood")
    args = ap.parse_args()
    box = None
    if args.scenario == "wam_chunks":
        dx, dy, dz = args.box_size
        box = BoxSpec(mass=args.box_mass, dx=dx, dy=dy, dz=dz,
                      mu_bottom=args.box_friction, mu_shell=args.box_shell_friction,
                      x=args.box_distance, y=args.box_y, dof=args.box_dof)
    model = load_model(args.knee_radius, args.joint_damping, args.joint_friction, args.foot_friction,
                       com_shift=args.com_shift, box=box)
    if args.scenario == "check-fk":
        sys.exit(0 if check_fk(model) else 1)

    if args.scenario == "quad":
        # harness validity anchor: the QUADRUPED policy (verified on the real
        # robot) from a level four-legged stand, zero velocity command
        h = Harness(Sim(model), task=QUADRUPED, policy_path=args.checkpoint)
        s = h.sim
        s.set_pose(DEFAULT_JOINT_POS, np.array([1.0, 0.0, 0.0, 0.0]), 0.4)
        s.drop_to_floor(0.005)
        s.command(DEFAULT_JOINT_POS, kp=STAND_KP, kd=STAND_KD)
        h.mode = "HOLD"
        h.run(1.0, print_every=0.5, label="stand")
        print("--- POLICY (quadruped)")
        h.enter_policy()
        ok = h.run(args.duration, None, args.print_every, "quad")
        g = quat_rotate_inverse(s.quat, np.array([0.0, 0.0, -1.0]))
        print(f"\n=== RESULT: {'STANDING' if ok else 'FELL'} at t={s.t:.2f}: gravity_b {g}, "
              f"base z {s.data.xpos[s.base_body][2]:.3f} m, gravity error {QUADRUPED.gravity_error(g):.2f}")
        return

    if args.scenario == "wam_chunks":
        if args.chunks < 1 or args.infer_delay < 0.0 or args.settle < 1.0:
            ap.error("--chunks >= 1, --infer-delay >= 0 and --settle >= 1.0 required")
        h = Harness(Sim(model), task=QUADRUPED, policy_path=args.checkpoint)
        h.true_vel = args.true_vel
        if args.record:
            w, hh_ = (int(v) for v in args.size.lower().split("x"))
            h.recorder = Recorder(model, args.record, fps=args.fps, width=w, height=hh_,
                                  azimuth=args.cam_azimuth, elevation=args.cam_elevation,
                                  distance=args.cam_distance, font=args.font)
        try:
            res = run_wam_chunks(h, args)
        finally:
            if h.recorder is not None:
                h.recorder.close()
        print_wam_chunks_result(res)
        sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
