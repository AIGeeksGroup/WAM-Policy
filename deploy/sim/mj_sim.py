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
  quad_demo   the scripted controller checks of scripts/demo_sequence.py (stand_fl_up, plate_walk,
              handshake_stand, box_push) with the sequencer's timeline; prints front-left tracking,
              base drift and tilt, compliance displacement (handshake_stand)
  wam_chunks  the WAM-Policy bridge's execution model: STAND -> POLICY -> the bridge's episode driver
              with a scripted mock policy and --infer-delay seconds of simulated latency per request
              (needs the QLM-Bench live package: QLM_BENCH_SRC)

    python3 sim/mj_sim.py quad_demo --demo plate_walk --plate-mass 0.5 --walk-time 10
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
    JOINT_POS_LOWER,
    JOINT_TORQUE_LIMIT,
    POS_LIMIT_MARGIN,
)
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.heading_hold import HeadingHold  # noqa: E402
from wam_go2.control.math_utils import (  # noqa: E402
    quat_rotate,
    quat_rotate_inverse,
    rp_rotation_from_gravity_b,
)
from wam_go2.control.policy import RamboPolicy  # noqa: E402
from wam_go2.control.force_command import (  # noqa: E402
    ForceCommandWatchdog, push_motion_phases,
)
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

# Pushable box (quadruped box_push demo). dx is the DEPTH along the push
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
    push slides first; a push at the PLATE pose (contact 0.23-0.30 m) can tip
    it. A slippery bottom sheet at mu 0.30 drops the
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
        # external force hook: None or callable(sim) -> (force_world (3,),
        # point_world (3,), body_id), applied through mj_applyFT every
        # physics sub-step (see plate_load / handshake)
        self.external_force = None
        self.last_external_force = np.zeros(3)
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

    def box_push_force(self):
        """Magnitude of the contact force between any ROBOT geom and the box
        [N], 0 when they are not touching. Ground truth the real robot has no
        way to measure - used to score the demo, never to drive it."""
        if self.box_geoms is None:
            return 0.0
        d, total = self.data, np.zeros(3)
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            hit = (g1 in self.box_geoms) ^ (g2 in self.box_geoms)
            if not hit:
                continue
            other = g2 if g1 in self.box_geoms else g1
            if other == self.floor:
                continue
            mujoco.mj_contactForce(self.model, d, i, buf)
            f_world = c.frame.reshape(3, 3).T @ buf[:3]
            total += f_world if g2 in self.box_geoms else -f_world
        return float(np.linalg.norm(total))

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
        # external force: rebuilt from scratch every sub-step so nothing
        # accumulates in qfrc_applied once the generator is removed
        self.data.qfrc_applied[:] = 0.0
        self.last_external_force[:] = 0.0
        if self.external_force is not None:
            force, point, body = self.external_force(self)
            force = np.asarray(force, dtype=float)
            if np.any(force != 0.0):
                mujoco.mj_applyFT(self.model, self.data, force, np.zeros(3),
                                  np.asarray(point, dtype=float), int(body),
                                  self.data.qfrc_applied)
                self.last_external_force[:] = force
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
        # motor gains while in POLICY: the node's 40/1, or (12,) RAMBO-order
        # arrays (set_ee_compliance lowers the manipulation-leg joints only)
        self.policy_kp = POLICY_KP
        self.policy_kd = POLICY_KD
        self.ee_compliant = False
        # box_push: the node's FL force command watchdog. None for every other
        # demo (the force command then stays zero).
        self.force_watchdog = None  # control/force_command.ForceCommandWatchdog
        self.push_moving = False
        self.push_target = 0.0      # the force the demo asks for [N] (zero)
        self.push_cmd = 0.0         # what the watchdog let through [N]

    def push_step(self):
        """One tick of the FL force command, exactly as controller_node does it
        (control/force_command.py): the demand passes the staleness watchdog and
        becomes the controller's ee force command."""
        if self.force_watchdog is None:
            return
        s = self.sim
        command = self.force_watchdog.step([self.push_target, 0.0, 0.0],
            self.push_moving, now=s.t, command_time=s.t)
        self.push_cmd = float(command[0])
        self.controller.set_commands(ee_force_cmd=command, leg_slot=0)

    def set_ee_compliance(self, on, kp=15.0, kd=0.8, soft=True):
        """Mirror of the node's rambo/ee_compliance: ON lowers the PD gains of
        the manipulation-leg joints (task.ee_joint_indices: FL hip/thigh/calf
        for the quadruped) from 40/1 to kp/kd, the other nine joints keep
        40/1, and with `soft` (the node's ee_compliant_soft, default True)
        sets controller.ee_soft: no policy joint residual and no QP torque on
        the leg, gravity compensation only. OFF restores 40/1 everywhere and
        clears ee_soft (node: _set_ee_compliance)."""
        kp_all = np.full(12, POLICY_KP)
        kd_all = np.full(12, POLICY_KD)
        if on:
            kp_all[self.task.ee_joint_indices] = float(kp)
            kd_all[self.task.ee_joint_indices] = float(kd)
            print(f"--- ee compliance ON (kp {float(kp):g} kd {float(kd):g} on joints "
                  f"{self.task.ee_joint_indices}, soft {'ON' if soft else 'OFF'}) at t={self.sim.t:.2f}")
        elif self.ee_compliant:
            print(f"--- ee compliance OFF at t={self.sim.t:.2f}")
        self.policy_kp, self.policy_kd = kp_all, kd_all
        self.ee_compliant = bool(on)
        self.controller.ee_soft = bool(on and soft)

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
        if self.ee_compliant:          # the node forces compliance OFF on POLICY entry
            self.set_ee_compliance(False)
        self.controller.ee_soft = False
        self.policy_time = 0.0
        if self.force_watchdog is not None:
            self.force_watchdog.reset()
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
                self.push_step()          # no-op unless box_push enabled the driver
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
# quad_demo: external forces on the FL foot, ground-truth metrics, sequencer
# ---------------------------------------------------------------------------
GRAVITY = 9.81
DEMOS = ("stand_fl_up", "plate_walk", "handshake_stand", "box_push")
# A three-leg stand measured on the real robot: base 0.257 m, FL foot directly
# under the hip (base frame x +0.185, y +0.142), 0.15 m above the ground, FL
# hip 0 / thigh 1.40 / calf -2.64 = the calf folded onto its soft limit. Produced by this ee command: the IK would
# need calf -2.67 < the -2.62 clamp, so the controller clamps the calf and the
# leg folds fully - the target itself is not reachable, the clamp is the pose.
FL_FOLDED_EE = (0.1934, 0.142, 0.20)
CALF_CLAMP = float(JOINT_POS_LOWER[8] + POS_LIMIT_MARGIN)      # -2.6227 rad (rambo_controller._pos_lower)
# FL extended forward like the RAMBO paper's plate figure: thigh forward-down,
# shin forward-up, paw ~29 cm ahead of the hip near hip height (real base 0.26 m:
# thigh -0.67 / calf -1.66 rad, paw 24 cm up); 2 cm inside the trained x range.
FL_PLATE_EE = (0.48, 0.142, 0.24)
# FL braced against a box face for box_push. NOT the plate pose: pushing wants
# a different compromise and the numbers say so (base height 0.295 m, measured
# in real plate-walk runs).
#   det(J_FL) = 0 exactly when the commanded ee z equals the base height (the
#   leg horizontal). The plate pose sits 5.5 cm from that surface: |det| 0.0025,
#   cond 6.6, and one real trace spent 7.5 % of its ticks below the estimator's
#   1e-4 skip threshold in runs up to 170 ms. At z 0.16 the margin is 13.5 cm:
#   |det| 0.0061 (61x the floor), cond 2.8 - the best of every pose tried, and
#   the force estimate the contact gate depends on stays alive.
#   Torque is never the binding constraint (158 N of +x authority here, 5x the
#   30 N command clip; a 30 N push costs 19 % of the thigh budget, 5 % of the
#   calf), and the low contact keeps the box far from tipping.
#   y stays at the hip line 0.142: pulling the paw inboard to the robot's
#   centreline is reachable only below z ~0.15 and costs 44 deg of hip
#   abduction (16 deg from the limit). The box is offset instead - it is free
#   to place, the leg is not.
FL_PUSH_EE = (0.44, 0.142, 0.16)
DEMO_EE_DEFAULT = {                  # projected com frame [m]
    "stand_fl_up": FL_FOLDED_EE,              # three-leg stand, FL folded up under the hip
    "plate_walk": FL_PLATE_EE,                # FL out front, the paw carries the tray
    "handshake_stand": (0.32, 0.142, 0.20),   # paw raised to a person's hand
    "box_push": FL_PUSH_EE,                   # FL braced low against a box face
}
ZERO_VEL_HOLD = 0.5                  # demo_sequence.ZERO_VEL_HOLD: zeros held after each walk leg [s]
FL_FREE_AFTER = 1.0                  # demo_sequence.FL_FREE_AFTER: the node frees FL 1 s into POLICY
REF_WINDOW = 3.0                     # no-force reference: at most these last seconds of the ee-hold gap (40/1)
TAIL = 1.0                           # 'steady' = mean over the last TAIL s of a phase


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


def fl_foot_projected(sim):
    """FL foot centre in the JPC's 'projected com' frame from ground truth,
    built like the JPC does (p_proj = R_rp p_body, z += base_height): x/y =
    (foot - base) along projected_axes(); z = the foot's world z (the true
    base height stands in for the KF's; R_rp's third row is world up)."""
    base = sim.data.xpos[sim.base_body]
    foot = sim.data.geom_xpos[sim.foot_geom[0]]
    fwd, left = projected_axes(sim.quat)
    d = foot - base
    return np.array([d @ fwd, d @ left, foot[2]])


def plate_load(mass_kg):
    """External force generator: a plate / object of mass_kg resting on the FL
    paw = constant -m g at the foot centre (world z)."""
    fz = -float(mass_kg) * GRAVITY

    def gen(sim):
        return (np.array([0.0, 0.0, fz]), sim.data.geom_xpos[sim.foot_geom[0]].copy(),
                sim.calf_body[0])
    return gen


def handshake(t0, duration, steady_N=5.0, amp_N=8.0, freq_Hz=2.0, lateral_pulse_N=10.0,
              pulse_t=None, pulse_dur=0.3):
    """External force generator: a person shaking the FL paw from sim time
    t0 for `duration` s, all at the foot centre: a steady push of steady_N
    along the projected frame's forward axis (the frame the FL target lives
    in), TOWARD the body (the person stands in front of the robot and leans
    on the paw; steady_N < 0 pulls the paw forward), a vertical sinusoid
    amp_N sin(2 pi f t) (the pumping), and one lateral pulse of
    lateral_pulse_N along the frame's +y (the robot's left, outward for FL)
    lasting pulse_dur s from pulse_t s after t0 (default: mid-window). Zero
    outside the window."""
    if pulse_t is None:
        pulse_t = 0.5 * duration

    def gen(sim):
        p = sim.data.geom_xpos[sim.foot_geom[0]].copy()
        tau = sim.t - t0
        if tau < 0.0 or tau > duration:
            return np.zeros(3), p, sim.calf_body[0]
        fwd, left = projected_axes(sim.quat)
        f = -float(steady_N) * fwd
        f[2] += float(amp_N) * np.sin(2.0 * np.pi * float(freq_Hz) * tau)
        if pulse_t <= tau < pulse_t + pulse_dur:
            f += float(lateral_pulse_N) * left
        return f, p, sim.calf_body[0]
    return gen


def static_push(t0, duration, force_N=5.0):
    """External force generator: a constant force_N at the FL foot centre
    along the projected frame's forward axis TOWARD the body (horizontal - a
    person leaning on the raised paw), from sim time t0 for duration s, zero
    outside. The probe behind the 'cm per N' compliance number: the steady
    foot displacement it produces / force_N."""
    def gen(sim):
        p = sim.data.geom_xpos[sim.foot_geom[0]].copy()
        tau = sim.t - t0
        if tau < 0.0 or tau > duration or force_N == 0.0:
            return np.zeros(3), p, sim.calf_body[0]
        fwd, _ = projected_axes(sim.quat)
        return -float(force_N) * fwd, p, sim.calf_body[0]
    return gen


class DemoMetrics:
    """100 Hz ground-truth samples of a quad_demo run and their per-phase /
    whole-run summaries (FL tracking error vs the live ee command in the
    projected frame, FL joint angles and motor torque, base roll/pitch/yaw,
    heading speed, base height, |F_ext|, compliance state)."""

    def __init__(self):
        self.rows = []
        self.phase_id = -1
        self.phase_name = ""
        self.forced = False

    def begin_phase(self, name, forced=False):
        self.phase_id += 1
        self.phase_name = name
        self.forced = forced

    def sample(self, h):
        s = h.sim
        fl = fl_foot_projected(s)
        cmd = np.asarray(h.controller.ee_pos_commands[0], dtype=float).copy()
        roll, pitch, yaw = base_rpy(s.quat)
        fwd, left = projected_axes(s.quat)
        fl_free = not bool(h.controller.contact_generator.desired_contact_state[0])
        base = s.data.xpos[s.base_body].copy()
        fl_j = h.task.ee_joint_indices[:3]              # FL hip / thigh / calf (RAMBO order)
        box_pos, box_yaw = s.box_state()
        self.rows.append(dict(
            t=s.t, phase=self.phase_id, forced=self.forced, fl=fl, cmd=cmd, err=fl - cmd,
            fl_free=fl_free, roll=roll, pitch=pitch, yaw=yaw, vx=float(s.data.qvel[0:3] @ fwd),
            base=base, fwd=fwd, left=left, z=float(base[2]),
            fext=float(np.linalg.norm(s.last_external_force)),
            # what the FL motors do: the applied torque (PD + feedforward,
            # clipped) and the feedforward alone (gravity comp [+ QP])
            q_fl=s.q[fl_j], tau_fl=s.tau_applied[fl_j], tau_ff_fl=s.tau_ff[fl_j].copy(),
            compliant=bool(h.ee_compliant),
            # box_push: the box's ground truth and the FL force command. All zero
            # (and box_pos None) for every other demo.
            box_pos=box_pos, box_yaw=box_yaw,
            box_contact_n=s.box_push_force(),
            push_cmd=float(h.push_cmd)))

    def _rows(self, phase=None, free_only=False, t_min=None, forced=None):
        """forced: None = all rows, True = only rows with an external hand /
        probe force active, False = those without it."""
        out = self.rows
        if phase is not None:
            out = [r for r in out if r["phase"] == phase]
        if free_only:
            out = [r for r in out if r["fl_free"]]
        if t_min is not None:
            out = [r for r in out if r["t"] >= t_min - 1e-9]
        if forced is not None:
            out = [r for r in out if r["forced"] == bool(forced)]
        return out

    def phase_tail(self, phase, last):
        """The last `last` seconds of a phase's rows."""
        rows = self._rows(phase)
        if rows:
            t_end = rows[-1]["t"]
            rows = [r for r in rows if r["t"] >= t_end - last - 1e-9]
        return rows

    @staticmethod
    def drift(rows):
        """Horizontal base displacement first -> last row [m] along the
        projected frame's (fwd, left) at the first row; None without samples."""
        if len(rows) < 2:
            return None
        d = rows[-1]["base"] - rows[0]["base"]
        return float(d @ rows[0]["fwd"]), float(d @ rows[0]["left"])

    @staticmethod
    def stats(rows):
        if not rows:
            return None
        err = np.array([r["err"] for r in rows])
        en = np.linalg.norm(err, axis=1)
        fl = np.array([r["fl"] for r in rows])
        tau_fl = np.array([r["tau_fl"] for r in rows])
        tau_ff = np.array([r["tau_ff_fl"] for r in rows])
        yaw = np.array([r["yaw"] for r in rows])
        dyaw = yaw[-1] - yaw[0]
        return dict(
            n=len(rows), err_max=float(en.max()), err_mean=float(en.mean()),
            err_mean_xyz=err.mean(axis=0), err_max_abs_xyz=np.abs(err).max(axis=0),
            p2p_xyz=fl.max(axis=0) - fl.min(axis=0),
            fl_mean=fl.mean(axis=0), fl_z_min=float(fl[:, 2].min()),
            q_fl_mean=np.mean([r["q_fl"] for r in rows], axis=0),
            tau_fl_peak=np.abs(tau_fl).max(axis=0),                 # per joint, PD + ff
            tau_pd_fl_peak=np.abs(tau_fl - tau_ff).max(axis=0),     # the PD part alone
            tau_ff_fl_mean=np.abs(tau_ff).mean(axis=0),
            roll_max=float(np.degrees(np.abs([r["roll"] for r in rows]).max())),
            pitch_max=float(np.degrees(np.abs([r["pitch"] for r in rows]).max())),
            yaw_change=float(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw)))),
            vx_mean=float(np.mean([r["vx"] for r in rows])),
            z_min=float(np.min([r["z"] for r in rows])),
            z_mean=float(np.mean([r["z"] for r in rows])),
            fext_max=float(np.max([r["fext"] for r in rows])),
            compliant=all(r["compliant"] for r in rows),
            forced=any(r["forced"] for r in rows),
        )

    def phase_summary(self, phase, name):
        st = self.stats(self._rows(phase))
        stf = self.stats(self._rows(phase, free_only=True))
        if st is None:
            return f"[phase] {name}: no samples"
        line = (f"[phase] {name:34s} {st['n'] / 100.0:4.1f}s  roll/pitch max {st['roll_max']:4.1f}/"
                f"{st['pitch_max']:4.1f} deg  vx mean {st['vx_mean']:+.3f} m/s  z min {st['z_min']:.3f}"
                f"  |Fext| max {st['fext_max']:4.1f} N")
        if stf is not None:
            line += (f"\n        FL err (free leg) mean {stf['err_mean'] * 100:.1f} cm max {stf['err_max'] * 100:.1f} cm"
                     f"  mean xyz {stf['err_mean_xyz'] * 100} cm  p2p xyz {stf['p2p_xyz'] * 100} cm"
                     f"  FL |tau| peak {stf['tau_fl_peak']} Nm")
        return line


def box_result(m, args):
    """Score a box_push run. 'ok' from the harness only means the robot did not
    fall; a push has its own success criterion and this is it."""
    rows = [r for r in m.rows if r.get("box_pos") is not None]
    if not rows:
        return None
    start, end = rows[0], rows[-1]
    dz = args.box_size[2]
    moved = float(end["box_pos"][0] - start["box_pos"][0])
    side = float(end["box_pos"][1] - start["box_pos"][1])
    yaw = float(np.degrees(end["box_yaw"] - start["box_yaw"]))
    # tipped: the centre dropped well below its resting height
    tipped = bool(end["box_pos"][2] < start["box_pos"][2] - 0.25 * dz)
    contact = [r["box_contact_n"] for r in rows]
    base_moved = float(end["base"][0] - start["base"][0])
    return dict(
        moved_m=moved, side_m=side, yaw_deg=yaw, tipped=tipped,
        base_moved_m=base_moved,
        slip_m=base_moved - moved,          # how far the robot walked past the box
        contact_max_n=float(max(contact)) if contact else 0.0,
        contact_mean_n=float(np.mean([c for c in contact if c > 1.0])) if any(
            c > 1.0 for c in contact) else 0.0,
        push_max_n=float(max((r["push_cmd"] for r in rows), default=0.0)),
    )


def run_quad_demo(h, args):
    """The quadruped demo as the ROS demo sequencer drives it, in MuJoCo
    (scripts/demo_sequence.py build_plan()): the ee target is in force from
    POLICY entry (the sequencer pre-publishes it; the node's entry reset is
    overwritten by the 20 Hz topic within 50 ms), FL is freed at 1.0 s and
    lifts straight to it -> settle -> gap (the
    plate is put on the paw here, once it is out, as on the robot) -> the
    demo body -> final gap.
    Body: stand_fl_up = the folded FL held for --hold; plate_walk = +vx for
    --walk-time, zeros 0.5 s [--return: gap, -vx --walk-time, zeros 0.5 s];
    handshake_stand = ee held for a gap,
    compliance ON (--soft -> controller.ee_soft), the handshake force for
    --hold, a 'let go' gap (still compliant, no force), the static push
    probe (still compliant, --probe-force toward the body for --probe-time;
    0 skips it), compliance OFF. Instantaneous plan steps (ee, vel,
    compliance) are the setup of the phase that follows.
    Returns a dict: ok, metrics, phases [(name, id)], walk_phases,
    force_phases (a hand on the paw), ref_phase (the ee hold: FL free, no
    force, gains 40/1), probe_phase / probe_ref_phase (the static push and
    the compliant no-force phase right before it; None without a probe),
    ee_set_t, plate_mass, probe_force, comp_txt."""
    s = h.sim
    m = DemoMetrics()
    phases = []
    walk_phases = []          # phase ids of the walking legs
    force_phases = []         # phase ids with the handshake (hand) force active
    state = {"ee_set_t": None, "ref_phase": None, "probe_phase": None, "probe_ref_phase": None,
             "vel": [0.0, 0.0, 0.0], "wz_abs_max": 0.0, "wz_sum": 0.0, "wz_n": 0}
    # the node's heading hold (wam_go2/control/heading_hold.py) on the
    # operator command, same defaults as config/wam_go2.yaml
    hold = HeadingHold(enabled=bool(args.heading_hold), kp=args.heading_hold_kp, ki=args.heading_hold_ki,
                       max_wz=args.heading_hold_max_wz, deadband_deg=args.heading_hold_deadband_deg,
                       dt=CTRL_DT)
    demo = args.demo
    ee = np.asarray(args.ee if args.ee is not None else DEMO_EE_DEFAULT[demo], dtype=float)
    if demo == "box_push":
        # the same object controller_node runs, so a sim result transfers
        h.force_watchdog = ForceCommandWatchdog()
    kp_c, kd_c, soft = args.compliant_kp, args.compliant_kd, bool(args.soft)
    hand_on = any(v != 0.0 for v in (args.hand_steady, args.hand_amp, args.hand_lateral))
    probe_on = demo == "handshake_stand" and args.probe_force != 0.0
    comp_txt = (f"kp {kp_c:g} kd {kd_c:g} soft {'ON' if soft else 'OFF'}" if not args.no_compliance
                else "40/1 (--no-compliance)")
    print(f"=== quad_demo {demo}: ee {ee} vx {args.vx} walk {args.walk_time}s"
          f"{' + return' if args.walk_return else ''} gap {args.gap}s settle {args.settle}s hold {args.hold}s "
          f"compliance {comp_txt} plate {args.plate_mass} kg handshake steady {args.hand_steady} N "
          f"amp {args.hand_amp} N {args.hand_freq} Hz lateral {args.hand_lateral} N "
          f"probe {args.probe_force} N for {args.probe_time}s heading hold "
          f"{'ON' if hold.enabled else 'OFF'} (kp {args.heading_hold_kp:g} ki {args.heading_hold_ki:g} "
          f"max {args.heading_hold_max_wz:g} rad/s)")

    def tick(hh, t):
        if hh.mode == "POLICY":
            # as controller_node._control_tick: the operator command goes
            # through the heading hold every tick, yaw from the IMU quaternion
            cmd = hold.update(state["vel"], base_rpy(s.quat)[2])
            hh.controller.set_commands(vel_cmd=cmd)
            if hold.event == "latched":
                print(f"   [demo] t {t:5.2f} heading hold: latched yaw {np.degrees(hold.yaw_ref):+.1f} deg")
            elif hold.event == "released":
                print(f"   [demo] t {t:5.2f} heading hold: released")
            if hold.active:
                state["wz_abs_max"] = max(state["wz_abs_max"], abs(cmd[2]))
                state["wz_sum"] += cmd[2]
                state["wz_n"] += 1
        m.sample(hh)
        if abs((t / 0.5) - round(t / 0.5)) < 1e-6:
            r = m.rows[-1]
            print(f"   [demo] t {t:5.2f} {m.phase_name[:28]:28s} FL {r['fl']} cmd {r['cmd']} "
                  f"err {r['err'] * 100} cm  vx {r['vx']:+.2f}  roll/pitch {np.degrees(r['roll']):+.1f}/"
                  f"{np.degrees(r['pitch']):+.1f}  z {r['z']:.3f}  |F| {r['fext']:4.1f} N"
                  f"  q_FL {r['q_fl']}  tau_FL {r['tau_fl']}"
                  + ("  FL free" if r["fl_free"] else "  FL stance"))

    def step(duration, caption, label, setup=None, forced=False, hand=False, walking=False):
        """forced: an external force acts on the paw (rows excluded from the
        tracking error); hand: it is the handshake (-> force_phases)."""
        if setup is not None:
            setup()
        h.phase = caption
        m.begin_phase(label, forced)
        phases.append((label, m.phase_id))
        if hand:
            force_phases.append(m.phase_id)
        if walking:
            walk_phases.append(m.phase_id)
        print(f"\n--- t={s.t:.2f} {label}")
        ok = h.run(duration, tick, args.print_every, label)
        print(m.phase_summary(m.phase_id, label))
        if walking and state["wz_n"] > 0:
            print(f"    heading hold during '{label}': wz mean {state['wz_sum'] / state['wz_n']:+.3f} rad/s, "
                  f"|wz| max {state['wz_abs_max']:.3f} rad/s, yaw error at the end "
                  f"{np.degrees(hold.last_error):+.1f} deg")
            state["wz_abs_max"], state["wz_sum"], state["wz_n"] = 0.0, 0.0, 0
        return ok

    def set_ee():
        h.controller.set_commands(ee_pos_cmd=ee, leg_slot=0)
        # the target is in force now, but the paw only starts moving when the
        # node frees FL 1 s into POLICY: tracking statistics count from there
        state["ee_set_t"] = s.t + FL_FREE_AFTER

    def arm_plate():
        # the plate goes on the paw after the settle, when FL is out
        if demo == "plate_walk" and args.plate_mass > 0.0:
            s.external_force = plate_load(args.plate_mass)

    def enter_policy_with_ee():
        h.enter_policy()               # resets the ee command to the default...
        set_ee()                       # ...and the pre-published topic restores it

    ok = True
    # 1) level stand under the STAND gains (the node's STAND end state)
    s.set_pose(DEFAULT_JOINT_POS, np.array([1.0, 0.0, 0.0, 0.0]), 0.4)
    s.drop_to_floor(0.005)
    s.command(DEFAULT_JOINT_POS, kp=STAND_KP, kd=STAND_KD)
    h.mode = "HOLD"
    ok = step(1.0, "STAND", "stand")
    if demo == "box_push":
        # The real box is already present during POLICY settle and approach.
        s.place_box(s.data.xpos[s.base_body][0] + args.box_distance,
                    y=args.box_y, yaw=0.0)
    # 2) POLICY with the FL target already published; FL is freed at 1.0 s
    #    and lifts straight to it
    ok = ok and step(args.settle, f"POLICY (settle, FL target {np.round(ee, 3)} pre-published)",
                     "policy settle", setup=enter_policy_with_ee)

    def set_vx(v, vy=0.0):
        def f():
            # the operator command; tick() passes it through the heading hold
            # and applies it before the first controller step of the phase
            state["vel"] = [v, float(vy), 0.0]
        return f

    def compliance(on):
        def f():
            if args.no_compliance:
                print(f"--- (compliance {'ON' if on else 'OFF'} requested; --no-compliance keeps 40/1)")
            else:
                h.set_ee_compliance(on, kp_c, kd_c, soft)
        return f

    def combo(*fns):
        def f():
            for fn in fns:
                fn()
        return f

    def arm_handshake(duration):
        def f():
            s.external_force = handshake(s.t, duration, args.hand_steady, args.hand_amp,
                                         args.hand_freq, args.hand_lateral)
        return f

    def arm_probe(duration):
        def f():
            s.external_force = static_push(s.t, duration, args.probe_force)
        return f

    def disarm():
        s.external_force = None
        s.last_external_force[:] = 0.0

    plate_txt = f" (plate {args.plate_mass:g} kg)" if (demo == "plate_walk" and args.plate_mass > 0) else ""
    # 3) the sequencer's gap after the settle: FL free at the target, no
    #    force, gains 40/1 = the no-force reference. stand_fl_up then holds
    #    the pose for --hold (the demo itself, also the reference).
    if demo == "stand_fl_up":
        ok = ok and step(args.gap, "pause", "gap after settle")
        ok = ok and step(args.hold, f"three-leg stand, FL folded up {np.round(ee, 3)}",
                         "ee target (hold)")
    else:
        ok = ok and step(args.gap, f"FL at the target {np.round(ee, 3)}{plate_txt}, pause",
                         "ee target (hold)", setup=arm_plate)
    if ok:
        state["ref_phase"] = m.phase_id
    # 4) demo body + the final gap (build_plan: ... -> [let go gap ->]
    #    compliance OFF -> gap -> DAMP; compliance OFF is instantaneous and
    #    therefore the setup of the final gap)
    if ok and demo == "plate_walk":
        ok = step(args.walk_time, f"plate walk +{args.vx:g} m/s{plate_txt}", "walk fwd",
                  setup=set_vx(+args.vx), walking=True)
        ok = ok and step(ZERO_VEL_HOLD, "stop: zero velocity", "zeros after fwd",
                         setup=set_vx(0.0))
        if args.walk_return:
            ok = ok and step(args.gap, "pause", "gap between legs")
            ok = ok and step(args.walk_time, f"plate walk -{args.vx:g} m/s{plate_txt}", "walk back",
                             setup=set_vx(-args.vx), walking=True)
            ok = ok and step(ZERO_VEL_HOLD, "stop: zero velocity", "zeros after back",
                             setup=set_vx(0.0))
    elif ok and demo == "box_push":
        # Same phase source as demo_sequence.build_plan: push with the leg's
        # position stiffness (zero force), stop, then hold zeros for 0.5 s.
        for phase in push_motion_phases(args.vx, 0.0, args.walk_time,
                                        vy=args.push_vy):
            def apply_push_phase(phase=phase):
                state["vel"] = list(phase.velocity)
                h.push_target = phase.force[0]
                h.push_moving = bool(np.linalg.norm(phase.velocity[:2]) > 1e-6)
            moving = phase.velocity[0] != 0.0
            ok = ok and step(phase.seconds,
                "three-leg box push" if moving else "stop and release",
                "box push" if moving else "zeros after push", setup=apply_push_phase,
                walking=moving)
    elif ok and demo == "handshake_stand":
        if hand_on:
            caption = f"handshake, compliance ON ({comp_txt})"
        else:
            caption = f"no hand, compliance ON ({comp_txt}, sag test)"
        ok = step(args.hold, caption, "handshake",
                  setup=combo(compliance(True), arm_handshake(args.hold)), forced=hand_on, hand=hand_on)
        ok = ok and step(args.gap, "let go, still compliant", "let-go gap", setup=disarm)
        if ok and probe_on:
            state["probe_ref_phase"] = m.phase_id
            ok = step(args.probe_time, f"static push probe {args.probe_force:g} N",
                      "static probe", setup=arm_probe(args.probe_time), forced=True)
            if ok:
                state["probe_phase"] = m.phase_id
        ok = ok and step(args.gap, "compliance OFF, pause (end)", "final gap",
                         setup=combo(disarm, compliance(False)))
    if ok and demo != "handshake_stand":
        ok = step(args.gap, "pause (end)", "final gap")
    s.external_force = None
    if os.environ.get("QUAD_DEMO_DUMP"):
        np.savez_compressed(os.environ["QUAD_DEMO_DUMP"],
                            **{k: np.array([r[k] for r in m.rows])
                               for k in ("t", "phase", "box_yaw", "box_contact_n",
                                         "push_cmd")},
                            box_pos=np.array([r["box_pos"] if r["box_pos"] is not None
                                              else [np.nan] * 3 for r in m.rows]),
                            base=np.array([r["base"] for r in m.rows]),
                            fl=np.array([r["fl"] for r in m.rows]))
    ee_set_t = state["ee_set_t"] if state["ee_set_t"] is not None else s.t
    return dict(ok=ok, metrics=m, phases=phases, walk_phases=walk_phases, force_phases=force_phases,
                ref_phase=state["ref_phase"], probe_phase=state["probe_phase"],
                probe_ref_phase=state["probe_ref_phase"], ee_set_t=ee_set_t,
                box=box_result(m, args) if demo == "box_push" else None,
                plate_mass=float(args.plate_mass) if demo == "plate_walk" else 0.0,
                probe_force=float(args.probe_force) if probe_on else 0.0, comp_txt=comp_txt)


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
    """STAND -> POLICY with the FL push pose (box placed as in box_push) -> the WAM
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
    ee = np.asarray(args.ee if args.ee is not None else DEMO_EE_DEFAULT["box_push"], dtype=float)
    print(f"=== wam_chunks: {args.chunks} chunks x 16 commands, mock policy vx {args.vx} ee {ee}, "
          f"inference latency {args.infer_delay} s per RPC (hold), settle {args.settle} s gap {args.gap} s, "
          f"box at {args.box_distance} m")
    s.set_pose(DEFAULT_JOINT_POS, np.array([1.0, 0.0, 0.0, 0.0]), 0.4)
    s.drop_to_floor(0.005)
    s.command(DEFAULT_JOINT_POS, kp=STAND_KP, kd=STAND_KD)
    h.mode = "HOLD"
    h.phase = "STAND"
    m.begin_phase("stand", False)
    ok = h.run(1.0, lambda hh, t: m.sample(hh), args.print_every, "stand")
    s.place_box(s.data.xpos[s.base_body][0] + args.box_distance, y=args.box_y, yaw=0.0)
    h.enter_policy()
    h.controller.set_commands(ee_pos_cmd=ee, leg_slot=0)
    h.phase = "POLICY settle"
    m.begin_phase("policy settle", False)
    ok = ok and h.run(args.settle + args.gap, lambda hh, t: m.sample(hh), args.print_every, "settle")
    box_start, _ = s.box_state()
    base_start = s.data.xpos[s.base_body].copy()
    runtime = MujocoRuntime(h, m, args.print_every)
    m.begin_phase("wam chunks", False)
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


def print_quad_demo_result(h, res):
    """Summary of a run_quad_demo() result dict. Every block tolerates phases
    without samples (a fall, --gap 0) and says 'no samples' instead. Returns
    the headline numbers as a flat dict (for sweeps)."""
    s = h.sim
    m = res["metrics"]
    ok, ee_set_t = res["ok"], res["ee_set_t"]
    name_of = {pid: name for name, pid in res["phases"]}
    foot_r = float(s.model.geom_size[s.foot_geom[0]][0])      # foot sphere: height above ground = z - r
    g = quat_rotate_inverse(s.quat, np.array([0.0, 0.0, -1.0]))
    roll, pitch, _ = base_rpy(s.quat)
    out = dict(ok=bool(ok), t_end=float(s.t))
    print(f"\n=== RESULT: {'STANDING' if ok else 'FELL'} at t={s.t:.2f}: roll {np.degrees(roll):+.1f} "
          f"pitch {np.degrees(pitch):+.1f} deg, base z {s.data.xpos[s.base_body][2]:.3f} m, "
          f"gravity error {QUADRUPED.gravity_error(g):.2f}")
    if not m.rows:
        print("    no samples")
        return out
    b = res.get("box")
    if b is not None:
        verdict = ("PUSHED" if (b["moved_m"] > 0.15 and not b["tipped"]
                                and abs(b["yaw_deg"]) < 15.0) else "NOT PUSHED")
        print(f"\n--- box: {verdict}")
        print(f"    moved {b['moved_m']:+.3f} m (robot {b['base_moved_m']:+.3f} m, "
              f"robot past the box {b['slip_m']:+.3f} m)")
        print(f"    sideways {b['side_m']:+.3f} m   yaw {b['yaw_deg']:+.1f} deg   "
              f"{'TIPPED' if b['tipped'] else 'not tipped'}")
        print(f"    contact force (sim ground truth) peak {b['contact_max_n']:.1f} N, mean while in contact "
              f"{b['contact_mean_n']:.1f} N")
        out.update({f"box_{k}": v for k, v in b.items()})
    policy = m.stats(m._rows(t_min=m.rows[0]["t"] + 1.0))        # after STAND
    # tracking error: the free leg, after the move to the target, WITHOUT
    # the hand / probe windows (a hand on the paw makes 'error' meaningless);
    # the plate load is the plate_walk demo itself and stays included
    steady = m.stats(m._rows(free_only=True, t_min=ee_set_t + 1.0, forced=False))
    trans = m.stats(m._rows(free_only=True, t_min=ee_set_t, forced=False))
    under_force = m.stats(m._rows(free_only=True, forced=True))
    load_txt = f", under the {res['plate_mass']:g} kg plate" if res["plate_mass"] > 0 else ""
    if steady is not None:
        print(f"    FL pos error vs ee cmd (free leg, from ee target + 1 s, hand windows excluded{load_txt}): "
              f"max {steady['err_max'] * 100:.1f} cm mean {steady['err_mean'] * 100:.1f} cm, "
              f"mean xyz {steady['err_mean_xyz'] * 100} cm, max |xyz| {steady['err_max_abs_xyz'] * 100} cm "
              f"({steady['n'] / 100.0:.1f} s)")
    else:
        print("    FL pos error vs ee cmd (free leg, no force): no samples")
    if trans is not None:
        print(f"    FL pos error incl. the move to the target (no force): max {trans['err_max'] * 100:.1f} cm")
    if under_force is not None:
        print(f"    FL pos error vs ee cmd UNDER an external force (hand / probe, separate): "
              f"max {under_force['err_max'] * 100:.1f} cm mean {under_force['err_mean'] * 100:.1f} cm, "
              f"mean xyz {under_force['err_mean_xyz'] * 100} cm ({under_force['n'] / 100.0:.1f} s)")
    # the no-force reference: the last REF_WINDOW s of the ee hold before
    # anything else happens (FL free, gains 40/1, no force), minus the first
    # second after the ee command (the move to the target, excluded like the
    # headline: with --gap 3 the whole gap is 'the last 3 s')
    ref_pid = res["ref_phase"]
    ref_rows = [r for r in (m.phase_tail(ref_pid, REF_WINDOW) if ref_pid is not None else [])
                if r["t"] >= ee_set_t + 1.0 - 1e-9]
    ref = m.stats(ref_rows)
    # the held FL pose (what the folded / raised leg looks like without a hand)
    held = m.stats(m.phase_tail(ref_pid, TAIL)) if ref_pid is not None else None
    if held is not None:
        q = held["q_fl_mean"]
        off = float(q[2] - CALF_CLAMP)
        at_clamp = abs(off) < 0.03
        print(f"    FL pose held (last {held['n'] / 100.0:.1f} s of '{name_of.get(ref_pid, '?')}', 40/1, no hand): "
              f"foot projected {held['fl_mean']} m = {(held['fl_mean'][2] - foot_r) * 100:.1f} cm above the ground "
              f"(min {(held['fl_z_min'] - foot_r) * 100:.1f} cm); FL hip/thigh/calf {q} rad, "
              f"calf {'AT its clamp' if at_clamp else f'{off:+.3f} rad from its clamp'} ({CALF_CLAMP:.3f}); "
              f"base z mean {held['z_mean']:.3f} m, roll/pitch max {held['roll_max']:.1f}/{held['pitch_max']:.1f} deg")
        out.update(fl_height_cm=(held["fl_mean"][2] - foot_r) * 100, fl_pos=held["fl_mean"], q_fl=q,
                   calf_at_clamp=at_clamp, base_z=held["z_mean"])
    for pid in res["force_phases"]:
        rows = m._rows(pid)
        st = m.stats(rows)
        name = name_of.get(pid, f"phase {pid}")
        if st is None:
            print(f"    FL displacement during '{name}': no samples")
            continue
        p2p = st["p2p_xyz"]
        out.update(p2p_max_cm=p2p.max() * 100)
        print(f"    FL displacement during '{name}' ({st['n'] / 100.0:.1f} s, peak-to-peak xyz): {p2p * 100} cm, "
              f"max {p2p.max() * 100:.1f} cm; |Fext| max {st['fext_max']:.1f} N")
        if ref is not None:
            ref_p2p = ref["p2p_xyz"]
            print(f"      no-force reference (40/1, last {ref['n'] / 100.0:.1f} s of '{name_of.get(ref_pid, '?')}', "
                  f"move to target excluded): p2p xyz {ref_p2p * 100} cm, max {ref_p2p.max() * 100:.1f} cm")
            print(f"      force-attributable p2p (handshake - reference): xyz {(p2p - ref_p2p) * 100} cm, "
                  f"max axis {(p2p - ref_p2p).max() * 100:.1f} cm")
            out.update(p2p_attr_max_cm=(p2p - ref_p2p).max() * 100)
        else:
            print("      no-force reference (40/1): no samples")
        print(f"      FL motor torque during '{name}' (what the hand works against): peak |tau| hip/thigh/calf "
              f"{st['tau_fl_peak']} Nm (PD part alone {st['tau_pd_fl_peak']} Nm; feedforward |tau_ff| mean "
              f"{st['tau_ff_fl_mean']} Nm)")
        out.update(tau_peak_Nm=float(st["tau_fl_peak"].max()), tau_pd_peak_Nm=float(st["tau_pd_fl_peak"].max()))
        dr = m.drift(rows)
        if dr is not None:
            print(f"      base drift over '{name}': fwd {dr[0]:+.3f} m, left {dr[1]:+.3f} m; "
                  f"roll/pitch max {st['roll_max']:.1f}/{st['pitch_max']:.1f} deg")
            out.update(hand_drift_m=float(np.hypot(*dr)), hand_roll_max=st["roll_max"], hand_pitch_max=st["pitch_max"])
    # the static push probe: steady displacement = mean FL over the last
    # TAIL s of the probe minus the same over the compliant no-force phase
    # right before it (same gains, so the sag of the low gains cancels)
    ppid, prpid = res["probe_phase"], res["probe_ref_phase"]
    if ppid is not None:
        st = m.stats(m._rows(ppid))
        tail = m.stats(m.phase_tail(ppid, TAIL))
        pref = m.stats(m.phase_tail(prpid, TAIL)) if prpid is not None else None
        F = abs(res["probe_force"])
        if st is None or tail is None or pref is None:
            print("    static push probe: no samples")
        else:
            d = tail["fl_mean"] - pref["fl_mean"]
            dn = float(np.linalg.norm(d))
            per_n = dn * 100.0 / F
            print(f"    static push probe {F:g} N toward the body for {st['n'] / 100.0:.1f} s ({res['comp_txt']}): "
                  f"steady FL displacement xyz {d * 100} cm, |d| {dn * 100:.1f} cm -> {per_n:.2f} cm/N "
                  f"(stiffness {F / max(dn * 100, 1e-6):.2f} N/cm) [last {TAIL:g} s of the probe minus last {TAIL:g} s "
                  f"of '{name_of.get(prpid, '?')}']; |Fext| max {st['fext_max']:.1f} N")
            print(f"      FL motor torque during the probe: peak |tau| hip/thigh/calf {st['tau_fl_peak']} Nm "
                  f"(PD part {st['tau_pd_fl_peak']} Nm; |tau_ff| mean {st['tau_ff_fl_mean']} Nm); "
                  f"base drift fwd/left {m.drift(m._rows(ppid))} m, roll/pitch max "
                  f"{st['roll_max']:.1f}/{st['pitch_max']:.1f} deg")
            out.update(probe_disp_cm=dn * 100, probe_cm_per_N=per_n, probe_disp_xyz_cm=d * 100,
                       probe_tau_peak_Nm=float(st["tau_fl_peak"].max()))
    # FL sag: compliant phases without any external force - does the raised
    # leg drift or drop when only the low gains (soft: only gravity comp +
    # the low PD) hold it? End position vs the 40/1 reference, max z drop.
    for name, pid in res["phases"]:
        rows = m._rows(pid)
        st = m.stats(rows)
        if st is None or not st["compliant"] or st["forced"] or ref is None:
            continue
        tail = m.stats(m.phase_tail(pid, TAIL))
        shift = tail["fl_mean"] - ref["fl_mean"]
        drop = ref["fl_mean"][2] - st["fl_z_min"]
        print(f"    FL sag without a hand (compliant {res['comp_txt']}, '{name}', {st['n'] / 100.0:.1f} s): "
              f"end shift vs the 40/1 reference xyz {shift * 100} cm (|d| {np.linalg.norm(shift) * 100:.1f} cm), "
              f"max z drop {drop * 100:.1f} cm; FL |tau| peak {st['tau_fl_peak']} Nm; roll/pitch max "
              f"{st['roll_max']:.1f}/{st['pitch_max']:.1f} deg")
        out.setdefault("sag_phases", []).append(dict(
            name=name, shift_cm=shift * 100, shift_norm_cm=float(np.linalg.norm(shift)) * 100, drop_cm=drop * 100))
    for pid in res["walk_phases"]:
        rows = m._rows(pid)
        st = m.stats(rows)
        name = name_of.get(pid, f"phase {pid}")
        if st is None:
            print(f"    mean forward speed during '{name}': no samples")
            continue
        dur = st["n"] / 100.0
        dr = m.drift(rows) or (0.0, 0.0)
        print(f"    mean forward speed during '{name}': {st['vx_mean']:+.3f} m/s over {dur:.1f} s "
              f"(roll/pitch max {st['roll_max']:.1f}/{st['pitch_max']:.1f} deg); base moved fwd {dr[0]:+.3f} m, "
              f"left {dr[1]:+.3f} m -> lateral drift {dr[1] * 100 / max(dur, 1e-6):+.2f} cm/s, yaw change "
              f"{st['yaw_change']:+.1f} deg; FL foot height mean {(st['fl_mean'][2] - foot_r) * 100:.1f} cm, "
              f"min {(st['fl_z_min'] - foot_r) * 100:.1f} cm")
        out.setdefault("walks", []).append(dict(
            name=name, vx_mean=st["vx_mean"], fwd_m=dr[0], left_m=dr[1], lateral_cm_per_s=dr[1] * 100 / max(dur, 1e-6),
            yaw_deg=st["yaw_change"], fl_height_cm=(st["fl_mean"][2] - foot_r) * 100,
            fl_height_min_cm=(st["fl_z_min"] - foot_r) * 100, roll_max=st["roll_max"], pitch_max=st["pitch_max"]))
    if policy is not None:
        print(f"    base max |roll| {policy['roll_max']:.1f} deg, max |pitch| {policy['pitch_max']:.1f} deg, "
              f"min base height {policy['z_min']:.3f} m (POLICY period)")
        out.update(roll_max=policy["roll_max"], pitch_max=policy["pitch_max"], z_min=policy["z_min"])
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["check-fk", "quad", "quad_demo", "wam_chunks"])
    # wam_chunks: the WAM bridge's chunked, stop-and-wait execution through
    # the real controller code, with a mock policy and simulated inference latency
    ap.add_argument("--chunks", type=int, default=10, help="wam_chunks: chunks to execute (16 commands each)")
    ap.add_argument("--infer-delay", type=float, default=1.0,
                    help="wam_chunks: simulated policy latency per RPC [s] (predict and feedback each); "
                         "the robot holds zero velocity meanwhile, sim time keeps running")
    ap.add_argument("--sample-wait", type=float, default=0.3, help="wam_chunks: bounded wait for sample frames [s]")
    # quad_demo: --ee/--vx/--walk-time/--gap/--settle/--hold default to the demo
    # sequencer's values (scripts/demo_sequence.py); the force/gain parameters
    # are this harness's own
    ap.add_argument("--demo", choices=DEMOS, default="handshake_stand")
    ap.add_argument("--ee", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="FL ee target, projected com frame [m] (default per demo: stand_fl_up "
                         "0.1934 0.142 0.20 = FL folded up under the hip, the real robot's three-leg stand; "
                         "plate_walk 0.48 0.142 0.24 = FL extended forward like the RAMBO paper's plate figure; "
                         "handshake_stand 0.32 0.142 0.20)")
    ap.add_argument("--vx", type=float, default=0.15, help="walk speed [m/s]")
    ap.add_argument("--walk-time", type=float, default=10.0,
                    help="plate_walk: seconds of walking (per leg with --return); on the robot the walk "
                         "is open-ended until the operator stops it")
    ap.add_argument("--return", dest="walk_return", action="store_true",
                    help="plate_walk: also walk back (gap, -vx for --walk-time, zeros); default forward only")
    ap.add_argument("--gap", type=float, default=3.0, help="pause between demo steps [s]")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds in POLICY before the sequencer's first gap; the ee target is in force "
                         "from entry and FL lifts to it when freed at 1.0 s (so >= 1.0)")
    ap.add_argument("--compliant-kp", type=float, default=15.0, help="node ee_compliant_kp (FL joints)")
    ap.add_argument("--compliant-kd", type=float, default=0.8, help="node ee_compliant_kd (FL joints)")
    ap.add_argument("--no-compliance", action="store_true", help="keep kp 40 / kd 1 on FL (baseline)")
    ap.add_argument("--soft", action=argparse.BooleanOptionalAction, default=True,
                    help="node ee_compliant_soft: while compliant, no policy joint residual and no QP torque "
                         "on FL (gravity compensation only) -> controller.ee_soft; --no-soft = gains only")
    ap.add_argument("--plate-mass", type=float, default=0.5, help="plate_walk: mass on the FL paw [kg]")
    # box_push. Defaults describe the article the operator is asked to prepare:
    # a ~4 kg carton with a smooth sheet taped underneath. On the bare
    # high-friction mat (mu ~0.7) it needs ~27 N to slide and ~35 N to break
    # away (the 30 N figure is only the policy's feedforward command clip, not a
    # bound on position-stiffness contact force), and static tipping would need
    # a contact above 0.21 m - the 0.16 m paw is below that. At mu 0.30 it needs
    # ~12 N. MuJoCo agrees with the arithmetic: 11 N does not move it, 12 N does.
    ap.add_argument("--box-mass", type=float, default=4.0, help="box_push: box mass [kg]")
    ap.add_argument("--box-size", type=float, nargs=3, default=(0.30, 0.35, 0.35),
                    metavar=("DEPTH", "WIDTH", "HEIGHT"), help="box_push: box size [m]")
    ap.add_argument("--box-friction", type=float, default=0.30,
                    help="box_push: friction of the box's BOTTOM against the floor (the taped-on "
                         "sheet). This is what decides the push force: mu*m*g (default 0.30)")
    ap.add_argument("--box-shell-friction", type=float, default=0.8,
                    help="box_push: friction of the box's sides. The paw contact actually uses the "
                         "FOOT's parameters (priority 1), so this only matters for other contacts")
    ap.add_argument("--box-distance", type=float, default=0.62,
                    help="box_push: box centre ahead of the base at the start [m] (default 0.62 = "
                         "the paw at x 0.44 just clear of a 0.30 m deep box's near face)")
    ap.add_argument("--box-y", type=float, default=0.142,
                    help="box_push: box centre offset to the robot's left [m]. Defaults to the FL "
                         "hip line so the paw lands on the box's CENTRE - the paw cannot reach the "
                         "robot's centreline at this pose, so the BOX is offset instead. Pushing "
                         "off-centre yaws the box out from under the paw (measured: 68 deg)")
    ap.add_argument("--box-dof", choices=("free", "planar"), default="free",
                    help="box_push: 'free' (default) is an unconstrained box - honest, and it spins "
                         "out from under the paw in a few seconds, because a single point contact "
                         "cannot control a sliding box's heading (the push and the box's bottom "
                         "friction form a couple as soon as the contact is ~1 cm off the box's "
                         "centre line, and a sliding box has almost no yaw resistance left). "
                         "'planar' models a box the OPERATOR has constrained against rotation "
                         "(guides / a rail / a tray): it slides in x and y but cannot yaw or tip")
    ap.add_argument("--push-vy", type=float, default=0.0,
                    help="box_push: constant lateral velocity command held during the push [m/s]. "
                         "The push reaction makes the robot crab sideways (measured: -0.19 m over "
                         "12 s of pushing vs -0.03 m walking untouched), which drags the paw across "
                         "the box face and spins the box out from under it")
    ap.add_argument("--heading-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="node heading_hold: PI on the IMU yaw commands the yaw rate while walking "
                         "straight (--no-heading-hold = the raw operator command, shows the drift)")
    ap.add_argument("--heading-hold-kp", type=float, default=1.0, help="node heading_hold_kp")
    ap.add_argument("--heading-hold-ki", type=float, default=0.3, help="node heading_hold_ki")
    ap.add_argument("--heading-hold-max-wz", type=float, default=0.3, help="node heading_hold_max_wz [rad/s]")
    ap.add_argument("--heading-hold-deadband-deg", type=float, default=1.0,
                    help="node heading_hold_deadband_deg")
    ap.add_argument("--hand-steady", type=float, default=5.0,
                    help="handshake: steady push on the paw toward the body [N] (< 0 pulls)")
    ap.add_argument("--hand-amp", type=float, default=8.0, help="handshake: vertical sinusoid amplitude [N]")
    ap.add_argument("--hand-freq", type=float, default=2.0, help="handshake: shake frequency [Hz]")
    ap.add_argument("--hand-lateral", type=float, default=10.0,
                    help="handshake: one 0.3 s lateral pulse mid-window [N]")
    ap.add_argument("--hold", type=float, default=20.0,
                    help="stand_fl_up: seconds the folded FL is held; handshake_stand: compliant window = "
                         "seconds of handshake force (sequencer default 20)")
    ap.add_argument("--probe-force", type=float, default=5.0,
                    help="handshake_stand: after the let-go gap, a constant push on the paw toward the body "
                         "[N] for --probe-time while still compliant -> steady displacement in cm/N (0 = none)")
    ap.add_argument("--probe-time", type=float, default=3.0, help="handshake_stand: static probe duration [s]")
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
    if args.scenario == "quad_demo":
        # the sequencer's own checks (demo_sequence._validate)
        for cond, msg in ((args.hold > 0.0, "--hold must be > 0"),
                          (args.walk_time > 0.0, "--walk-time must be > 0"),
                          (args.probe_time > 0.0, "--probe-time must be > 0"),
                          (args.gap >= 0.0, "--gap must be >= 0"),
                          (args.settle >= 1.0, "--settle must be >= 1.0 s: the node keeps all four feet "
                                               "planted for the first second of POLICY")):
            if not cond:
                ap.error(msg)

    box = None
    if (getattr(args, "demo", None) == "box_push" and args.scenario == "quad_demo") or args.scenario == "wam_chunks":
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

    if args.scenario == "quad_demo":
        h = Harness(Sim(model), task=QUADRUPED, policy_path=args.checkpoint)
        h.true_vel = args.true_vel
        if args.record:
            w, hh_ = (int(v) for v in args.size.lower().split("x"))
            h.recorder = Recorder(model, args.record, fps=args.fps, width=w, height=hh_,
                                  azimuth=args.cam_azimuth, elevation=args.cam_elevation,
                                  distance=args.cam_distance, font=args.font)
        try:
            res = run_quad_demo(h, args)
        finally:
            if h.recorder is not None:
                h.recorder.close()
        print_quad_demo_result(h, res)
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
