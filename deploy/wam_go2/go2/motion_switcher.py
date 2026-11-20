"""Release the Go2 high-level motion service before low-level control.

Implements the unitree_api request/response pattern over ROS2 topics
(reference: unitree_sdk2 motion_switcher_client.hpp / motion_switcher_api.hpp
and unitree_ros2 example go2_robot_state_client.cpp):

- motion_switcher service: /api/motion_switcher/request
    CHECK_MODE = 1001 -> response data {"form": ..., "name": ...}
    RELEASE_MODE = 1003 (no parameter)
- robot_state service fallback: /api/robot_state/request
    SERVICE_SWITCH = 1001, parameter {"name": "sport_mode", "switch": 0}

The official flow (go2_stand_example.cpp:128-142) loops CheckMode ->
ReleaseMode until the reported mode name is empty.
"""

import json
import time

from unitree_api.msg import Request, Response

MOTION_SWITCHER_REQUEST_TOPIC = "/api/motion_switcher/request"
MOTION_SWITCHER_RESPONSE_TOPIC = "/api/motion_switcher/response"
ROBOT_STATE_REQUEST_TOPIC = "/api/robot_state/request"

API_CHECK_MODE = 1001
API_SELECT_MODE = 1002
API_RELEASE_MODE = 1003
API_SERVICE_SWITCH = 1001


class MotionSwitcherClient:
    def __init__(self, node):
        self._node = node
        self._pub = node.create_publisher(Request, MOTION_SWITCHER_REQUEST_TOPIC, 10)
        self._rs_pub = node.create_publisher(Request, ROBOT_STATE_REQUEST_TOPIC, 10)
        self._sub = node.create_subscription(
            Response, MOTION_SWITCHER_RESPONSE_TOPIC, self._on_response, 10)
        self._responses = {}
        self._pending = {}
        self._last_request_id = 0
        self._release_req = None
        self.release_state = "idle"
        self.release_error = ""

    def _on_response(self, msg):
        identity = msg.header.identity
        if self._pending.get(identity.id) == identity.api_id:
            self._responses[identity.id] = msg

    def _send(self, pub, api_id, parameter="", *, expect_response=True):
        req = Request()
        self._last_request_id = max(time.monotonic_ns(), self._last_request_id + 1)
        req.header.identity.id = self._last_request_id
        req.header.identity.api_id = api_id
        req.parameter = parameter
        if expect_response:
            self._pending[req.header.identity.id] = api_id
        pub.publish(req)
        return req.header.identity.id

    def _wait(self, req_id, timeout, spin):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            spin()
            if req_id in self._responses and time.monotonic() < deadline:
                self._pending.pop(req_id, None)
                return self._responses.pop(req_id)
        self._pending.pop(req_id, None)
        self._responses.pop(req_id, None)
        return None

    @staticmethod
    def _mode_name(resp):
        """Unknown/error is None; only an explicit valid empty name means free."""
        if resp is None or resp.header.status.code != 0:
            return None
        try:
            data = json.loads(resp.data)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            return None
        return data["name"]

    # ------------------------------------------------------------------
    def check_mode(self, spin, timeout=2.0):
        """Return the active name ('' if confirmed released), or None if unknown."""
        req_id = self._send(self._pub, API_CHECK_MODE)
        return self._mode_name(self._wait(req_id, timeout, spin))

    def release_mode(self, spin, timeout=2.0):
        req_id = self._send(self._pub, API_RELEASE_MODE)
        resp = self._wait(req_id, timeout, spin)
        return resp is not None and resp.header.status.code == 0

    def select_mode(self, name, spin, timeout=3.0):
        """SelectMode (MOTION_SWITCHER_API_ID_SELECT_MODE = 1002,
        motion_switcher_api.hpp:26): hand the robot back to a high-level
        motion service, e.g. "normal" (the Go2 sport controller) after
        low-level control. Returns (ok, status_code)."""
        req_id = self._send(self._pub, API_SELECT_MODE, json.dumps({"name": name}))
        resp = self._wait(req_id, timeout, spin)
        if resp is None:
            return False, None
        return resp.header.status.code == 0, resp.header.status.code

    def switch_off_sport_mode(self):
        """Fire-and-forget fallback via the robot_state service."""
        parameter = json.dumps({"name": "sport_mode", "switch": 0})
        self._send(self._rs_pub, API_SERVICE_SWITCH, parameter, expect_response=False)

    def release_mode_async(self):
        """Fire-and-forget ReleaseMode request (no response wait)."""
        self._send(self._pub, API_RELEASE_MODE, expect_response=False)

    def start_release(self, now=None, *, allow_release=True, max_tries=5,
                      timeout=2.0, settle_s=2.0):
        """Start a confirmed handoff; no spin, sleep, or response wait.

        With allow_release=False this only queries ownership. A failed handoff
        stays failed until explicitly restarted; callers must deny active motor
        commands while poll_release() is false.
        """
        if max_tries < 0 or timeout <= 0 or settle_s < 0:
            raise ValueError("invalid motion handoff retry/timing configuration")
        now = time.monotonic() if now is None else now
        if self._release_req is not None:
            self._pending.pop(self._release_req, None)
            self._responses.pop(self._release_req, None)
        self._allow_release = bool(allow_release)
        self._release_max_tries = max_tries
        self._release_tries = 0
        self._check_timeouts = 0     # consecutive CheckMode attempts with no usable answer
        self._release_timeout = timeout
        self._release_settle_s = settle_s
        self.release_error = ""
        self._release_request(API_CHECK_MODE, now)

    def _release_request(self, api_id, now):
        self._release_req = self._send(self._pub, api_id)
        self._release_deadline = now + self._release_timeout
        self.release_state = "checking" if api_id == API_CHECK_MODE else "releasing"

    def poll_release(self, now=None):
        """Advance at most one handoff step and return confirmed ownership.

        Suitable for a 100 Hz controller callback. Confirmation always comes
        from CheckMode, never from a ReleaseMode acknowledgement alone.
        """
        now = time.monotonic() if now is None else now
        if self.release_state in ("idle", "failed", "confirmed"):
            return self.release_state == "confirmed"
        if self.release_state == "settling":
            if now >= self._release_deadline:
                self._release_request(API_CHECK_MODE, now)
            return False

        req_id = self._release_req
        resp = self._responses.pop(req_id, None)
        if now >= self._release_deadline:
            resp = None
        if resp is None and now < self._release_deadline:
            return False
        self._pending.pop(req_id, None)
        self._release_req = None
        if self.release_state == "checking":
            name = self._mode_name(resp)
            if name == "":
                self.release_state = "confirmed"
                self.release_error = ""
                return True
            if name is None and resp is not None:
                # The service ANSWERED, and not usably (error status, non-JSON,
                # no string name). That is a protocol/firmware problem, not a
                # lost packet: fail closed when we cannot act on it, otherwise
                # fall through to the bounded official release loop below.
                self.release_error = "CheckMode returned an invalid/error response"
                if not self._allow_release or self._release_tries >= self._release_max_tries:
                    self.release_state = "failed"
                    return False
                self.switch_off_sport_mode()
                self._release_tries += 1
                self._release_request(API_RELEASE_MODE, now)
                return False
            if name is None:
                # SILENCE is not evidence that a motion service is active. On a
                # fresh node the very first request is routinely lost to DDS
                # discovery (in the mock-robot tests the mock answers every
                # CheckMode it receives, yet without a retry 3 of 8 node starts
                # end 'failed' here in check-only mode and refuse STAND for the
                # rest of the run). Ask again; give up after max_tries
                # consecutive silences. With release allowed, the second
                # silence in a row fires the robot_state fallback and a
                # ReleaseMode, i.e. the official example's loop.
                self.release_error = "CheckMode timed out (no response)"
                self._check_timeouts += 1
                if self._allow_release:
                    if self._check_timeouts < 2:
                        self._release_request(API_CHECK_MODE, now)   # discovery grace
                    elif self._release_tries >= self._release_max_tries:
                        self.release_state = "failed"
                    else:
                        self.switch_off_sport_mode()
                        self._release_tries += 1
                        self._release_request(API_RELEASE_MODE, now)
                elif self._check_timeouts >= self._release_max_tries:
                    self.release_state = "failed"
                else:
                    self._release_request(API_CHECK_MODE, now)
                return False
            self._check_timeouts = 0
            self.release_error = f"motion service is still active: {name!r}"
            if not self._allow_release or self._release_tries >= self._release_max_tries:
                self.release_state = "failed"
                return False
            self._release_tries += 1
            self._release_request(API_RELEASE_MODE, now)
        else:
            if resp is None or resp.header.status.code != 0:
                self.release_error = "ReleaseMode timed out or returned an error; awaiting CheckMode"
            self.release_state = "settling"
            self._release_deadline = now + self._release_settle_s
        return False

    def release_until_clear(self, spin, max_tries=5, settle_s=2.0):
        """Blocking CLI wrapper around the same confirmed handoff state machine."""
        self.start_release(max_tries=max_tries, settle_s=settle_s)
        while self.release_state != "failed":
            if self.poll_release():
                return True
            spin()
        return False
