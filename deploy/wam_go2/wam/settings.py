"""Settings shared by the WAM-Policy bridge, the mock policy service and the server launcher.

The released WAM-Policy models (``python -m wam_policy.inference.serve`` of this repository)
take two 640x360 RGB views (``dual-rgb-360p-v1``) and the task instruction, and no body state.
"""
INSTRUCTION = "Push the box inside the outlined area."   # default task (push_box)
RGB_PROFILE = "dual-rgb-360p-v1"                          # the released teacher's rgb_profile (= observation.RGB_360)
ROOT_STATE = False                                        # WAM-Policy takes no proprioception

# --policy of wam_policy.inference.serve (WAM_POLICY); WAM_MOCK=1 runs the scripted stand-in instead
POLICIES = ("opsd_student", "mopd_student", "teacher", "specialist")
DEFAULT_POLICY = "opsd_student"
