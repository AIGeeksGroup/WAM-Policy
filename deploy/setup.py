import os
from glob import glob

from setuptools import find_packages, setup

package_name = "wam_go2_deploy"


def files(pattern):
    """glob() restricted to regular files (setuptools cannot copy directories)."""
    return [p for p in sorted(glob(pattern)) if os.path.isfile(p)]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.xml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "policies"), ["policies/README.md"]),
        (os.path.join("share", package_name, "policies", "quadruped"),
         files("policies/quadruped/*")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "osqp"],
    zip_safe=True,
    maintainer="WAM-Policy authors",
    maintainer_email="wam-policy@example.com",
    description="WAM-Policy on a real Unitree Go2: RAMBO whole-body controller and policy bridge",
    license="CC-BY-NC-4.0",
    entry_points={
        "console_scripts": [
            "controller = wam_go2.nodes.controller_node:main",
            "lowstate_echo = wam_go2.nodes.lowstate_echo:main",
            "release_sport_mode = wam_go2.nodes.release_sport_mode:main",
            # the WAM-Policy bridge (qlm-live-v1 client), the ego camera stream and a scripted
            # stand-in for the policy server
            "wam_bridge = wam_go2.nodes.wam_bridge_node:main",
            "ego_stream = wam_go2.nodes.ego_stream_node:main",
            "wam_mock_service = wam_go2.wam.mock_service:main",
        ],
    },
)
