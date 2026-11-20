# deploy/: WAM-Policy on a real Unitree Go2

Runs the released WAM-Policy models on a real Unitree Go2 with the
[RAMBO](https://github.com/catachiii/rambo) whole-body controller. Needs a laptop with Ubuntu 24.04,
ROS 2 Jazzy and an NVIDIA GPU, connected to the robot by Ethernet, and a D435i on an RDK X5 on the
robot ([docs/cameras.md](docs/cameras.md)).

## Installation

From the repository root, with the WAM-Policy environment installed ([top-level README](../README.md#installation)):

```bash
git clone -b qlm-bench https://github.com/AIGeeksGroup/WAM-Policy.git ../QLM-Bench
pip install -e "../QLM-Bench[live]"
cp configs/site.example.env configs/site.env       # edit the paths
scripts/download.sh weights
scripts/download.sh base

sudo apt install python3-opencv
python3 -m venv --system-site-packages ~/venvs/wam-go2
~/venvs/wam-go2/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
~/venvs/wam-go2/bin/pip install -r deploy/requirements.txt
~/venvs/wam-go2/bin/pip install -e "../QLM-Bench[live]"
deploy/scripts/install_unitree_msgs.sh
deploy/scripts/fetch_controller.sh
```

## Configuration

```bash
cp deploy/config/deploy.example.env deploy/config/deploy.env
```

Set `DEMO_NIC` (the wired interface), `DEMO_ROBOT_IP`, `DEMO_LAPTOP_IP`, `X5_IP` and
`WAM_POLICY_PYTHON` (the Python of the WAM-Policy environment) in it. Then set up the camera host
once:

```bash
deploy/scripts/x5/laptop_chrony_install.sh
deploy/scripts/x5/x5_install.sh
deploy/scripts/x5/x5_chrony_setup.sh
```

## Running

```bash
deploy/tasks/push_box.sh                         # one script per task in deploy/tasks/
WAM_POLICY=teacher deploy/tasks/push_box.sh      # opsd_student (default), mopd_student, teacher, specialist
WAM_MOCK=1 deploy/tasks/push_box.sh              # without a model: the robot steps in place
```

Press Enter to stand the robot up. **Ctrl+C stops the robot.** Results go to `deploy/result/<task>/`.

## License

[CC BY-NC 4.0](LICENSE), attributions in [NOTICE](NOTICE); the rest of the repository is Apache-2.0.
The controller is this project's own implementation of the RAMBO control pipeline, built on
open-source libraries.
