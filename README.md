# ROSCon KR IL Workshop: LeRobot with Real Robots

이 레포지토리는 ROSCon 모방학습 워크샵을 위한 자료입니다. Hugging Face의 LeRobot 프레임워크를 사용하여 데이터 수집, 데이터셋 관리, 모델 학습(ACT, SmolVLA), 그리고 실물 로봇 제어를 위한 추론 과정을 다룹니다.

---

# Prerequisites & Installation

### System Requirements

LeRobot 구동을 위한 권장 최소 사양입니다.

* **OS:** Linux (Ubuntu 20.04 or 22.04)
* **Python:** 3.10
* **Hardware:** NVIDIA GPU (CUDA 12+ recommended for training)

### Installation

LeRobot 및 하드웨어 제어 관련 의존성을 설치합니다.

```bash
# 가상환경 생성 (Python 3.10 필수)
conda create -y -n lerobot python=3.10
conda activate lerobot

# 레포지토리 클론 및 설치
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 기본 패키지 설치
pip install -e .

# 로봇 하드웨어 제어용 추가 패키지 (Dynamixel, Feetech, Cameras)
pip install -e ".[dynamixel, feetech, intelrealsense]"

```

---

# Workshop Workflow

본 워크샵은 데이터 수집부터 비동기 추론까지 총 4단계의 파이프라인으로 구성됩니다. 각 단계는 준비된 실물 로봇 스테이션에서 진행됩니다.

## 1. Data Collection (데이터 수집)

텔레오퍼레이션(Teleoperation)을 통해 로봇의 상태(State)와 이미지(Image) 데이터를 수집합니다.

**Robot Setting:** [OpenManipulator-X](https://ai.robotis.com/omx/lerobot_imitation_learning_omx), [SO-101](https://huggingface.co/docs/lerobot/so101)
각 링크를 참조하여 로봇 세팅을 완료해주세요.


**Teleoperation Command Example:**
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm
```

**Record Command Example:**

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube"
```

## 2. Demonstration Review & Train (데모 리뷰 및 학습)

수집된 데이터의 품질을 검증하고 정책(Policy) 모델을 학습합니다.

### 데이터 시각화 및 검수

- 관찰과 액션 값의 동기화가 잘 되었는지 점검
- 수집 시 액션이 부드러운지 점검
- 조명 혹은 블러가 되었는지 점검

**Replay Dataset**
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0
```
**[Web-based Dataset Check](https://huggingface.co/spaces/lerobot/visualize_dataset)**

### 모델 학습 (ACT / SmolVLA)

데이터셋을 사용하여 모델을 학습시킵니다.

**Option A: ACT (Action Chunking Transformer)**

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

**Option B: SmolVLA (Vision-Language Action)**

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

## 3. Inference (추론)

학습된 모델을 로봇에 배포하여 추론을 수행합니다. 기본 `eval.py` 스크립트는 동기식(Synchronous)으로 동작합니다.

**특징:**

* `Observation` → `Inference` → `Action` 과정이 순차적으로 실행됩니다.
* 추론 연산 시간 동안 로봇의 제어 루프가 대기(Block)하게 되어, 모델이 무거울수록 제어 주기가 불안정해질 수 있습니다.

**Command:**

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/eval_act_your_dataset \
  --dataset.num_episodes=10 \
  --dataset.single_task="Your task description" \
  --policy.path=${HF_USER}/act_policy
```

## 4. Asynchronous Inference (비동기 추론)

LeRobot 프레임워크가 제공하는 비동기 추론 기능을 활용합니다. 워크샵에서는 이 기능을 효율적으로 호출하고 관찰할 수 있도록 별도로 작성된 코드를 사용합니다.

**개요:**
동기식 추론의 한계를 극복하기 위해 제어 루프와 추론 루프를 분리합니다.

* **Control Loop:** 하드웨어의 입출력(I/O)을 담당하며 고정된 고주파수(예: 60Hz)를 유지합니다.
* **Inference Loop:** 백그라운드에서 모델 연산을 수행하고 최신 액션을 갱신합니다.

**장점:**

* 추론 시간(Latency)이 제어 주기에 영향을 주지 않습니다.
* 로봇이 연산 대기 시간 없이 부드럽게 움직입니다(Jitter 감소).

**(Policy)Server Start Command:**

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=25 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
```
**(RoBot)Client Start Command:**

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower \
    --robot.id=follower \
    --robot.cameras='{ \
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25}, \
        wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25}, \
    }' \
    --task=IMsubin/pick_tomato_place_pot \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=IMsubin/pick_tomato_place_pot \
    --policy_device=cuda \
    --actions_per_chunk=100 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

## Reference

* [LeRobot Installation](https://huggingface.co/docs/lerobot/installation)
* [LeRobot Dataset Guide](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
* [ACT Policy Info](https://huggingface.co/docs/lerobot/act)
* [LeRobot Asynchronous Design](https://huggingface.co/docs/lerobot/async)

### 데이터 취득 가이드

* **Safety & Hardware:** (작성 예정: E-Stop 위치, 카메라 고정, 조명 등 하드웨어 체크리스트)
* **High Quality Data:** (작성 예정: 다양성 확보, 멈춤 없는 동작, 복구 동작 포함 등 노하우 기술)
