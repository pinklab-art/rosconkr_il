# ROSCon KR IL Workshop: Imitation Learning with Real Robots

2026 ROSCon KR 모방학습 워크샵을 위한 자료입니다. 
Hugging Face의 LeRobot 프레임워크 기반으로 데이터 수집, 데이터셋 관리, 모델 학습(ACT, SmolVLA), 그리고 실물 로봇 제어를 통한 추론 과정을 배웁니다.

---

# Prerequisites & Installation

## System Requirements

* **Python: 3.10**
* **Ubuntu (recommended)**
* **NVIDIA GPU (CUDA 12+ 학습 시 권장)**
* **GIT**
* **2개 이상의 카메라**
* **USB 허브**

### Installation

LeRobot 및 하드웨어 제어 관련 의존성을 설치합니다.

```bash
# 가상환경 생성 
mkdir -p ~/venv && cd ~/venv
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y

python3.10 -m venv lerobot

# 가상환경 활성화
source ~/venv/lerobot/bin/activate

# ffmpeg 설치
sudo apt update
sudo apt install ffmpeg

# lerobot git clone
git clone https://github.com/huggingface/lerobot.git
cd ~/lerobot

# 패키지 설치
pip install -e .

# 추가 기능
pip install -e ".[all]"           # 모든 기능 설치 
pip install -e ".[aloha,pusht]"   # (Aloha 및 Pusht) 
pip install -e ".[feetech]"       # Feetech motor(so-101 사용시 설치)
pip install -e ".[dynamixel]"     # Dynamixel motor
```

---

# Workshop Workflow

본 워크샵은 데이터 수집부터 비동기 추론까지 총 4단계의 파이프라인으로 구성됩니다. 각 단계는 준비된 실물 로봇 스테이션에서 진행됩니다.

## 1. Data Collection (데이터 수집)

텔레오퍼레이션(Teleoperation)을 통해 로봇의 상태(State)와 이미지(Image) 데이터를 수집합니다.

### Robot Setting
* [OpenManipulator-X](https://ai.robotis.com/omx/lerobot_imitation_learning_omx)
* [SO-101](https://huggingface.co/docs/lerobot/so101)

각 링크를 참조하여 로봇 세팅을 완료해주세요.

<details>
<summary>SO-101 Udev Setting</summary>

**USB 포트 고정**

* 포트 접근 권한 설정
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```
* dialout 추가 (Leader: ttyACM0, Follower: ttyACM1)
```bash
sudo usermod -a -G dialout $USER
```
* 고유 serial 넘버확인
```bash
udevadm info -a -n /dev/ttyACM0 | grep '{serial}' -m 1
udevadm info -a -n /dev/ttyACM1 | grep '{serial}' -m 1
```
* udev 생성
```
sudo nano /etc/udev/rules.d/99-serial.rules

# 아래 내용 추가 후 저장
SUBSYSTEM=="tty", ATTRS{serial}=="5AB0183022", SYMLINK+="so101_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="5AB0182087", SYMLINK+="so101_follower"
```
* 적용
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
* 적용 확인
```bash
ls -l /dev/so101_*
```
**카메라 포트 고정**

* video 장치목록 확인
    * 위치에 해당하는 비디오 넘버를 기억 (ex. top: /dev/video0, wrist: /dev/video2)
```bash
sudo apt update
sudo apt install v4l-utils
```
```bash
v4l2-ctl --list-devices
```
* 포트 위치 확인
```bash
udevadm info -a -n /dev/video0 | grep 'KERNELS=="[0-9]' | head -n 1
udevadm info -a -n /dev/video2 | grep 'KERNELS=="[0-9]' | head -n 1
```
* udev 생성
```
sudo nano /etc/udev/rules.d/99-camera.rules

# 아래 내용 추가 후 저장
SUBSYSTEM=="video4linux", KERNELS=="1-1:1.0", ATTR{index}=="0", SYMLINK+="cam_top"
SUBSYSTEM=="video4linux", KERNELS=="1-5:1.0", ATTR{index}=="0", SYMLINK+="cam_wrist"
```
* 적용
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
* 적용 확인
```bash
ls -l /dev/cam_*
```
</details>

<details>
<summary>허깅페이스 계정 등록</summary>
* HuggingFace CLI 토큰 로그인
    * Write 권한으로 토큰을 미리 생성
    
```bash
cd ~/lerobot
hf auth login --add-to-git-credential --token <YOUR_TOKEN_HERE>
```

* 환경 변수 설정

```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

* bashrc 설정

```bash
export HF_USER=$(python - <<'PY'
from huggingface_hub import whoami
print(whoami().get("name", ""))
PY
)
```
```bash
source ~/.bashrc
```

</details>

### Teleoperation Example

**Command**
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --robot.cameras='{
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25},
        wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
    }' \
```

### Record Example

**Command**
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttACM0 \
    --robot.id=follower \
    --robot.cameras='{
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25},
        wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \ # hf cli login required
    --dataset.single_task="Grab the black cube" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5
```
* local 저장 위치
```bash
~/.cache/huggingface/datasets/${HF_USER}/
```

**데이터 추가 수집 command**
```bash
lerobot-record \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.cameras='{
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 25},
        wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
    }' \
    --dataset.repo_id=${HF_USER}/record-test \ # hf cli login required
    --dataset.single_task="Grab the black cube" \
    --dataset.num_episodes=5 \ # if you want to record 5 more episode
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --display_data=true \
    --resume=true # true 시, 추가 수집
```


**데이터 제거 (optional)**

```bash
# Delete episodes 0, 2, and 5 (modifies original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

# Delete episodes and save to a new dataset (preserves original dataset)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"
```

**데이터셋 merge (optional)**
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"
```

**Features 제거 (optional)**
```bash
# Remove a camera feature
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.top']"
```

<details>
<summary>데이터 수집 팁</summary>
   
* **기본 팁**
   * **원격 조작으로 충분히 하려는 TASK에 익숙해진 다음 취득하는걸 추천**
   * **위치당 10개, 최소 50개의 에피소드 취득을 권장**
   * **카메라는 고정된 상태 유지, 녹화하는 동안 일관된 집기 동작을 유지**
   * **물체가 다 보이게 카메라 배치**
   * **동작 혹은 위치변경 등 변화가 너무 많을 시, 결과에 부정적인 영향을 줌** (적은 데이터셋 대비 변화가 많을 시)
* **이미지 품질 팁**
   * 가급적 **두 개의 카메라 뷰를 사용**
   * **흔들림 없이 안정적인 영상 촬영**
   * **중립적이고 안정적인 조명을 유지** (지나치게 노란색이나 파란색 톤은 피하기)
   * **일관된 노출과 선명한 초점을 유지**
   * **리더 팔은 프레임에 나타나지 않아야 함**
   * **움직이는 물체는 팔과 조작되는 물건만 있어야 함** (사람 팔다리/몸은 최대한 배제)
   * **정적이고 방해가 되지 않는 배경을 사용**하거나 통제된 변형을 적용
   * **고해상도로 녹화** (최소 480x640 / 720p)
* **작업 내용 팁**
   * **task필드를 사용하여 로봇의 목적을 명확하게 설명** (ex. Pick the yellow lego block and put it in the box)
   * **작업 설명은 간결하게 유지** (25~50자)
   * **task1, demo2, 등과 같이 모호하거나 일반적인 이름은 피하기**
   
</details>

## 2. Demonstration Review & Train (데모 리뷰 및 학습)

수집된 데이터의 품질을 검증하고 정책(Policy) 모델을 학습합니다.

### 데이터 시각화 및 검수

* 관찰과 액션 값의 동기화가 잘 되었는지 점검
* 수집 시 액션이 부드러운지 점검
* 조명 혹은 블러가 되었는지 점검

**Replay Dataset**
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --dataset.repo_id=${HF_USER}/record-test \ # hf cli login required
    --dataset.episode=0
```

**[Web-based Dataset Visualization](https://huggingface.co/spaces/lerobot/visualize_dataset)**
* dataset repo id 입력 후, 수집 데이터 관찰 가능

### 모델 학습 (ACT / SmolVLA)

데이터셋을 사용하여 모델을 학습시킵니다.

**Option A: ACT (Action Chunking Transformer)**

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \ # optional
  --policy.repo_id=${HF_USER}/act_policy # hf cli login required
```

**Option B: SmolVLA (Vision-Language Action)**

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/mydataset \ # hf cli login required
  --policy.repo_id=${HF_USER}/mydataset_smolvla \
  --policy.path=lerobot/smolvla_base \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true # optional
```

## 3. Inference (추론)

학습된 모델을 로봇에 배포하여 추론을 수행합니다. 기본 `eval.py` 스크립트는 동기식(Synchronous)으로 동작합니다.
학습한 데이터셋과 최대한 동일한 환경에서 진행

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

LeRobot 프레임워크가 제공하는 비동기 추론 기능을 활용합니다.
동기식 추론의 한계를 극복하기 위해 제어 루프와 추론 루프를 분리합니다.

* **Policy Server** 가속 하드웨어에서 호스팅되며 실제 로봇에 할당된 것보다 더 많은 컴퓨팅 리소스를 사용하여 추론을 실행
* **Robot Client:** 수신된 액션을 큐에 추가하고 다음 청크가 계산되는 동안 해당 액션을 실행

**장점:**
* 추론 시간(Latency)이 제어 주기에 영향을 주지 않아, 작업 시간 및 상호 작용 시간을 약 2배 감소
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
    --robot.port=/dev/ttyACM1 \
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
