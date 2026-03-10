# rad-rl-project: Demo Collection

Human-teleoperated demonstration collection for Spot in a warehouse environment.
Spot is driven with an Xbox-compatible controller; a trained low-level locomotion
policy translates velocity commands into joint actions while robot state, camera
frames, and actions are logged to HDF5 at 10 Hz.

---

## Contents

| File | Description |
|---|---|
| `collect_demos.py` | Main collection script |
| `spot-in-simple-warehouse.usd` | Warehouse simulation scene |
| `environment.yml` | Conda environment specification |
| `utils.ipynb` | Checkpoint upload to HuggingFace Hub |

---

## Environment Setup

### 1. Create the conda environment

```bash
conda env create -f scripts/rad-rl-project/environment.yml
conda activate isaaclab
```

### 2. Install IsaacLab from source

```bash
cd <IsaacLab-repo-root>
./isaaclab.sh --install
```

### 3. Add your user to the `input` group (required for controller access)

```bash
sudo usermod -aG input $USER
```

Log out and back in, or start a new shell with `newgrp input` to activate the
change without a full re-login.

---

## Xbox Controller Setup (Ubuntu 22.04)

Any **Xbox 360-compatible** wired controller is supported. The driver (`xpad`)
ships with the Linux kernel and requires no additional installation.

### Verify the controller is detected

```bash
ls /dev/input/js0          # joystick node
cat /proc/bus/input/devices | grep -A5 -i xbox
```

### Confirm device permissions

```bash
ls -la /dev/input/event*   # your user needs rw access
groups $USER               # should include 'input'
```

If `/dev/input/eventN` shows `crw-rw----` and you are in the `input` group,
you are ready. The `xpad` module presents the controller as a standard Xbox 360
layout, so no additional mapping or configuration is needed.

### Controller bindings

```
Left stick up / down        Forward / backward  (v_x,  up to ±3 m/s)
Left stick left / right     Strafe              (v_y,  up to ±1.5 m/s)
Right stick left / right    Turn                (ω_z,  up to ±2 rad/s)

A                           Mark episode SUCCESS and reset
B                           Mark episode FAILURE and reset
START (MENU1)               Quit collection and save dataset
```

Velocity commands are clamped to the training ranges before being passed to the
policy:

| Axis | Range |
|---|---|
| `v_x` (forward) | −2.0 to +3.0 m/s |
| `v_y` (lateral) | −1.5 to +1.5 m/s |
| `ω_z` (yaw) | −2.0 to +2.0 rad/s |

---

## Running Demo Collection

### Minimal launch

```bash
./isaaclab.sh -p scripts/rad-rl-project/collect_demos.py \
    --checkpoint logs/rsl_rl/spot_flat/2026-03-09_15-47-17/model_19999.pt
```

The output file defaults to `./datasets/warehouse_demos_<timestamp>.hdf5`.
A session timestamp is always appended so no previous session is overwritten.

### Full options

```bash
./isaaclab.sh -p scripts/rad-rl-project/collect_demos.py \
    --checkpoint   logs/rsl_rl/spot_flat/2026-03-09_15-47-17/model_19999.pt \
    --dataset_file datasets/warehouse_demos.hdf5 \
    --num_episodes 20 \
    --goal_position 8.0 0.0 -5.0 \
    --success_radius 0.5 \
    --min_episode_steps 10 \
    --camera_height 240 \
    --camera_width 320
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pt` policy checkpoint |
| `--dataset_file` | `./datasets/warehouse_demos.hdf5` | HDF5 output path (timestamp appended) |
| `--num_episodes` | `0` (infinite) | Stop after N episodes; 0 = run until START |
| `--goal_position X Y Z` | `8.0 0.0 -5.0` | World-frame position of the green goal sphere |
| `--success_radius` | `0.5` | Auto-success distance to goal (metres) |
| `--min_episode_steps` | `10` | Minimum 10 Hz steps to save an episode |
| `--camera_height` | `240` | Logged RGB frame height (pixels) |
| `--camera_width` | `320` | Logged RGB frame width (pixels) |

---

## Collection Workflow

1. **Isaac Sim loads** — expect 30–60 s on first launch.
2. **The policy is loaded** and the warehouse scene is built with a forward-facing
   camera attached to Spot's body and a green goal sphere placed at `--goal_position`.
3. The terminal prints:
   ```
   [INFO] Collection running. Drive Spot with the Xbox controller.
          A = success  |  B = failure/reset  |  START (MENU1) = quit
   ```
4. **Drive Spot** toward the green sphere. Data is buffered from the moment the
   message appears — no further action is needed to start recording.
5. **End each episode** with A (success) or B (failure). Spot resets immediately
   and the next episode begins.
6. Episodes shorter than `--min_episode_steps` (1 second at 10 Hz) are silently
   discarded to prevent accidental saves.
7. **Auto-success** triggers if Spot reaches within `--success_radius` metres of
   the goal in the XY plane.
8. Press **START (MENU1)** to flush and close the dataset when you are done.

---

## HDF5 Dataset Structure

Each episode is stored as `episode_N` with the following layout:

```
episode_N/
    obs/
        camera_rgb          (T, H, W, 3)    uint8   — forward camera
        base_pose           (T, 7)          float32 — position (3) + quaternion wxyz (4)
        base_lin_vel        (T, 3)          float32 — world frame
        base_ang_vel        (T, 3)          float32 — world frame
        joint_pos           (T, 12)         float32
        joint_vel           (T, 12)         float32
        contact_forces      (T, N_feet, 3)  float32 — net forces, world frame
        goal_relative       (T, 3)          float32 — goal minus robot position
        policy_obs          (T, 48)         float32 — raw locomotion policy input
    action/
        velocity_cmd        (T, 3)          float32 — gamepad command [v_x, v_y, ω_z]
        joint_targets       (T, 12)         float32 — raw policy output
    attrs:
        success             bool
        episode_length      int
        goal_position_world (3,)            float32
        timestamp           str             — ISO-8601 wall-clock time
```

`T` is the number of 10 Hz timesteps in the episode.

---

## Policy Checkpoint

The checkpoint used for collection is hosted on HuggingFace:

```
lorinachey/rad-rl-checkpoints
└── checkpoints/spot_flat/model_19999.pt
```

To upload a new checkpoint, use `utils.ipynb`.

**Policy details:**

| Parameter | Value |
|---|---|
| Architecture | MLP [512 → 256 → 128], ELU |
| Algorithm | PPO (RSL-RL) |
| Observation dim | 48 |
| Action dim | 12 (joint position targets) |
| Policy frequency | 50 Hz |
| Training iterations | 20 000 |
| Trained velocity range | v_x: [−2, +3] m/s, v_y: ±1.5 m/s, ω_z: ±2 rad/s |
