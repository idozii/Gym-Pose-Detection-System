# Gym Pose Detection (Real-Time Correct vs Incorrect Form)

## Milestone 1: Modular baseline (new)

The original notebook below remains available as a historical baseline. A new
installable package adds named 103-feature extraction, visibility checks, resettable
probability smoothing, model schema validation, tests and CLI workflows.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera,dev]'
python -m pytest -q

# Capture images for each class, then train an explicitly versioned baseline:
gym-pose capture --label correct --count 100
gym-pose capture --label incorrect --count 100
gym-pose extract
gym-pose train
gym-pose webcam
```

**Evaluation warning:** `gym-pose train` uses a random frame split for an exploratory
baseline only. For independent group evaluation, collect truthful `subject_id`/
`session_id` metadata and use `--group-column subject_id`. No real-world
form accuracy or rep-counting claims are made. Existing old joblib bundles are not
schema-versioned and must be retrained for the new CLI.

See [Milestone 1 setup, scope and limitations](docs/MILESTONE_1.md).

---

This project is structured for building and running a real-time gym form detection pipeline using:

- MediaPipe Pose for landmark extraction
- Scikit-learn for form classification
- OpenCV for webcam capture and live overlay
- YOLO weights available under `models/` for related detection experiments

The current notebook workflow supports one exercise at a time (default: squat), with two labels:

- `correct`
- `incorrect`

## Project Structure

```text
.
├── data/
│   ├── raw/                       # Captured training images by exercise/label
│   └── features/                  # Generated CSV feature files
├── models/                        # Trained classifiers and YOLO weights
├── notebooks/
│   └── gym_pose_detection.ipynb   # Main end-to-end notebook
├── README.md
└── requirements.txt
```

## Setup

Use your existing conda env or create a new one.

### Option A: Conda (recommended)

```bash
conda create -n gym_pose python=3.10 -y
conda activate gym_pose
pip install -r requirements.txt
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Workflow

Open:

- `notebooks/gym_pose_detection.ipynb`

Run cells in order:

1. Imports and path setup
2. Feature utility functions
3. Dataset capture
4. Feature CSV generation
5. Classifier training
6. Real-time webcam inference

### Data Collection

Capture images per label from webcam:

```python
collect_samples("correct", target_count=250)
collect_samples("incorrect", target_count=250)
```

Captured images are saved under:

- `data/raw/squat/correct`
- `data/raw/squat/incorrect`

### Build Features and Train

```python
build_feature_csv()
train_form_classifier()
```

Generated artifacts:

- Feature CSV: `data/features/squat_features.csv`
- Trained model: `models/squat_form_rf.joblib`

### Run Real-Time Form Detection

```python
run_realtime_form_check()
```

Controls:

- Press `q` to stop webcam inference.

## Labeling Guidance

To improve model quality:

- Keep camera angle consistent during one exercise session.
- Record different body positions, clothing, and lighting.
- Ensure each class has similar sample counts.
- Start with one exercise before adding more.

## Troubleshooting

- If webcam does not open, check camera permissions and `camera_index`.
- If MediaPipe import fails, verify package in active environment:

```bash
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
```

- If notebook paths fail, make sure you run from the repository root and use the updated notebook in `notebooks/`.