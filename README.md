# 🎯 CV Sniper (Stage 4 - High-Performance Edition)

A real-time, gesture- and eye-controlled computer vision sniper simulation game built with **Python**, **OpenCV**, and **Google MediaPipe**. 

Control the crosshair with your index finger, scope in with a finger-gun gesture, and trigger sniper shots with a natural eye blink.

---

## ⚡ Performance Optimization Highlights

The codebase has been refactored and profiled for low-latency, high-framerate gameplay:

| Component | Unoptimized / Naive | Optimized Implementation | Improvement |
|---|---|---|---|
| **Scope Optic Rendering** | 39.66 ms / frame | **3.89 ms / frame** | **10.2x Faster** (saves ~36 ms/frame) |
| **HUD Overlay Blending** | 1.75 ms / frame | **0.22 ms / frame** | **7.9x Faster** |
| **Webcam Latency** | Buffered frame lag | **`BUFFERSIZE = 1`** | **Real-time instant response** |
| **Frame Memory Allocations** | ~20 MB/frame copied | **Sub-ROI in-place memory** | **Zero full-frame allocations** |
| **Target Simulation Math** | Scalar NumPy overhead | **Pure Python math** | **Zero wrap/unwrap overhead** |

---

## 🎮 How to Play & Controls

### Motion & Eye Controls

| Action | Gesture / Movement | Visual Feedback |
|---|---|---|
| **Equip Rifle** | Raise hand into upper frame | Status: `EQUIPPED` (Cyan HUD) |
| **Scope In** | Form a **Finger Gun** (Index + Thumb extended, others curled) | Optic scope engages; surrounding world darkens |
| **Aim** | Move your index fingertip across the camera view | Crosshair smoothly tracks fingertip with exponential smoothing |
| **Fire** | **Blink** naturally (single eye or both eyes) | Muzzle flash, recoil kickback, hit/miss marker, and score sound/feedback |
| **Un-scope** | Lower hand or release finger-gun | Scope gracefully disengages |

### Keyboard Shortcuts

- `Q` or `ESC`: Quit game cleanly
- `R`: Reset score, shots, and target positions
- `S`: Toggle skeleton and eye contour overlays
- `D`: Toggle HUD diagnostic panel

---

## 🏗️ Project Architecture

```
CV-Sniper/
├── main.py                     # High-performance main game loop and camera orchestrator
├── requirements.txt            # Minimal runtime dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── cv_controller/              # Modular Computer Vision Engine
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Centralized game, camera, and threshold configuration
│   ├── detector.py             # MediaPipe task lifecycle and landmark extraction
│   ├── gestures.py             # Gesture math (finger gun, hand raised, EAR, blink detection)
│   ├── game.py                 # State machine, target physics, collision detection, and score
│   └── renderer.py             # Precomputed masks, sub-ROI optic scopes, and HUD rendering
│
├── models/                     # MediaPipe task model assets
│   ├── hand_landmarker.task    # Google MediaPipe Hand Landmarker model
│   └── face_landmarker.task    # Google MediaPipe Face Landmarker model
│
└── tests/                      # Automated test suite (15 unit & integration tests)
    └── test_cv_sniper.py       # Physics, state machine, EAR, blink, and rendering tests
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- A working webcam

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VANSHxGIT/CV-Sniper.git
   cd CV-Sniper
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files:**
   Ensure `hand_landmarker.task` and `face_landmarker.task` are present in the `models/` directory:
   - [Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
   - [Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

### 3. Launch the Game
```bash
python main.py
```

---

## 🧪 Running Automated Tests

Run the complete test suite to verify math, physics, blink logic, and OpenCV bounds checking:

```bash
python -m unittest discover -s tests -v
```

---

## ⚙️ Configuration & Customization

All tuning parameters are centralized in [`cv_controller/config.py`](file:///e:/E/College%20stuffs/Summer%20Time%20Coding%20saga/SummerTime%20Builds/CV%20Sniper/cv_controller/config.py):

- **Camera Resolution**: `FRAME_WIDTH`, `FRAME_HEIGHT` (default: 1280x720)
- **Scope Size & Dimming**: `SCOPE_RADIUS` (155px), `SCOPE_DARKNESS_ALPHA` (0.35)
- **Target Physics**: `TARGET_COUNT` (4), `TARGET_RADIUS` (22-34px), `TARGET_SPEED` (75-150 px/s)
- **Blink Calibration**: `CLOSED_EAR` (0.20), `OPEN_EAR` (0.235), `MIN_BLINK_DURATION` (0.05s)
- **Cooldowns**: `FIRE_COOLDOWN` (0.80s), `GESTURE_GRACE` (0.40s)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
