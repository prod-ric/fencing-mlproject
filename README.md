# Fencing Action Recognition from Video

**Real-time pose-based action classification for fencing footage**

An end-to-end machine learning system that automatically recognizes four fundamental fencing actions (idle, advance, retreat, lunge) from video footage using pose estimation and temporal modeling.

---

## 🎯 Overview

This project demonstrates a complete ML pipeline for sports video analysis:

1. **Video Processing** → Extract pose keypoints from fencing videos using MediaPipe
2. **Feature Engineering** → Convert raw poses into 23 biomechanical features per frame
3. **Temporal Modeling** → Train lightweight CNN to classify action sequences
4. **Real-Time Inference** → Process live video with action predictions overlay

**Key Results:**
- 97.14% validation accuracy on training data
- Real-time inference at 30+ FPS on CPU (M2 MacBook)
- 246K parameters (~0.25MB model) suitable for edge deployment
- Works well on dynamic actions; style-dependent on static positions

**Limitations:**
- Trained on single fencer (limited style diversity)
- Struggles with wide guard positions (sometimes misclassifies as lunge)
- Best performance on clear, isolated actions

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd fencingveo

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0.0
- opencv-python >= 4.8.0
- mediapipe >= 0.10.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

### 1. Extract Poses from Videos

Organize your fencing videos by action type (there are already some default ones loaded):

```
src/data/videoes/
  ├── idle/
  │   └── idle_video.mp4
  ├── advance/
  │   └── advance_video.mp4
  ├── retreat/
  │   └── retreat_video.mp4
  └── lunge/
      └── lunge_video.mp4
```

Process all videos at once:

```bash
python src/data/batch_process_videos.py \
    --videos_dir src/data/videoes \
    --output_dir data/real \
    --sequence_length 60 \
    --overlap 30
```

Or process individual videos:

```bash
python src/data/extract_poses_from_video.py \
    --video_path path/to/video.mp4 \
    --output_dir data/real \
    --action_label lunge \
    --sequence_length 60 \
    --visualize
```

**What this does:**
- Uses MediaPipe to detect poses frame-by-frame
- Segments video into 60-frame sequences (2 seconds at 30fps)
- Saves pose sequences as `.npy` files
- Creates/updates `data/real/labels.csv` with action labels

### 2. Train the Model

```bash
python src/training/train.py \
    --data_dir data/real \
    --model_type cnn \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 0.0005
```

**Training outputs:**
- `models/best_model.pt` - Best model checkpoint
- `models/final_model.pt` - Final model after training
- `models/training_history.json` - Loss/accuracy logs
- `models/training_curves.png` - Visualization

**Expected training time:** ~5-10 minutes on CPU for 200-300 sequences

### 3. Evaluate the Model

```bash
python src/training/evaluate.py \
    --model_path models/best_model.pt \
    --data_dir data/real
```

**Evaluation outputs:**
- Per-class precision, recall, F1 scores
- Confusion matrix (saved as PNG)
- Overall accuracy metrics
- `results/evaluation_results_test.json`

### 4. Run Real-Time Inference

Process a fencing video with live predictions:

```bash
python src/inference/video_inference.py \
    --video_path path/to/test_video.mp4 \
    --model_path models/best_model.pt \
    --output_path results/annotated.mp4
```

**What you'll see:**
- Pose skeleton overlay on video
- Current action prediction with confidence
- Color-coded by action (idle=blue, advance=green, retreat=yellow, lunge=red)
- Press 'q' to quit

**Performance:** ~30 FPS on M2 MacBook (CPU only)

---

## 📊 How It Works

### 1. Pose Detection

**MediaPipe Pose** extracts 33 3D landmarks per frame, which we convert to 18 2D keypoints (OpenPose format):

```
Keypoints: Nose, Neck, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Eyes, Ears
```

### 2. Feature Engineering

From raw pose keypoints, we extract **23 biomechanical features** per frame:

- **6 joint angles**: Elbows, knees, hips (radians)
- **7 distances**: Torso, upper arms, thighs, shoulder/hip width
- **6 spatial stats**: Mean position, std deviation, span
- **2 center of mass**: (x, y) coordinates
- **2 velocities**: Frame-to-frame COM displacement

These features capture the biomechanical patterns that distinguish fencing actions.

### 3. Temporal CNN Model

**Architecture:**
```
Input: [batch, 60 frames, 23 features]
  ↓
Conv1D Block 1: 23 → 64 channels (kernel=5)
  ↓ MaxPool + Dropout
Conv1D Block 2: 64 → 128 channels
  ↓ MaxPool + Dropout
Conv1D Block 3: 128 → 256 channels
  ↓ MaxPool + Dropout
Global Average Pooling
  ↓
FC: 256 → 128 → 4 classes
```

**Total parameters:** 246,916 (~0.25MB)

The model learns temporal patterns across the 60-frame sequences:
- Early layers detect local motion (5-10 frames)
- Deeper layers recognize full action patterns (20+ frames)

### 4. Real-Time Inference

**Pipeline:**
1. Read video frame
2. MediaPipe pose detection (~20ms)
3. Maintain sliding 60-frame buffer
4. Extract features when buffer full
5. Model prediction every 15 frames (~5ms)
6. Overlay prediction on video

**Total latency:** <100ms per prediction

---

## 📁 Project Structure

```
fencingveo/
├── README.md                          # This file
├── MEDIUM_ARTICLE.md                  # Detailed technical writeup
├── requirements.txt                   # Python dependencies
├── src/
│   ├── data/
│   │   ├── batch_process_videos.py   # Batch video processing
│   │   ├── extract_poses_from_video.py  # Single video pose extraction
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   └── videoes/                  # Place your training videos here
│   ├── features/
│   │   └── pose_features.py          # Feature extraction from keypoints
│   ├── models/
│   │   ├── temporal_cnn.py           # CNN model architecture
│   │   └── lstm_model.py             # LSTM alternative (optional)
│   ├── training/
│   │   ├── train.py                  # Training pipeline
│   │   └── evaluate.py               # Model evaluation
│   └── inference/
│       └── video_inference.py        # Real-time video inference
├── data/
│   └── real/                          # Extracted pose sequences go here
├── models/                            # Trained model checkpoints
└── results/                           # Evaluation outputs, annotated videos
```

---

## 🎥 Recording Your Own Videos

**Best practices for recording:**

1. **Camera Position**: Side view, 3-5 meters from fencer
2. **Frame Rate**: 30 FPS or higher
3. **Lighting**: Good, even lighting (avoid shadows)
4. **Background**: Uncluttered, contrasting with fencer
5. **Full Body**: Keep entire body in frame throughout action
6. **Clothing**: Regular clothes work fine (no fencing gear needed)

**Video organization:**

Place videos in folders by action type:
```bash
src/data/videoes/
  idle/     # Standing in en-garde position
  advance/  # Forward footwork movements
  retreat/  # Backward footwork movements
  lunge/    # Attack lunges
```

**How many videos?**
- Minimum: 1-2 videos per action (~30-60 seconds each)
- Better: 3-5 videos per action with variation
- Ideal: Multiple fencers, different styles, various speeds

---

## 🔧 Customization

### Changing Sequence Length

Default is 60 frames (2 seconds at 30fps). Adjust based on your actions:

```bash
# Shorter sequences for quick actions
python src/data/extract_poses_from_video.py --sequence_length 30

# Longer for complex combinations
python src/data/extract_poses_from_video.py --sequence_length 90
```

### Adding More Actions

1. Create new folder in `src/data/videoes/`
2. Add videos of the new action
3. Update action list in `src/data/dataset.py`:
   ```python
   ACTIONS = ['idle', 'advance', 'retreat', 'lunge', 'parry']  # Add your action
   ```
4. Re-extract poses and retrain

### Using LSTM Model

Alternative to Temporal CNN:

```bash
python src/training/train.py \
    --data_dir data/real \
    --model_type lstm \
    --batch_size 16 \
    --epochs 50
```

LSTM has more parameters (~520K) but can capture longer-range dependencies.

---


### Improving Generalization

To make the model more robust:

1. **Record multiple fencers** - 3-5 different people with various styles
2. **Vary guard positions** - Include both compact and wide stances
3. **Mix video conditions** - Different lighting, backgrounds, camera angles
4. **Data augmentation** - Time warping, spatial jittering during training

---

## 🎓 Technical Details

### Data Format

**Pose sequences:** NumPy arrays `[T, K, 2]`
- `T` = 60 frames (time dimension)
- `K` = 18 keypoints
- Last dim = (x, y) normalized coordinates in [0, 1]

**Labels:** CSV file with columns:
- `sequence_path`: Path to .npy file
- `label`: Action name (idle/advance/retreat/lunge)

### Training Details

**Data split:** 70% train / 15% validation / 15% test  
**Optimizer:** Adam (lr=0.0005, weight_decay=1e-4)  
**Loss:** CrossEntropyLoss  
**Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)  
**Early stopping:** Patience=10 epochs on validation accuracy  

### Model Architecture

**Temporal CNN:**
- 3 Conv1D blocks (64→128→256 channels)
- Kernel size: 5 (captures ~0.15 sec patterns)
- BatchNorm + ReLU + MaxPool + Dropout(0.3)
- Global average pooling
- FC layers: 256→128→4

**Parameters:** 246,916 (~0.25MB file size)

---

## 🤝 Use Case: Sports Video Analysis

This system demonstrates capabilities relevant to sports analytics platforms:

1. **Automated Tagging** - Automatically label video segments by action type
2. **Performance Metrics** - Count action frequencies (advances per minute)
3. **Tactical Analysis** - Track movement patterns and tendencies
4. **Coaching Tools** - Identify technique issues in real-time
5. **Highlight Generation** - Detect exciting moments (lunges, exchanges)

**For Veo specifically:** The edge-deployable model (~0.25MB) can run on camera hardware, enabling real-time on-device analysis without cloud dependencies.




---

## 📄 License

MIT License - Free to use for educational and portfolio purposes.

---


