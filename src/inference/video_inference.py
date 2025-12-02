"""
Real-Time Video Action Recognition

Process fencing videos and predict actions in real-time or batch mode.
Combines pose extraction with trained action recognition model.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import torch
from pathlib import Path
from collections import deque
import sys
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from features.pose_features import extract_sequence_features
from models.temporal_cnn import create_temporal_cnn
from models.lstm_model import create_lstm_classifier
from data.dataset import ACTIONS


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_pose_from_frame(frame: np.ndarray, pose_detector) -> Optional[np.ndarray]:
    """Extract 18 keypoints from a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)
    
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    keypoints = np.zeros((18, 2))
    
    keypoints[0] = [landmarks[0].x, landmarks[0].y]  # Nose
    keypoints[1] = [(landmarks[11].x + landmarks[12].x) / 2, (landmarks[11].y + landmarks[12].y) / 2]  # Neck
    keypoints[2] = [landmarks[12].x, landmarks[12].y]  # R-Shoulder
    keypoints[3] = [landmarks[14].x, landmarks[14].y]  # R-Elbow
    keypoints[4] = [landmarks[16].x, landmarks[16].y]  # R-Wrist
    keypoints[5] = [landmarks[11].x, landmarks[11].y]  # L-Shoulder
    keypoints[6] = [landmarks[13].x, landmarks[13].y]  # L-Elbow
    keypoints[7] = [landmarks[15].x, landmarks[15].y]  # L-Wrist
    keypoints[8] = [(landmarks[23].x + landmarks[24].x) / 2, (landmarks[23].y + landmarks[24].y) / 2]  # Mid-Hip
    keypoints[9] = [landmarks[24].x, landmarks[24].y]   # R-Hip
    keypoints[10] = [landmarks[26].x, landmarks[26].y]  # R-Knee
    keypoints[11] = [landmarks[28].x, landmarks[28].y]  # R-Ankle
    keypoints[12] = [landmarks[23].x, landmarks[23].y]  # L-Hip
    keypoints[13] = [landmarks[25].x, landmarks[25].y]  # L-Knee
    keypoints[14] = [landmarks[27].x, landmarks[27].y]  # L-Ankle
    keypoints[15] = [landmarks[2].x, landmarks[2].y]    # R-Eye
    keypoints[16] = [landmarks[5].x, landmarks[5].y]    # L-Eye
    keypoints[17] = [landmarks[8].x, landmarks[8].y]    # R-Ear
    
    return keypoints


def load_model(model_path: str, device: torch.device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint['model_type']
    feature_dim = checkpoint['feature_dim']
    num_classes = checkpoint['num_classes']
    
    if model_type == 'cnn':
        model = create_temporal_cnn(input_dim=feature_dim, num_classes=num_classes)
    elif model_type == 'lstm':
        model = create_lstm_classifier(input_dim=feature_dim, num_classes=num_classes, use_attention=False)
    elif model_type == 'lstm_attention':
        model = create_lstm_classifier(input_dim=feature_dim, num_classes=num_classes, use_attention=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict_action(model, pose_sequence, device):
    """Predict action from pose sequence."""
    if len(pose_sequence) < 30:  # Need minimum frames
        return None, None
    
    # Convert to numpy array
    sequence_array = np.array(list(pose_sequence))
    
    # Extract features
    features = extract_sequence_features(sequence_array)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_idx].item()
    
    return ACTIONS[predicted_idx], confidence


def process_video(
    video_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    sequence_length: int = 60,
    display: bool = True
):
    """Process video and predict actions in real-time."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer if output specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Sliding window of poses
    pose_buffer = deque(maxlen=sequence_length)
    current_action = "warming up..."
    current_confidence = 0.0
    
    print(f"\nProcessing video (press 'q' to quit)...")
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Extract pose
            keypoints = extract_pose_from_frame(frame, pose)
            
            if keypoints is not None:
                pose_buffer.append(keypoints)
                
                # Predict every 15 frames (twice per second at 30fps)
                if len(pose_buffer) == sequence_length and frame_count % 15 == 0:
                    action, confidence = predict_action(model, pose_buffer, device)
                    if action:
                        current_action = action
                        current_confidence = confidence
            
            # Draw results on frame
            display_frame = frame.copy()
            
            # Draw pose skeleton if detected
            if keypoints is not None:
                h, w = frame.shape[:2]
                for i, (x, y) in enumerate(keypoints):
                    px, py = int(x * w), int(y * h)
                    cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)
            
            # Draw prediction overlay
            overlay = display_frame.copy()
            
            # Semi-transparent background
            cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Text
            cv2.putText(display_frame, f"Action: {current_action.upper()}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {current_confidence*100:.1f}%", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Buffer status
            buffer_percent = len(pose_buffer) / sequence_length * 100
            cv2.putText(display_frame, f"Buffer: {buffer_percent:.0f}%", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Save frame if writing output
            if out:
                out.write(display_frame)
            
            # Display
            if display:
                cv2.imshow('Fencing Action Recognition', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames")
    if output_path:
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time fencing action recognition from video"
    )
    
    parser.add_argument(
        '--video_path', type=str, required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save output video with predictions (optional)'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=60,
        help='Number of frames for action prediction'
    )
    parser.add_argument(
        '--no_display', action='store_true',
        help='Disable video display window'
    )
    
    args = parser.parse_args()
    
    process_video(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()
