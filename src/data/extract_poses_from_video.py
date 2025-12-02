"""
Extract Pose Keypoints from Fencing Videos

Uses MediaPipe Pose to extract body keypoints from video files.
Processes videos frame-by-frame and saves pose sequences.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm


# MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Mapping from MediaPipe's 33 keypoints to our 18-keypoint format (OpenPose-like)
# MediaPipe indices: https://google.github.io/mediapipe/solutions/pose.html
MEDIAPIPE_TO_OPENPOSE_MAPPING = {
    0: 0,   # Nose
    12: 1,  # Mid-shoulder (approximate as average of shoulders)
    12: 2,  # R-Shoulder
    14: 3,  # R-Elbow
    16: 4,  # R-Wrist
    11: 5,  # L-Shoulder
    13: 6,  # L-Elbow
    15: 7,  # L-Wrist
    23: 8,  # Mid-Hip (approximate as average of hips)
    24: 9,  # R-Hip
    26: 10, # R-Knee
    28: 11, # R-Ankle
    23: 12, # L-Hip
    25: 13, # L-Knee
    27: 14, # L-Ankle
    2: 15,  # R-Eye
    5: 16,  # L-Eye
    8: 17,  # R-Ear
}


def extract_pose_from_frame(
    frame: np.ndarray,
    pose_detector,
    target_keypoints: int = 18
) -> Optional[np.ndarray]:
    """
    Extract pose keypoints from a single frame.
    
    Args:
        frame: RGB image
        pose_detector: MediaPipe pose detector instance
        target_keypoints: Number of keypoints to extract (18 for OpenPose-like)
    
    Returns:
        Array of shape [K, 2] with normalized (x, y) coordinates, or None if no pose detected
    """
    # Convert to RGB (MediaPipe expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = pose_detector.process(frame_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract keypoints
    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks.landmark
    
    # Convert to our 18-keypoint format
    keypoints = np.zeros((target_keypoints, 2))
    
    keypoints[0] = [landmarks[0].x, landmarks[0].y]  # Nose
    
    # Neck (average of shoulders)
    keypoints[1] = [
        (landmarks[11].x + landmarks[12].x) / 2,
        (landmarks[11].y + landmarks[12].y) / 2
    ]
    
    keypoints[2] = [landmarks[12].x, landmarks[12].y]  # R-Shoulder
    keypoints[3] = [landmarks[14].x, landmarks[14].y]  # R-Elbow
    keypoints[4] = [landmarks[16].x, landmarks[16].y]  # R-Wrist
    keypoints[5] = [landmarks[11].x, landmarks[11].y]  # L-Shoulder
    keypoints[6] = [landmarks[13].x, landmarks[13].y]  # L-Elbow
    keypoints[7] = [landmarks[15].x, landmarks[15].y]  # L-Wrist
    
    # Mid-Hip (average of hips)
    keypoints[8] = [
        (landmarks[23].x + landmarks[24].x) / 2,
        (landmarks[23].y + landmarks[24].y) / 2
    ]
    
    keypoints[9] = [landmarks[24].x, landmarks[24].y]   # R-Hip
    keypoints[10] = [landmarks[26].x, landmarks[26].y]  # R-Knee
    keypoints[11] = [landmarks[28].x, landmarks[28].y]  # R-Ankle
    keypoints[12] = [landmarks[23].x, landmarks[23].y]  # L-Hip
    keypoints[13] = [landmarks[25].x, landmarks[25].y]  # L-Knee
    keypoints[14] = [landmarks[27].x, landmarks[27].y]  # L-Ankle
    keypoints[15] = [landmarks[2].x, landmarks[2].y]    # R-Eye
    keypoints[16] = [landmarks[5].x, landmarks[5].y]    # L-Eye
    keypoints[17] = [landmarks[8].x, landmarks[8].y]    # R-Ear (use outer ear)
    
    # Coordinates are already normalized to [0, 1] by MediaPipe
    return keypoints


def extract_poses_from_video(
    video_path: str,
    confidence_threshold: float = 0.5,
    visualize: bool = False
) -> List[np.ndarray]:
    """
    Extract pose sequences from a video file.
    
    Args:
        video_path: Path to video file
        confidence_threshold: Minimum detection confidence
        visualize: Whether to show visualization window
    
    Returns:
        List of pose arrays, each with shape [K, 2]
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    
    poses = []
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold
    ) as pose:
        
        pbar = tqdm(total=total_frames, desc="Extracting poses")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract pose from frame
            keypoints = extract_pose_from_frame(frame, pose)
            
            if keypoints is not None:
                poses.append(keypoints)
                
                # Visualization
                if visualize:
                    # Draw keypoints on frame
                    vis_frame = frame.copy()
                    h, w = frame.shape[:2]
                    
                    for i, (x, y) in enumerate(keypoints):
                        px, py = int(x * w), int(y * h)
                        cv2.circle(vis_frame, (px, py), 5, (0, 255, 0), -1)
                        cv2.putText(vis_frame, str(i), (px+5, py-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    cv2.imshow('Pose Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            pbar.update(1)
        
        pbar.close()
    
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"Extracted {len(poses)} poses from {total_frames} frames")
    print(f"Detection rate: {len(poses)/total_frames*100:.1f}%")
    
    return poses


def segment_into_sequences(
    poses: List[np.ndarray],
    sequence_length: int = 60,
    overlap: int = 30
) -> List[np.ndarray]:
    """
    Segment a long pose sequence into fixed-length windows.
    
    Args:
        poses: List of pose arrays
        sequence_length: Number of frames per sequence
        overlap: Number of overlapping frames between sequences
    
    Returns:
        List of sequences, each with shape [T, K, 2]
    """
    if len(poses) < sequence_length:
        print(f"Warning: Video too short ({len(poses)} frames < {sequence_length})")
        # Pad with last frame if needed
        while len(poses) < sequence_length:
            poses.append(poses[-1].copy())
    
    sequences = []
    stride = sequence_length - overlap
    
    for start_idx in range(0, len(poses) - sequence_length + 1, stride):
        end_idx = start_idx + sequence_length
        sequence = np.array(poses[start_idx:end_idx])
        sequences.append(sequence)
    
    print(f"Created {len(sequences)} sequences of length {sequence_length}")
    
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose keypoints from fencing videos"
    )
    
    parser.add_argument(
        '--video_path', type=str, required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/real',
        help='Directory to save extracted pose sequences'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=60,
        help='Number of frames per sequence (at ~30fps, 60 = 2 seconds)'
    )
    parser.add_argument(
        '--overlap', type=int, default=30,
        help='Number of overlapping frames between sequences'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Minimum detection confidence (0-1)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Show visualization window during processing'
    )
    parser.add_argument(
        '--action_label', type=str, default='unknown',
        choices=['idle', 'advance', 'retreat', 'lunge', 'unknown'],
        help='Action label for this video (for annotation)'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FENCING VIDEO POSE EXTRACTION")
    print("=" * 70)
    
    # Extract poses from video
    poses = extract_poses_from_video(
        str(video_path),
        confidence_threshold=args.confidence,
        visualize=args.visualize
    )
    
    if len(poses) == 0:
        print("Error: No poses detected in video!")
        return
    
    # Segment into sequences
    sequences = segment_into_sequences(
        poses,
        sequence_length=args.sequence_length,
        overlap=args.overlap
    )
    
    # Save sequences
    video_name = video_path.stem
    labels_data = []
    
    print(f"\nSaving sequences to {output_dir}...")
    for i, sequence in enumerate(sequences):
        filename = f"{video_name}_seq_{i:03d}.npy"
        filepath = output_dir / filename
        np.save(filepath, sequence)
        
        labels_data.append({
            'file': filename,
            'label': args.action_label,
            'video_source': video_name,
            'sequence_index': i,
            'num_frames': len(sequence)
        })
    
    print(f"Saved {len(sequences)} sequences")
    
    # Save or append to labels file
    labels_path = output_dir / 'labels.csv'
    
    import pandas as pd
    new_labels_df = pd.DataFrame(labels_data)
    
    if labels_path.exists():
        # Append to existing labels
        existing_labels_df = pd.read_csv(labels_path)
        combined_df = pd.concat([existing_labels_df, new_labels_df], ignore_index=True)
        combined_df.to_csv(labels_path, index=False)
        print(f"Appended to existing labels file: {labels_path}")
    else:
        new_labels_df.to_csv(labels_path, index=False)
        print(f"Created new labels file: {labels_path}")
    
    # Save metadata
    metadata = {
        'video_path': str(video_path),
        'video_name': video_name,
        'total_frames_processed': len(poses),
        'sequences_created': len(sequences),
        'sequence_length': args.sequence_length,
        'overlap': args.overlap,
        'action_label': args.action_label,
        'keypoint_format': 'openpose_18',
        'pose_detector': 'mediapipe'
    }
    
    metadata_path = output_dir / f'{video_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review extracted sequences in: {output_dir}")
    print(f"2. If needed, re-run with correct --action_label")
    print(f"3. Once labeled, train on real data:")
    print(f"   python src/training/train.py --data_dir {output_dir}")


if __name__ == '__main__':
    main()
