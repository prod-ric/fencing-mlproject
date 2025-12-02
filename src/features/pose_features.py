"""
Pose Feature Extraction

Converts raw pose keypoints into meaningful features for action classification:
- Joint angles (knee, elbow, hip)
- Body segment distances
- Temporal velocities
- Spatial statistics

These features capture biomechanical properties relevant to fencing actions.
"""

import numpy as np
from typing import Tuple, List


def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 2D points.
    
    Args:
        p1: Point 1 with shape [2] (x, y)
        p2: Point 2 with shape [2] (x, y)
    
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute angle at p2 formed by three points p1-p2-p3.
    
    Args:
        p1, p2, p3: Points with shape [2] (x, y)
        p2 is the vertex of the angle
    
    Returns:
        Angle in radians [0, π]
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle


def extract_joint_angles(pose: np.ndarray) -> np.ndarray:
    """
    Extract key joint angles from a single pose frame.
    
    Keypoint indices (OpenPose body model):
        2: RShoulder, 3: RElbow, 4: RWrist
        5: LShoulder, 6: LElbow, 7: LWrist
        9: RHip, 10: RKnee, 11: RAnkle
        12: LHip, 13: LKnee, 14: LAnkle
    
    Args:
        pose: Keypoints with shape [K, 2]
    
    Returns:
        Array of joint angles [6] (in radians):
            [r_elbow, l_elbow, r_knee, l_knee, r_hip, l_hip]
    """
    angles = np.zeros(6)
    
    # Right elbow angle (shoulder-elbow-wrist)
    if pose.shape[0] > 4:
        angles[0] = compute_angle(pose[2], pose[3], pose[4])
    
    # Left elbow angle
    if pose.shape[0] > 7:
        angles[1] = compute_angle(pose[5], pose[6], pose[7])
    
    # Right knee angle (hip-knee-ankle)
    if pose.shape[0] > 11:
        angles[2] = compute_angle(pose[9], pose[10], pose[11])
    
    # Left knee angle
    if pose.shape[0] > 14:
        angles[3] = compute_angle(pose[12], pose[13], pose[14])
    
    # Right hip angle (knee-hip-shoulder)
    if pose.shape[0] > 10:
        angles[4] = compute_angle(pose[10], pose[9], pose[2])
    
    # Left hip angle
    if pose.shape[0] > 13:
        angles[5] = compute_angle(pose[13], pose[12], pose[5])
    
    return angles


def extract_distances(pose: np.ndarray) -> np.ndarray:
    """
    Extract key body segment distances from a single pose frame.
    
    Args:
        pose: Keypoints with shape [K, 2]
    
    Returns:
        Array of distances [7]:
            [torso_length, r_upper_arm, l_upper_arm, r_thigh, l_thigh,
             shoulder_width, hip_width]
    """
    distances = np.zeros(7)
    
    # Torso length (neck to mid-hip)
    if pose.shape[0] > 8:
        distances[0] = compute_distance(pose[1], pose[8])
    
    # Right upper arm (shoulder to elbow)
    if pose.shape[0] > 3:
        distances[1] = compute_distance(pose[2], pose[3])
    
    # Left upper arm
    if pose.shape[0] > 6:
        distances[2] = compute_distance(pose[5], pose[6])
    
    # Right thigh (hip to knee)
    if pose.shape[0] > 10:
        distances[3] = compute_distance(pose[9], pose[10])
    
    # Left thigh
    if pose.shape[0] > 13:
        distances[4] = compute_distance(pose[12], pose[13])
    
    # Shoulder width
    if pose.shape[0] > 5:
        distances[5] = compute_distance(pose[2], pose[5])
    
    # Hip width
    if pose.shape[0] > 12:
        distances[6] = compute_distance(pose[9], pose[12])
    
    return distances


def compute_center_of_mass(pose: np.ndarray) -> np.ndarray:
    """
    Compute approximate center of mass from pose keypoints.
    Uses weighted average of key body points.
    
    Args:
        pose: Keypoints with shape [K, 2]
    
    Returns:
        Center of mass [2] (x, y)
    """
    # Key points for COM: neck, mid-hip, shoulders, hips
    # Weights approximate body mass distribution
    if pose.shape[0] >= 13:
        key_points = pose[[1, 8, 2, 5, 9, 12]]  # neck, mid-hip, shoulders, hips
        weights = np.array([0.15, 0.35, 0.1, 0.1, 0.15, 0.15])  # torso-heavy
        com = np.average(key_points, axis=0, weights=weights)
    else:
        # Fallback to simple mean
        com = np.mean(pose, axis=0)
    
    return com


def extract_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity of center of mass.
    
    Args:
        sequence: Pose sequence with shape [T, K, 2]
    
    Returns:
        Velocities with shape [T-1, 2] (vx, vy for each frame transition)
    """
    T = sequence.shape[0]
    velocities = np.zeros((T - 1, 2))
    
    for t in range(T - 1):
        com_t = compute_center_of_mass(sequence[t])
        com_t1 = compute_center_of_mass(sequence[t + 1])
        velocities[t] = com_t1 - com_t
    
    return velocities


def extract_spatial_statistics(pose: np.ndarray) -> np.ndarray:
    """
    Extract spatial distribution statistics from a single pose.
    
    Args:
        pose: Keypoints with shape [K, 2]
    
    Returns:
        Statistics array [6]: [mean_x, mean_y, std_x, std_y, span_x, span_y]
    """
    stats = np.zeros(6)
    
    # Mean position
    stats[0] = np.mean(pose[:, 0])
    stats[1] = np.mean(pose[:, 1])
    
    # Standard deviation (spread of keypoints)
    stats[2] = np.std(pose[:, 0])
    stats[3] = np.std(pose[:, 1])
    
    # Span (range)
    stats[4] = np.max(pose[:, 0]) - np.min(pose[:, 0])
    stats[5] = np.max(pose[:, 1]) - np.min(pose[:, 1])
    
    return stats


def extract_frame_features(pose: np.ndarray) -> np.ndarray:
    """
    Extract all features from a single pose frame.
    
    Args:
        pose: Keypoints with shape [K, 2]
    
    Returns:
        Feature vector [21]:
            - 6 joint angles
            - 7 distances
            - 6 spatial statistics
            - 2 center of mass coordinates
    """
    features = []
    
    # Joint angles (6)
    features.append(extract_joint_angles(pose))
    
    # Distances (7)
    features.append(extract_distances(pose))
    
    # Spatial statistics (6)
    features.append(extract_spatial_statistics(pose))
    
    # Center of mass (2)
    features.append(compute_center_of_mass(pose))
    
    return np.concatenate(features)


def extract_sequence_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract features from an entire pose sequence.
    
    Combines per-frame features with temporal features (velocities).
    
    Args:
        sequence: Pose sequence with shape [T, K, 2]
    
    Returns:
        Feature sequence with shape [T, F] where F is the feature dimension:
            - 21 frame features per timestep
            - 2 velocity features (padded with zeros for first frame)
        Total: F = 23 features per frame
    """
    T = sequence.shape[0]
    
    # Extract per-frame features
    frame_features = []
    for t in range(T):
        features = extract_frame_features(sequence[t])
        frame_features.append(features)
    
    frame_features = np.array(frame_features)  # [T, 21]
    
    # Extract velocities
    velocities = extract_velocity(sequence)  # [T-1, 2]
    
    # Pad velocities with zeros for first frame
    velocities = np.vstack([np.zeros(2), velocities])  # [T, 2]
    
    # Concatenate all features
    all_features = np.concatenate([frame_features, velocities], axis=1)  # [T, 23]
    
    return all_features


def normalize_features(features: np.ndarray, 
                       mean: np.ndarray = None, 
                       std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array with shape [N, T, F] or [T, F]
        mean: Pre-computed mean (optional, will compute if None)
        std: Pre-computed std (optional, will compute if None)
    
    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=tuple(range(features.ndim - 1)), keepdims=True)
    
    if std is None:
        std = np.std(features, axis=tuple(range(features.ndim - 1)), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean.squeeze(), std.squeeze()


if __name__ == '__main__':
    """
    Test feature extraction on a sample sequence.
    """
    # Create a dummy sequence
    dummy_sequence = np.random.rand(60, 18, 2)  # 60 frames, 18 keypoints
    
    print("Testing feature extraction...")
    print(f"Input sequence shape: {dummy_sequence.shape}")
    
    # Extract features
    features = extract_sequence_features(dummy_sequence)
    print(f"Output feature shape: {features.shape}")
    print(f"Features per frame: {features.shape[1]}")
    
    # Test individual components
    single_frame = dummy_sequence[0]
    angles = extract_joint_angles(single_frame)
    distances = extract_distances(single_frame)
    stats = extract_spatial_statistics(single_frame)
    com = compute_center_of_mass(single_frame)
    
    print(f"\nPer-frame feature breakdown:")
    print(f"  Joint angles: {angles.shape[0]}")
    print(f"  Distances: {distances.shape[0]}")
    print(f"  Spatial stats: {stats.shape[0]}")
    print(f"  Center of mass: {com.shape[0]}")
    print(f"  Velocities: 2")
    print(f"  Total: {angles.shape[0] + distances.shape[0] + stats.shape[0] + com.shape[0] + 2}")
    
    print("\nFeature extraction test passed! ✓")
