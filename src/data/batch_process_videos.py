"""
Batch Process Videos

Automatically extract poses from all videos in labeled folders.
Processes entire directory structure: data/videos/[action]/*.mp4
"""

import argparse
from pathlib import Path
import subprocess
import sys


def find_video_files(base_dir: Path) -> dict:
    """
    Find all video files organized by action folders.
    
    Expected structure:
        base_dir/
            lunge/*.mp4, *.mov, *.avi
            advance/*.mp4, *.mov, *.avi
            retreat/*.mp4, *.mov, *.avi
            idle/*.mp4, *.mov, *.avi
    
    Returns:
        Dictionary mapping action -> list of video paths
    """
    video_extensions = ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']
    actions = ['idle', 'advance', 'retreat', 'lunge']
    
    videos_by_action = {}
    
    for action in actions:
        action_dir = base_dir / action
        if not action_dir.exists():
            print(f"Warning: Directory not found: {action_dir}")
            continue
        
        videos = []
        for ext in video_extensions:
            videos.extend(action_dir.glob(f'*{ext}'))
        
        videos_by_action[action] = sorted(videos)
    
    return videos_by_action


def process_video(
    video_path: Path,
    action_label: str,
    output_dir: Path,
    sequence_length: int,
    overlap: int,
    confidence: float,
    visualize: bool
) -> bool:
    """
    Process a single video using extract_poses_from_video.py
    
    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / 'extract_poses_from_video.py'
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        '--video_path', str(video_path),
        '--output_dir', str(output_dir),
        '--action_label', action_label,
        '--sequence_length', str(sequence_length),
        '--overlap', str(overlap),
        '--confidence', str(confidence)
    ]
    
    if visualize:
        cmd.append('--visualize')
    
    print(f"\n{'='*70}")
    print(f"Processing: {video_path.name}")
    print(f"Action: {action_label}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all videos in action folders"
    )
    
    parser.add_argument(
        '--videos_dir', type=str, default='src/data/videoes',
        help='Base directory containing action folders (default: src/data/videoes)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/real',
        help='Directory to save extracted pose sequences (default: data/real)'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=60,
        help='Number of frames per sequence (default: 60)'
    )
    parser.add_argument(
        '--overlap', type=int, default=30,
        help='Number of overlapping frames (default: 30)'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Minimum detection confidence (default: 0.5)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Show visualization for each video (slower)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Find all videos
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"Error: Videos directory not found: {videos_dir}")
        return
    
    print("="*70)
    print("BATCH VIDEO PROCESSING")
    print("="*70)
    print(f"\nSearching for videos in: {videos_dir}")
    
    videos_by_action = find_video_files(videos_dir)
    
    # Print summary
    print("\nFound videos:")
    print("-"*70)
    total_videos = 0
    for action, videos in videos_by_action.items():
        count = len(videos)
        total_videos += count
        print(f"  {action:>10}: {count:>3} videos")
    print("-"*70)
    print(f"  {'Total':>10}: {total_videos:>3} videos")
    print()
    
    if total_videos == 0:
        print("No videos found! Check your directory structure.")
        print(f"Expected: {videos_dir}/[action]/*.mp4")
        return
    
    if args.dry_run:
        print("\nDRY RUN - Videos that would be processed:")
        for action, videos in videos_by_action.items():
            print(f"\n{action.upper()}:")
            for video in videos:
                print(f"  - {video.name}")
        return
    
    # Process all videos
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Sequence length: {args.sequence_length} frames")
    print(f"Overlap: {args.overlap} frames")
    print(f"Confidence threshold: {args.confidence}")
    print()
    
    input("Press Enter to start processing (or Ctrl+C to cancel)...")
    
    successful = 0
    failed = 0
    
    for action, videos in videos_by_action.items():
        if not videos:
            continue
        
        print(f"\n\n{'#'*70}")
        print(f"# Processing {action.upper()} videos ({len(videos)} files)")
        print(f"{'#'*70}")
        
        for video_path in videos:
            success = process_video(
                video_path=video_path,
                action_label=action,
                output_dir=output_dir,
                sequence_length=args.sequence_length,
                overlap=args.overlap,
                confidence=args.confidence,
                visualize=args.visualize
            )
            
            if success:
                successful += 1
            else:
                failed += 1
    
    # Final summary
    print("\n\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {successful}/{total_videos} videos")
    if failed > 0:
        print(f"Failed: {failed} videos")
    
    print(f"\nExtracted sequences saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Check the data:")
    print(f"   python -c \"import pandas as pd; print(pd.read_csv('{output_dir}/labels.csv')['label'].value_counts())\"")
    print("\n2. Train the model:")
    print(f"   python src/training/train.py --data_dir {output_dir} --epochs 50 --batch_size 16")


if __name__ == '__main__':
    main()
