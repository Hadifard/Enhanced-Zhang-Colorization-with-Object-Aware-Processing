#!/usr/bin/env python3
"""
Automatic Model Downloader for Zhang Colorization Enhanced
Downloads all required model files for the colorization system.
"""

import urllib.request
import os
import sys

def download_file(url, filename, description):
    """Download a file with progress indicator."""
    filepath = os.path.join('models', filename)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    print('='*60)
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            size_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, report_progress)
        print(f"\n✓ {filename} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {str(e)}")
        return False

def main():
    print("""
╔═════════════════════════════════════════════════════════╗
║     Zhang Colorization Enhanced - Model Downloader      ║
║                                                         ║
║  This script will download all required model files:    ║
║  1. Zhang colorization model (3 files, ~132 MB)         ║
║  2. Dlib facial landmarks (optional, ~99 MB)            ║
║                                                         ║
║  Total download size: ~231 MB                           ║
╚═════════════════════════════════════════════════════════╝
    """)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    print("✓ Models directory ready")
    
    # Zhang colorization files
    zhang_files = {
        'colorization_deploy_v2.prototxt': {
            'url': 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt',
            'description': 'Zhang Network Architecture'
        },
        'colorization_release_v2.caffemodel': {
            'url': 'http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel',
            'description': 'Zhang Pre-trained Weights (~129 MB)'
        },
        'pts_in_hull.npy': {
            'url': 'https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy',
            'description': 'Color Cluster Centers'
        }
    }
    
    print("\n" + "="*60)
    print("PHASE 1: Downloading Zhang Colorization Model Files")
    print("="*60)
    
    success_count = 0
    for filename, info in zhang_files.items():
        if download_file(info['url'], filename, info['description']):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Zhang Model Files: {success_count}/{len(zhang_files)} downloaded successfully")
    print('='*60)
    
    # Ask about facial landmark model
    print("\n" + "="*60)
    print("PHASE 2: Facial Landmark Model (Optional)")
    print("="*60)
    print("\nThe facial landmark model enables:")
    print("  • Natural eye sclera coloring")
    print("  • Proper lip coloration")
    print("  • Enhanced portrait results")
    print("\nSize: ~99 MB")
    
    response = input("\nDownload facial landmark model? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        landmark_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        landmark_file = 'shape_predictor_68_face_landmarks.dat.bz2'
        
        if download_file(landmark_url, landmark_file, 'Dlib 68-Point Facial Landmarks'):
            print("\n" + "="*60)
            print("Extracting compressed file...")
            print("="*60)
            
            try:
                import bz2
                import shutil
                
                compressed_path = os.path.join('models', landmark_file)
                extracted_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
                
                with bz2.open(compressed_path, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                os.remove(compressed_path)
                print("✓ Facial landmark model extracted successfully!")
                success_count += 1
                
            except Exception as e:
                print(f"✗ Error extracting file: {str(e)}")
                print("Please extract manually using bunzip2 or 7-Zip")
    else:
        print("⊗ Skipping facial landmark model download")
    
    # Final summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    # Check what files exist
    files_to_check = [
        ('colorization_deploy_v2.prototxt', 'Zhang Architecture'),
        ('colorization_release_v2.caffemodel', 'Zhang Weights'),
        ('pts_in_hull.npy', 'Cluster Centers'),
        ('shape_predictor_68_face_landmarks.dat', 'Facial Landmarks (Optional)')
    ]
    
    print("\nFiles in models/ directory:")
    for filename, description in files_to_check:
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename:<45} ({size_mb:>6.2f} MB)")
        else:
            print(f"  ✗ {filename:<45} (Not found)")
    
    # Check if minimum requirements met
    required_files = [
        'models/colorization_deploy_v2.prototxt',
        'models/colorization_release_v2.caffemodel',
        'models/pts_in_hull.npy'
    ]
    
    all_required_present = all(os.path.exists(f) for f in required_files)
    
    print("\n" + "="*60)
    if all_required_present:
        print("✓ SUCCESS! All required model files are ready.")
        print("\nYou can now run the colorization scripts:")
        print("  python src/object_detection_colorization.py")
        print("  python src/facial_feature_enhancement.py")
        print("  python src/custom_recolorization.py")
    else:
        print("✗ WARNING: Some required files are missing!")
        print("\nPlease try downloading them manually:")
        print("  See models/README.md for instructions")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⊗ Download cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        sys.exit(1)
