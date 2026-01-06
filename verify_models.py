#!/usr/bin/env python3
"""
Model Files Verification Script
Checks if all required model files are present and have correct sizes.

Author: Hadi Sarhangi Fard
"""

import os
import sys
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def format_size(bytes_size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def check_file(filepath, expected_size_mb=None, tolerance=5):
    """
    Check if file exists and optionally verify size.
    
    Args:
        filepath: Path to file
        expected_size_mb: Expected size in MB (None to skip size check)
        tolerance: Allowed size difference in MB
        
    Returns:
        tuple: (exists, size_mb, size_ok)
    """
    if not os.path.exists(filepath):
        return False, 0, False
    
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    
    if expected_size_mb is None:
        return True, size_mb, True
    
    size_ok = abs(size_mb - expected_size_mb) <= tolerance
    return True, size_mb, size_ok

def print_header():
    """Print verification header."""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.BLUE}MODEL FILES VERIFICATION{Colors.END}")
    print("=" * 70 + "\n")

def print_section(title):
    """Print section header."""
    print(f"\n{Colors.BOLD}{title}{Colors.END}")
    print("-" * 70)

def print_file_status(filename, exists, size_mb, size_ok, expected_size_mb=None):
    """Print file verification status."""
    if exists and size_ok:
        status = f"{Colors.GREEN}✓{Colors.END}"
        size_str = format_size(size_mb * 1024 * 1024)
        print(f"  {status} {filename:<45} ({size_str})")
    elif exists and not size_ok:
        status = f"{Colors.YELLOW}⚠{Colors.END}"
        size_str = format_size(size_mb * 1024 * 1024)
        expected_str = format_size(expected_size_mb * 1024 * 1024)
        print(f"  {status} {filename:<45} ({size_str})")
        print(f"      {Colors.YELLOW}Warning: Expected ~{expected_str}{Colors.END}")
    else:
        status = f"{Colors.RED}✗{Colors.END}"
        print(f"  {status} {filename:<45} (Not found)")

def main():
    """Main verification function."""
    print_header()
    
    # Define required files
    required_files = [
        {
            'path': 'models/colorization_deploy_v2.prototxt',
            'expected_size': 0.004,  # ~4 KB
            'description': 'Zhang network architecture (Caffe format)'
        },
        {
            'path': 'models/colorization_release_v2.caffemodel',
            'expected_size': 128.99,  # ~129 MB
            'description': 'Zhang pre-trained weights'
        },
        {
            'path': 'models/pts_in_hull.npy',
            'expected_size': 0.003,  # ~3 KB
            'description': 'Color cluster centers (313 quantized bins)'
        }
    ]
    
    # Define optional files
    optional_files = [
        {
            'path': 'models/shape_predictor_68_face_landmarks.dat',
            'expected_size': 99.37,  # ~99 MB
            'description': 'Dlib 68-point facial landmark detector (optional)'
        }
    ]
    
    # Check required files
    print_section("Required Files (Zhang Colorization Network)")
    
    all_required_present = True
    total_size = 0
    
    for file_info in required_files:
        exists, size_mb, size_ok = check_file(
            file_info['path'], 
            file_info['expected_size'],
            tolerance=5
        )
        
        print_file_status(
            Path(file_info['path']).name,
            exists,
            size_mb,
            size_ok,
            file_info['expected_size']
        )
        
        if not exists:
            all_required_present = False
            print(f"      {Colors.RED}Required for all functionality{Colors.END}")
        else:
            total_size += size_mb
    
    # Check optional files
    print_section("Optional Files (Facial Feature Enhancement)")
    
    optional_present = []
    
    for file_info in optional_files:
        exists, size_mb, size_ok = check_file(
            file_info['path'],
            file_info['expected_size'],
            tolerance=5
        )
        
        print_file_status(
            Path(file_info['path']).name,
            exists,
            size_mb,
            size_ok,
            file_info['expected_size']
        )
        
        if exists:
            optional_present.append(file_info['path'])
            total_size += size_mb
            print(f"      {Colors.GREEN}Enables facial feature enhancement{Colors.END}")
        else:
            print(f"      {Colors.YELLOW}Optional: Facial feature correction not available{Colors.END}")
    
    # Print summary
    print_section("Summary")
    
    print(f"\n  Required files: ", end='')
    if all_required_present:
        print(f"{Colors.GREEN}✓ All present{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Some missing{Colors.END}")
    
    print(f"  Optional files: ", end='')
    if len(optional_present) == len(optional_files):
        print(f"{Colors.GREEN}✓ All present{Colors.END}")
    elif len(optional_present) > 0:
        print(f"{Colors.YELLOW}⚠ {len(optional_present)}/{len(optional_files)} present{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ None present{Colors.END}")
    
    print(f"\n  Total model size: {Colors.BOLD}{format_size(total_size * 1024 * 1024)}{Colors.END}")
    
    # Functionality status
    print_section("Available Functionality")
    
    if all_required_present:
        print(f"  {Colors.GREEN}✓{Colors.END} Object-aware colorization (Method 1)")
        print(f"  {Colors.GREEN}✓{Colors.END} Interactive custom recolorization (Method 3)")
        
        if len(optional_present) > 0:
            print(f"  {Colors.GREEN}✓{Colors.END} Facial feature enhancement (Method 2)")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} Facial feature enhancement (Method 2) - {Colors.YELLOW}disabled{Colors.END}")
            print(f"      Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    else:
        print(f"  {Colors.RED}✗{Colors.END} All methods disabled - missing required files")
    
    # Final status and next steps
    print("\n" + "=" * 70)
    
    if all_required_present:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ SUCCESS!{Colors.END} All required model files are ready.\n")
        print("You can now run the colorization scripts:")
        print(f"  {Colors.BLUE}python src/object_detection_colorization.py{Colors.END}")
        if len(optional_present) > 0:
            print(f"  {Colors.BLUE}python src/facial_feature_enhancement.py{Colors.END}")
        print(f"  {Colors.BLUE}python src/custom_recolorization.py{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ WARNING!{Colors.END} Some required files are missing.\n")
        print("To download missing files:")
        print(f"  {Colors.BLUE}python download_models.py{Colors.END}")
        print("\nOr download manually:")
        print(f"  {Colors.BLUE}See models/README.md for detailed instructions{Colors.END}")
    
    print("=" * 70 + "\n")
    
    # Return exit code
    return 0 if all_required_present else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⊗ Verification cancelled by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error during verification: {str(e)}{Colors.END}")
        sys.exit(1)
