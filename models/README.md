# Model Files Download Guide

This directory should contain the necessary model files for colorization and facial landmark detection.

## Required Files

### 1. Zhang Colorization Model Files (Required)

You need to download **3 files** from the original Zhang et al. repository:

| File Name | Size | Description |
|-----------|------|-------------|
| `colorization_deploy_v2.prototxt` | ~4 KB | Network architecture (Caffe format) |
| `colorization_release_v2.caffemodel` | ~129 MB | Pre-trained weights |
| `pts_in_hull.npy` | ~3 KB | Cluster center points for color quantization |

---

## Download Methods

### Method 1: Direct Download (Recommended)

**Step 1:** Download each file individually:

1. **colorization_deploy_v2.prototxt**
   ```
   https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt
   ```
   - Right-click → Save As → Save to `models/` folder

2. **colorization_release_v2.caffemodel**
   ```
   http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
   ```
   - Download and save to `models/` folder
   - **Note**: This is a large file (~129 MB)

3. **pts_in_hull.npy**
   ```
   https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy
   ```
   - Right-click → Save As → Save to `models/` folder

---

### Method 2: Using wget (Linux/Mac)

```bash
cd models/

# Download architecture
wget https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt

# Download weights
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel

# Download cluster points
wget https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy
```

---

### Method 3: Using curl (Mac/Linux)

```bash
cd models/

# Download architecture
curl -O https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt

# Download weights
curl -O http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel

# Download cluster points
curl -L -O https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy
```

---

### Method 4: Using Python Script

Create a file `download_models.py` in the project root:

```python
import urllib.request
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

files = {
    'colorization_deploy_v2.prototxt': 
        'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt',
    'colorization_release_v2.caffemodel': 
        'http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel',
    'pts_in_hull.npy': 
        'https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy'
}

print("Downloading Zhang colorization model files...")
for filename, url in files.items():
    filepath = os.path.join('models', filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✓ {filename} downloaded successfully!")

print("\n✓ All model files downloaded successfully!")
```

Then run:
```bash
python download_models.py
```

---

## Facial Landmark Model (Optional)

### Required for Facial Feature Enhancement

Download the **68-point facial landmark predictor** from dlib:

**File**: `shape_predictor_68_face_landmarks.dat` (~99 MB)

### Download Options:

#### Option 1: Direct Download
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
- Download and extract to `models/` folder

#### Option 2: Using wget
```bash
cd models/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

#### Option 3: Using curl
```bash
cd models/
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

---

## Verify Installation

After downloading, your `models/` directory should look like this:

```
models/
├── README.md
├── colorization_deploy_v2.prototxt         (4 KB)
├── colorization_release_v2.caffemodel      (129 MB)
├── pts_in_hull.npy                         (3 KB)
└── shape_predictor_68_face_landmarks.dat   (99 MB) [Optional]
```

### Quick Verification Script

Create `verify_models.py` in the project root:

```python
import os

required_files = [
    'models/colorization_deploy_v2.prototxt',
    'models/colorization_release_v2.caffemodel',
    'models/pts_in_hull.npy'
]

optional_files = [
    'models/shape_predictor_68_face_landmarks.dat'
]

print("Checking required model files...")
all_present = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"✓ {file} ({size:.2f} MB)")
    else:
        print(f"✗ {file} - MISSING!")
        all_present = False

print("\nChecking optional model files...")
for file in optional_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"✓ {file} ({size:.2f} MB)")
    else:
        print(f"✗ {file} - Not found (optional)")

if all_present:
    print("\n✓ All required model files are present!")
else:
    print("\n✗ Some required files are missing. Please download them.")
```

Run:
```bash
python verify_models.py
```

---

## Alternative Download Sources

If the original links are unavailable, you can try these mirrors:

### Google Drive Mirror
- [Zhang Models on Google Drive](https://drive.google.com/drive/folders/1ZSM4IvfjLk5kkJpHZhJL_F8wpH9Gqxw1)

### Hugging Face Mirror
- [Zhang Models on Hugging Face](https://huggingface.co/datasets/colorization/zhang-models)

---

## Important Notes

1. **Do NOT commit model files to Git**
   - Model files are large (~230 MB total)
   - They are already in `.gitignore`
   - Users should download them separately

2. **License Information**
   - Zhang models: BSD-2-Clause License
   - Dlib facial landmarks: Public domain

3. **Model File Integrity**
   - Verify file sizes after download
   - If a file is corrupted, delete and re-download

4. **Storage Requirements**
   - Minimum: 135 MB (Zhang models only)
   - With facial landmarks: 234 MB

---

## Troubleshooting

### Problem: Download fails or times out
**Solution**: Try an alternative download method or mirror

### Problem: Caffe model file is corrupted
**Solution**: 
```bash
cd models/
rm colorization_release_v2.caffemodel
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
```

### Problem: Cannot extract .bz2 file on Windows
**Solution**: Use 7-Zip or WinRAR to extract the file

### Problem: Model files not loading in code
**Solution**: Check file paths in your code match the structure above

---

## References

- [Original Zhang Colorization Repository](https://github.com/richzhang/colorization)
- [Berkeley Project Page](http://richzhang.github.io/colorization/)
- [Dlib Facial Landmarks](http://dlib.net/face_landmark_detection.py.html)

---

**Need help?** Open an issue in this repository if you encounter problems downloading the models.
