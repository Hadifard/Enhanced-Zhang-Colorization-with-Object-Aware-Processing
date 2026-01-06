# Technical Details - Zhang Colorization Enhanced

## Deep Dive into Implementation

This document provides in-depth technical information about the algorithms, architectures, and design decisions behind the enhanced colorization system.

---

## Table of Contents

1. [Color Space Theory](#color-space-theory)
2. [Zhang Network Architecture](#zhang-network-architecture)
3. [Segmentation Models Comparison](#segmentation-models-comparison)
4. [Facial Landmark Detection](#facial-landmark-detection)
5. [Object-by-Object Processing](#object-by-object-processing)
6. [Chroma-Based Region Detection](#chroma-based-region-detection)
7. [Color Blending Algorithms](#color-blending-algorithms)
8. [Performance Optimization](#performance-optimization)

---

## Color Space Theory

### Why Lab Color Space?

The Lab color space (also known as CIELAB) is crucial to our colorization approach:

```
Lab Color Space:
├── L (Lightness): 0-100
├── a (Green-Red): -128 to +127
└── b (Blue-Yellow): -128 to +127
```

**Advantages over RGB:**

1. **Perceptual Uniformity:** Equal distances in Lab space correspond to equal perceptual differences
2. **Separation of Luminance and Chrominance:** L channel is independent of color
3. **Natural for Colorization:** Input grayscale = L channel, predict only a and b
4. **Device Independence:** Not tied to specific display characteristics

**Mathematical Conversion:**

RGB to Lab (via XYZ):
```python
# Step 1: RGB to XYZ
X = 0.412453*R + 0.357580*G + 0.180423*B
Y = 0.212671*R + 0.715160*G + 0.072169*B
Z = 0.019334*R + 0.119193*G + 0.950227*B

# Step 2: XYZ to Lab
L = 116 * f(Y/Yn) - 16
a = 500 * (f(X/Xn) - f(Y/Yn))
b = 200 * (f(Y/Yn) - f(Z/Zn))

where f(t) = t^(1/3) if t > (6/29)^3
            = (1/3)*(29/6)^2*t + 4/29 otherwise
```

---

## Zhang Network Architecture

### Network Structure

```
Input: L channel (224x224x1)
    ↓
Conv1: 64 filters, 3x3, stride 1, ReLU
    ↓
Conv2: 128 filters, 3x3, stride 2, ReLU
    ↓
Conv3: 256 filters, 3x3, stride 1, ReLU
    ↓
Conv4: 512 filters, 3x3, stride 2, ReLU
    ↓
Conv5: 512 filters, 3x3, stride 1, ReLU, Dilation 2
    ↓
Conv6: 512 filters, 3x3, stride 1, ReLU, Dilation 2
    ↓
Conv7: 512 filters, 3x3, stride 1, ReLU
    ↓
Conv8: 313 filters, 1x1 (quantized ab prediction)
    ↓
Upsample to original resolution
    ↓
Output: ab channels (HxWx2)
```

### Key Innovations

**1. Quantized Color Space**

Instead of predicting continuous ab values, Zhang quantizes into 313 bins:

```python
# Bins are chosen by k-means clustering on ImageNet colors
# This makes training stable and predictions diverse
quantized_colors = kmeans(imagenet_colors, n_clusters=313)
```

**2. Class Rebalancing**

Rare colors (like bright orange) are upweighted during training:

```python
# Gaussian kernel weighting based on color rarity
weight = exp(-||ab_pred - ab_gt||^2 / (2*sigma^2)) / Z
loss = weight * cross_entropy(prediction, ground_truth)
```

**3. Temperature Parameter**

Controls color saturation in output:

```python
# T=0.38 (default) - realistic colors
# T=1.0 - more diverse but possibly unrealistic
# T=0.1 - very safe, potentially desaturated
p_colored = softmax(logits / T)
```

---

## Segmentation Models Comparison

### DeepLabV3+ (Default)

**Architecture:**
```
Input Image
    ↓
ResNet-101 Backbone (Atrous Convolution)
    ↓
Atrous Spatial Pyramid Pooling (ASPP)
├── 1x1 conv
├── 3x3 atrous conv, rate=6
├── 3x3 atrous conv, rate=12
├── 3x3 atrous conv, rate=18
└── Global Average Pooling
    ↓
Decoder with skip connections
    ↓
Output: Semantic segmentation map
```

**Specifications:**
- **Dataset:** PASCAL VOC (21 classes)
- **Input Size:** Any (resized internally)
- **Output:** Dense per-pixel classification
- **Speed:** ~2s per image (GPU)
- **Accuracy:** mIoU ~79% on PASCAL VOC

**Classes:**
```python
['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
 'diningtable', 'dog', 'horse', 'motorbike', 'person',
 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

---

### YOLOv8-X (Optional, Recommended)

**Architecture:**
```
Input Image
    ↓
CSPDarknet Backbone
    ↓
PAN (Path Aggregation Network)
    ↓
Detection Head + Segmentation Head
    ↓
Output: Bounding boxes + Instance masks
```

**Specifications:**
- **Dataset:** COCO (80 classes)
- **Input Size:** 640x640 (auto-resize)
- **Output:** Instance segmentation masks
- **Speed:** ~3s per image (GPU)
- **Accuracy:** mAP ~53% on COCO

**Additional Classes (not in PASCAL VOC):**
```python
['traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
 'baseball bat', 'skateboard', 'surfboard', 'tennis racket',
 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
 'hot dog', 'pizza', 'donut', 'cake', ...and more]
```

**Comparison:**

| Feature | DeepLabV3+ | YOLOv8-X |
|---------|------------|----------|
| Type | Semantic | Instance |
| Classes | 21 | 80 |
| Mask Quality | Good | Excellent |
| Speed | Faster | Slower |
| Installation | Built-in | Requires `ultralytics` |
| Best For | Simple scenes | Complex scenes |

---

## Facial Landmark Detection

### Dlib 68-Point Model

**Landmark Distribution:**
```
Facial Landmarks (68 points):
├── Jaw: 0-16 (17 points)
├── Left Eyebrow: 17-21 (5 points)
├── Right Eyebrow: 22-26 (5 points)
├── Nose Bridge: 27-30 (4 points)
├── Nose Tip: 31-35 (5 points)
├── Left Eye: 36-41 (6 points)
├── Right Eye: 42-47 (6 points)
├── Outer Lips: 48-59 (12 points)
└── Inner Lips: 60-67 (8 points)
```

**Model Details:**
- **Training:** iBUG 300-W dataset
- **Algorithm:** Ensemble of Regression Trees
- **Size:** 99.7 MB
- **Speed:** ~50ms per face (CPU)

---

### Sclera Mask Generation Algorithm

**Step-by-step process:**

```python
def generate_sclera_mask(eye_landmarks, height, width):
    """
    Generate precise mask for eye white (sclera).
    
    Strategy:
    1. Create polygon from 6 eye points
    2. Calculate eye center
    3. Estimate pupil size (15% of eye width)
    4. Exclude pupil circle
    5. Smooth transitions
    """
    
    #Step 1: Fill eye region
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_landmarks], 1)
    
    #step 2: Find center
    center_x = int(eye_landmarks[:, 0].mean())
    center_y = int(eye_landmarks[:, 1].mean())
    
    # Step 3: Estimate pupil
    eye_width = eye_landmarks[:, 0].max() - eye_landmarks[:, 0].min()
    pupil_radius = int(eye_width * 0.15)
    
    #Step 4: Remove pupil
    cv2.circle(mask, (center_x, center_y), pupil_radius, 0, -1)
    
    # Step 5: Smooth edges
    mask_float = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 2)
    mask_binary = (mask_float > 0.3).astype(np.uint8)
    
    return mask_binary
```

---

### Natural Eye and Lip Colors (Lab Values)

**Sclera (Eye White):**
```
Characteristics:
- Neutral white with slight warmth
- Not pure white (too harsh)
- Yellowish undertone for realism

Lab Values:
- L: Preserve from original (maintains lighting)
- a: 0 (perfectly neutral on green-red axis)
- b: 7 (slight yellow warmth)

Why these values?
- Pure white (a=0, b=0) looks artificial
- Slight yellow (b=7) mimics natural eye collagen
- Matches ophthalmology research on sclera color
```

**Lips (Natural Pink/Rose):**
```
Characteristics:
- Rose/pink with red dominance
- Warm undertone (not cool purple-pink)
- Works across all skin tones

Lab Values:
- L: Preserve from original (maintains texture/shading)
- a: 45 (strong red component)
- b: 25 (warm yellow undertone)

Why these values?
- High 'a' value gives vibrant red/pink
- Positive 'b' adds warmth (not cool blue-pink)
- Tested on diverse skin tones for universality
```

---

## Object-by-Object Processing

### Algorithm Workflow

```python
def colorize_with_objects(image, masks, class_names):
    """
    Process each object independently to prevent color bleeding.
    """
    
    height, width = image.shape[:2]
    
    # Initialize accumulation arrays
    ab_accumulated = np.zeros((height, width, 2))
    coverage_counter = np.zeros((height, width))
    
    # Process each object
    for mask, class_name in zip(masks, class_names):
        # 1. Extract object region with padding
        coords = np.where(mask > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add 20px padding for context
        y_min = max(0, y_min - 20)
        x_min = max(0, x_min - 20)
        y_max = min(height, y_max + 20)
        x_max = min(width, x_max + 20)
        
        #2. Crop region
        region = image[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]
        
        # 3. Colorize region
        ab_region = zhang_colorize(region)
        
        #4. Apply mask (zero out non-object pixels)
        ab_region[mask_crop == 0] = 0
        
        # 5.Accumulate
        ab_accumulated[y_min:y_max, x_min:x_max] += ab_region
        coverage_counter[y_min:y_max, x_min:x_max] += mask_crop
    
    #6. Handle overlaps (average colors)
    overlaps = coverage_counter > 1
    ab_accumulated[overlaps] /= coverage_counter[overlaps, np.newaxis]
    
    # 7. Fill gaps (background colorization)
    gaps = coverage_counter == 0
    if gaps.sum() > 0:
        ab_background = zhang_colorize(image)
        ab_accumulated[gaps] = ab_background[gaps]
    
    return ab_accumulated
```

---

### Why Padding Matters

**Without Padding:**
```
[Object boundary is cropped exactly]
Result: Zhang model lacks context
→ Colors may be inaccurate at edges
```

**With 20px Padding:**
```
[Object + 20px surrounding area]
Result: Zhang sees context around object
→ More accurate colors throughout
```

---

## Chroma-Based Region Detection

### Theory

**Chroma** measures color intensity in Lab space:

```python
chroma = sqrt(a² + b²)

Values:
- 0: Achromatic (gray/white/black)
- 5: Subtle color
- 20: Moderate color
- 50+: Vibrant color
```

### Implementation

```python
def identify_colorizable_regions(ab_channels, threshold=5.0):
    """
    Detect which regions Zhang successfully colored.
    
    This excludes:
    - Glass (naturally transparent/gray)
    - Metal/Chrome (reflective, achromatic)
    - White objects (inherently colorless)
    - Rubber/Tires (naturally black)
    """
    
    #Calculate chroma magnitude
    chroma = np.sqrt(ab_channels[:, :, 0]**2 + ab_channels[:, :, 1]**2)
    
    #threshold
    colorizable = (chroma > threshold).astype(np.uint8)
    
    #morphologicl operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Close: Fill small gaps
    colorizable = cv2.morphologyEx(colorizable, cv2.MORPH_CLOSE, kernel)
    
    #Open: Remove small noise
    colorizable = cv2.morphologyEx(colorizable, cv2.MORPH_OPEN, kernel)
    
    return colorizable
```

### Threshold Selection Guide

| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 3.0 | Very High | Detect even subtle colors |
| 5.0 | Moderate (Default) | Balanced approach |
| 7.0 | Low | Only strong colors |
| 10.0 | Very Low | Only vivid colors |

---

## Color Blending Algorithms

### Gaussian Smoothing for Transitions

```python
def smooth_color_transition(lab_image, mask, target_color):
    """
    Apply color with smooth transitions at boundaries.
    """
    
    #create smooth mask
    mask_float = mask.astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_float, (7, 7), 3)
    
    #Normalize to [0, 1]
    if mask_smooth.max() > 0:
        mask_smooth /= mask_smooth.max()
    
    # Blend colors
    lab_modified = lab_image.copy()
    
    #for each color channel (a and b)
    for channel in [1, 2]:
        lab_modified[:, :, channel] = (
            (1 - mask_smooth) * lab_image[:, :, channel] +
            mask_smooth * target_color[channel]
        )
    
    return lab_modified
```

**Gaussian Kernel Parameters:**
- **Kernel Size (7x7):** Controls transition width
- **Sigma (3):** Controls smoothness
- **Effect:** Creates natural-looking gradients

---

### Overlap Handling

When multiple objects overlap, average their colors:

```python
#Count contributions
for each object:
    coverage_counter[object_pixels] += 1
    ab_accumulated[object_pixels] += object_colors

#Average overlaps
overlapping_pixels = (coverage_counter > 1)
ab_accumulated[overlapping_pixels] /= coverage_counter[overlapping_pixels]
```

**Why averaging works:**
- Prevents one object dominating
- Smooth transitions at boundaries
- Perceptually plausible colors

---

## Performance Optimization

### Memory Management

```python
#Bad: Creates many intermediate arrays
result = cv2.resize(cv2.cvtColor(normalize(image)))

#Good: Reuse arrays
normalized = image.astype(np.float32)
normalized /= 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
```

### Batch Processing

```python
# Process multiple crops simultaneously
crops = [extract_crop(image, mask) for mask in masks]
crops_batch = np.stack(crops, axis=0)

# Single forward pass
ab_predictions = zhang_net(crops_batch)
```

### GPU Utilization

```python
#Move model to GPU
model = model.to('cuda')

# Move data to GPU
input_tensor = input_tensor.to('cuda')

#Process
with torch.no_grad():  # Disable gradients for inference
    output = model(input_tensor)
```

---

## Implementation Tips

### 1. Mask Quality Improvement

```python
def enhance_mask(mask):
    """
    Clean up noisy segmentation masks.
    """
    #Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    #Remove small blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_refined = np.zeros_like(mask)
    
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Minimum area threshold
            cv2.drawContours(mask_refined, [contour], -1, 1, thickness=cv2.FILLED)
    
    return mask_refined
```

### 2. Image Preprocessing

```python
def preprocess_for_colorization(image):
    """
    Prepare image for best colorization results.
    """
    #Denoise
    denoised = cv2.fastNlMeansDenoising(image)
    
    #enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Sharpen slightly
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened
```

### 3. Post-processing

```python
def postprocess_colorization(colorized):
    """
    Enhance final colorization output.
    """
    # Convert to Lab
    lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Slight saturation boost
    a = np.clip(a * 1.1, 0, 255).astype(np.uint8)
    b = np.clip(b * 1.1, 0, 255).astype(np.uint8)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced
```

---

## References

1. Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. ECCV.
2. Chen, L. C., et al. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. ECCV.
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv.
4. King, D. E. (2009). Dlib-ml: A machine learning toolkit. JMLR.
5. Fairchild, M. D. (2013). Color appearance models. John Wiley & Sons.

---

**Document Version:** 1.0  
**Last Updated:** January 2026 
**Author:** Hadi Sarhangi Fard
