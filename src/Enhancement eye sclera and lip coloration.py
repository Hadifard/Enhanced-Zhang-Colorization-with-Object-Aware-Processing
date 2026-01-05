#!/usr/bin/env python
# coding: utf-8

# ## &#x2611; Prevents skin tone bleeding to eye sclera and Natural lip coloration &darr;

# In[ ]:


"""
Author: Hadi Sarhangi Fard

*** Implements Zhang et al.'s colorization network with custom enhancements
for object-aware processing and interactive color modification

Advanced Facial Feature Colorization System
Specialized colorization with natural eye whites and lip tones

Core Enhancements:
    - Authentic white sclera rendering (prevents skin tone bleeding)
    - Natural lip coloration with appropriate red/pink hues
    - Automatic facial landmark detection and processing
    - Precision color application to sensitive facial regions
    - Complete preservation of base colorization capabilities

Technical Requirements:
    opencv-python>=4.5.0
    numpy>=1.19.0
    torch>=1.9.0
    torchvision>=0.10.0
    dlib>=19.22.0
    
    Facial landmark predictor model:
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
import cv2
from pathlib import Path
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[WARNING] dlib library unavailable - facial feature detection disabled")


class FacialFeatureColorizer:
    """
    Advanced colorization system with anatomically accurate facial feature rendering.
    Implements specialized color correction for eye sclera and lip tissue.
    """
    
    def __init__(self, zhang_model_dir="models", landmark_predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the facial feature colorization pipeline.
        
        Parameters:
            zhang_model_dir (str): Path to Zhang model components
            landmark_predictor_path (str): Path to dlib's 68-point facial landmark model
        """
        self.device = torch.device('cpu')
        print(f"[INFO] Computation device: {self.device} (optimized for stability)")
        
        self._initialize_zhang_network(zhang_model_dir)
        self._initialize_segmentation_network()
        self._initialize_facial_detection(landmark_predictor_path)
        
    def _initialize_zhang_network(self, model_dir):
        """Configure Zhang et al. colorization model with cluster centers."""
        model_dir = Path(model_dir)
        
        network_architecture = model_dir / "colorization_deploy_v2.prototxt"
        pretrained_weights = model_dir / "colorization_release_v2.caffemodel"
        cluster_centers = model_dir / "pts_in_hull.npy"
        
        print("[INFO] Initializing Zhang colorization network...")
        self.zhang_net = cv2.dnn.readNetFromCaffe(
            str(network_architecture), 
            str(pretrained_weights)
        )
        
        pts = np.load(str(cluster_centers))
        
        class8_id = self.zhang_net.getLayerId("class8_ab")
        conv8_id = self.zhang_net.getLayerId("conv8_313_rh")
        
        pts_reshaped = pts.transpose().reshape(2, 313, 1, 1)
        self.zhang_net.getLayer(class8_id).blobs = [pts_reshaped.astype("float32")]
        self.zhang_net.getLayer(conv8_id).blobs = [
            np.full([1, 313], 2.606, dtype="float32")
        ]
        
        print("  ✓ Zhang network ready")
    
    def _initialize_segmentation_network(self):
        """Load DeepLabV3+ for semantic segmentation of image regions."""
        print("[INFO] Loading DeepLabV3+ segmentation model (CPU optimized)...")
        
        from torchvision.models.segmentation import deeplabv3_resnet50
        self.seg_model = deeplabv3_resnet50(pretrained=True)
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.pascal_voc_labels = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        print("  ✓ DeepLabV3+ initialized")
    
    def _initialize_facial_detection(self, predictor_path):
        """Setup dlib for 68-point facial landmark detection."""
        if not DLIB_AVAILABLE:
            self.face_detector = None
            self.landmark_predictor = None
            print("[INFO] Facial detection unavailable - dlib not installed")
            return
        
        predictor_file = Path(predictor_path)
        if not predictor_file.exists():
            print(f"[WARNING] Landmark predictor not found: {predictor_path}")
            print("          Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.face_detector = None
            self.landmark_predictor = None
        else:
            print("[INFO] Loading dlib facial detector...")
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(str(predictor_file))
            print("  ✓ Facial detection system active")
    
    def extract_facial_features(self, image):
        """
        Detect and extract masks for eyes and lips using facial landmarks.
        
        Returns:
            tuple: (eye_sclera_masks, lip_region_mask)
                - eye_sclera_masks: List of masks for each eye's white region
                - lip_region_mask: Combined mask for upper and lower lips
        """
        if not DLIB_AVAILABLE or self.face_detector is None:
            return [], None
        
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_detector(grayscale, 1)
        
        sclera_masks = []
        lip_mask = None
        
        img_height, img_width = image.shape[:2]
        
        for face_rect in detected_faces:
            landmarks = self.landmark_predictor(grayscale, face_rect)
            
            # Left eye landmarks (points 36-41 in 68-point model)
            left_eye_coords = np.array([
                (landmarks.part(i).x, landmarks.part(i).y) 
                for i in range(36, 42)
            ])
            
            # Right eye landmarks (points 42-47)
            right_eye_coords = np.array([
                (landmarks.part(i).x, landmarks.part(i).y) 
                for i in range(42, 48)
            ])
            
            # Lip contour landmarks (points 48-67)
            lip_coords = np.array([
                (landmarks.part(i).x, landmarks.part(i).y) 
                for i in range(48, 68)
            ])
            
            # Generate sclera mask for left eye
            left_sclera = self._generate_sclera_mask(
                left_eye_coords, img_height, img_width
            )
            sclera_masks.append(left_sclera)
            
            # Generate sclera mask for right eye
            right_sclera = self._generate_sclera_mask(
                right_eye_coords, img_height, img_width
            )
            sclera_masks.append(right_sclera)
            
            # Generate lip region mask
            lip_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(lip_mask, [lip_coords], 1)
        
        return sclera_masks, lip_mask
    
    def _generate_sclera_mask(self, eye_landmarks, height, width):
        """
        Create mask specifically for eye white (sclera) excluding pupil/iris.
        
        Strategy:
            1. Fill entire eye region polygon
            2. Calculate eye center from landmark coordinates
            3. Estimate pupil radius (approximately 15% of eye width)
            4. Remove circular pupil region from mask
            5. Apply Gaussian smoothing for natural transitions
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill complete eye region
        cv2.fillPoly(mask, [eye_landmarks], 1)
        
        # Calculate eye center point
        center_x = int(eye_landmarks[:, 0].mean())
        center_y = int(eye_landmarks[:, 1].mean())
        
        # Estimate pupil dimensions
        eye_width = eye_landmarks[:, 0].max() - eye_landmarks[:, 0].min()
        pupil_radius = int(eye_width * 0.15)
        
        # Exclude pupil from sclera mask
        cv2.circle(mask, (center_x, center_y), pupil_radius, 0, -1)
        
        # Smooth edge transitions
        mask_float = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 2)
        mask_binary = (mask_float > 0.3).astype(np.uint8)
        
        return mask_binary
    
    def apply_sclera_coloration(self, lab_image, sclera_mask):
        """
        Apply natural white coloration to eye sclera region.
        
        Sclera characteristics in Lab color space:
            L: Preserve original (luminance/brightness)
            a: ~0 (neutral, no red/green bias)
            b: ~5-8 (slight yellow warmth for natural appearance)
        
        This prevents the common issue where Zhang colorization
        incorrectly applies skin tones to eye whites.
        """
        if sclera_mask.sum() == 0:
            return lab_image
        
        lab_modified = lab_image.copy()
        
        # Create smooth transition mask
        mask_smooth = cv2.GaussianBlur(
            sclera_mask.astype(np.float32), (7, 7), 3
        )
        mask_normalized = mask_smooth / max(mask_smooth.max(), 1)
        
        # Apply sclera color values to a and b channels
        # a channel: neutral (0)
        lab_modified[:, :, 1] = (
            (1 - mask_normalized) * lab_modified[:, :, 1] + 
            mask_normalized * 0
        )
        
        # b channel: slight warm yellow (7)
        lab_modified[:, :, 2] = (
            (1 - mask_normalized) * lab_modified[:, :, 2] + 
            mask_normalized * 7
        )
        
        return lab_modified
    
    def apply_lip_coloration(self, lab_image, lip_mask):
        """
        Apply natural pink/red coloration to lip region.
        
        Natural lip color in Lab space:
            L: Preserve original (maintains lip texture/shading)
            a: ~40-45 (strong red component for vibrant appearance)
            b: ~20-25 (warm undertone for natural flesh tone)
        
        These values produce a natural rose/pink color that
        appears authentic across various skin tones.
        """
        if lip_mask is None or lip_mask.sum() == 0:
            return lab_image
        
        lab_modified = lab_image.copy()
        
        # Create soft transition mask
        mask_smooth = cv2.GaussianBlur(
            lip_mask.astype(np.float32), (9, 9), 4
        )
        mask_normalized = mask_smooth / max(mask_smooth.max(), 1)
        
        # Apply lip color values
        # a channel: strong red (45)
        lab_modified[:, :, 1] = (
            (1 - mask_normalized) * lab_modified[:, :, 1] + 
            mask_normalized * 45
        )
        
        # b channel: warm undertone (25)
        lab_modified[:, :, 2] = (
            (1 - mask_normalized) * lab_modified[:, :, 2] + 
            mask_normalized * 25
        )
        
        return lab_modified
    
    def enhance_mask_quality(self, mask):
        """Improve mask quality through morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask_refined = np.zeros_like(mask_closed)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                cv2.drawContours(mask_refined, [contour], -1, 1, thickness=cv2.FILLED)
        
        return mask_refined
    
    def perform_segmentation(self, image):
        """Execute semantic segmentation using DeepLabV3+."""
        orig_height, orig_width = image.shape[:2]
        
        # Resize for efficient processing
        max_dimension = 512
        if max(orig_height, orig_width) > max_dimension:
            scale_factor = max_dimension / max(orig_height, orig_width)
            new_height = int(orig_height * scale_factor)
            new_width = int(orig_width * scale_factor)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image
            new_height, new_width = orig_height, orig_width
        
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.seg_transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.seg_model(input_tensor)['out'][0]
        
        segmentation_result = output.argmax(0).cpu().numpy()
        segmentation_result = cv2.resize(
            segmentation_result.astype(np.uint8), 
            (orig_width, orig_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        detected_masks = []
        detected_classes = []
        bounding_boxes = []
        
        for class_index in range(1, 21):
            class_mask = (segmentation_result == class_index).astype(np.uint8)
            
            if class_mask.sum() > 1000:
                refined_mask = self.enhance_mask_quality(class_mask)
                
                if refined_mask.sum() > 1000:
                    coordinates = np.column_stack(np.where(refined_mask > 0))
                    if len(coordinates) > 0:
                        y_min, x_min = coordinates.min(axis=0)
                        y_max, x_max = coordinates.max(axis=0)
                        
                        detected_masks.append(refined_mask)
                        detected_classes.append(self.pascal_voc_labels[class_index])
                        bounding_boxes.append([x_min, y_min, x_max, y_max])
        
        return detected_masks, detected_classes, bounding_boxes
    
    def colorize_masked_region(self, image, region_mask):
        """Apply Zhang colorization to a specific masked region."""
        img_height, img_width = image.shape[:2]
        
        coordinates = np.column_stack(np.where(region_mask > 0))
        if len(coordinates) == 0:
            return None
        
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        
        # Add padding for context
        pad_size = 30
        y_min_padded = max(0, y_min - pad_size)
        x_min_padded = max(0, x_min - pad_size)
        y_max_padded = min(img_height, y_max + pad_size)
        x_max_padded = min(img_width, x_max + pad_size)
        
        region = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded].copy()
        region_mask_cropped = region_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
        
        # Convert to Lab color space
        normalized = region.astype("float32") / 255.0
        lab_region = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        
        region_height, region_width = region.shape[:2]
        resized_lab = cv2.resize(lab_region, (224, 224))
        L_channel = cv2.split(resized_lab)[0]
        L_channel -= 50
        
        # Generate ab predictions
        self.zhang_net.setInput(cv2.dnn.blobFromImage(L_channel))
        ab_predicted = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize back to original region dimensions
        ab_resized = cv2.resize(
            ab_predicted, 
            (region_width, region_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Create smooth mask transition
        mask_float = region_mask_cropped.astype(np.float32)
        mask_smooth = cv2.GaussianBlur(mask_float, (21, 21), 10)
        
        if mask_smooth.max() > 0:
            mask_smooth = mask_smooth / mask_smooth.max()
        
        ab_masked = ab_resized * mask_smooth[:, :, np.newaxis]
        
        return ab_masked, (y_min_padded, x_min_padded, y_max_padded, x_max_padded), mask_smooth
    
    def generate_annotated_visualization(self, image, masks, classes, eye_masks, lip_mask):
        """Create annotated image showing detected objects and facial features."""
        annotated = image.copy()
        
        # Generate consistent colors for each class
        np.random.seed(42)
        color_palette = {}
        for class_name in set(classes):
            color_palette[class_name] = tuple(map(int, np.random.randint(80, 220, 3)))
        
        # Draw detected objects
        for mask, class_label in zip(masks, classes):
            color = color_palette[class_label]
            
            # Create colored overlay
            colored_overlay = np.zeros_like(annotated)
            colored_overlay[mask > 0] = color
            annotated = cv2.addWeighted(annotated, 0.7, colored_overlay, 0.3, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, color, 3)
            
            # Add class label
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                label_y, label_x = coords.min(axis=0)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(
                    class_label, font, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated, 
                    (label_x - 5, label_y - text_height - 15), 
                    (label_x + text_width + 5, label_y - 5), 
                    color, -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated, class_label, (label_x, label_y - 10),
                    font, font_scale, (255, 255, 255), thickness
                )
        
        # Highlight eye regions with cyan
        for eye_mask in eye_masks:
            contours, _ = cv2.findContours(
                eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (255, 255, 0), 6)
        
        # Highlight lip region with red
        if lip_mask is not None:
            contours, _ = cv2.findContours(
                lip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 6)
        
        return annotated
    
    def process_complete_colorization(self, image_input):
        """
        Execute full colorization pipeline with natural facial feature enhancement.
        
        Pipeline stages:
            1. Facial feature detection (eyes, lips)
            2. Semantic segmentation (objects)
            3. Annotated visualization generation
            4. Object-by-object colorization
            5. Facial feature color correction
        
        Returns:
            tuple: (original, annotated, colorized, masks, classes)
        """
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
        else:
            image = image_input
        
        img_height, img_width = image.shape[:2]
        
        print(f"\n{'=' * 70}")
        print(f"Processing: {image_input if isinstance(image_input, str) else 'image array'}")
        print(f"Dimensions: {img_width}x{img_height}")
        print(f"{'=' * 70}\n")
        
        # Stage 1: Facial feature extraction
        print("[1/5] Extracting facial features...")
        sclera_masks, lip_mask = self.extract_facial_features(image)
        if sclera_masks:
            print(f"      Located {len(sclera_masks)} eye regions and lip contours\n")
        else:
            print("      No facial features detected (using standard colorization)\n")
        
        # Stage 2: Object segmentation
        print("[2/5] Performing semantic segmentation...")
        object_masks, object_classes, bounding_boxes = self.perform_segmentation(image)
        unique_classes = ', '.join(set(object_classes))
        print(f"      Detected {len(object_masks)} objects: {unique_classes}\n")
        
        # Stage 3: Generate annotated visualization
        print("[3/5] Generating annotated visualization...")
        annotated_image = self.generate_annotated_visualization(
            image, object_masks, object_classes, sclera_masks, lip_mask
        )
        
        # Stage 4: Colorization process
        print("[4/5] Applying Zhang colorization...")
        
        # Prepare Lab color space
        normalized = image.astype("float32") / 255.0
        lab_full = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        L_channel_full = cv2.split(lab_full)[0]
        
        # Colorize background
        print("      - Processing background layer...")
        resized_lab = cv2.resize(lab_full, (224, 224))
        L_background = cv2.split(resized_lab)[0]
        L_background -= 50
        
        self.zhang_net.setInput(cv2.dnn.blobFromImage(L_background))
        ab_background = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_background = cv2.resize(
            ab_background, (img_width, img_height), interpolation=cv2.INTER_CUBIC
        )
        
        ab_composite = ab_background.copy()
        
        # Colorize each detected object
        for idx, (mask, class_name) in enumerate(zip(object_masks, object_classes)):
            print(f"      - Processing {class_name} ({idx + 1}/{len(object_masks)})...")
            
            colorization_result = self.colorize_masked_region(image, mask)
            
            if colorization_result is not None:
                ab_object, (y1, x1, y2, x2), smooth_mask = colorization_result
                
                # Blend object colors into composite
                for channel in range(2):
                    ab_composite[y1:y2, x1:x2, channel] = (
                        smooth_mask * ab_object[:, :, channel] +
                        (1 - smooth_mask) * ab_composite[y1:y2, x1:x2, channel]
                    )
        
        # Stage 5: Apply natural facial colors
        print("\n[5/5] Enhancing facial features with natural colors...")
        
        # Reconstruct complete Lab image
        colorized_lab = np.concatenate(
            (L_channel_full[:, :, np.newaxis], ab_composite), axis=2
        )
        
        # Apply natural sclera coloration
        if sclera_masks:
            print("      - Applying natural white to eye sclera...")
            for sclera_mask in sclera_masks:
                colorized_lab = self.apply_sclera_coloration(colorized_lab, sclera_mask)
        
        # Apply natural lip coloration
        if lip_mask is not None:
            print("      - Applying natural rose tone to lips...")
            colorized_lab = self.apply_lip_coloration(colorized_lab, lip_mask)
        
        # Convert back to BGR color space
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        colorized_bgr = (255 * colorized_bgr).astype("uint8")
        
        print(f"\n✓ Colorization complete!\n{'=' * 70}\n")
        
        return image, annotated_image, colorized_bgr, object_masks, object_classes


def main():
    """Main execution function with user interaction."""
    print("\n" + "=" * 70)
    print("NATURAL FACIAL FEATURE COLORIZATION SYSTEM")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Authentic white sclera rendering")
    print("  ✓ Natural rose/pink lip tones")
    print("  ✓ Automatic facial landmark detection")
    print("  ✓ Full object-aware colorization capabilities")
    print("=" * 70 + "\n")
    
    # Discover available images
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    available_images = []
    
    for format_ext in supported_formats:
        available_images.extend(list(Path('.').glob(f'*{format_ext}')))
    
    available_images = [
        img for img in available_images 
        if 'colorized' not in str(img)
    ]
    
    if not available_images:
        print("[ERROR] No compatible images found in working directory")
        return
    
    print("Available images:")
    for idx, img_path in enumerate(available_images, 1):
        print(f"  {idx}. {img_path.name}")
    
    user_selection = input("\nSelect image number (press Enter for first image): ").strip()
    selected_image = (
        available_images[0] if not user_selection 
        else available_images[int(user_selection) - 1]
    )
    
    # Initialize colorization system
    colorizer = FacialFeatureColorizer()
    
    # Execute colorization
    original, annotated, colorized, masks, classes = colorizer.process_complete_colorization(
        selected_image
    )
    
    # Save results
    output_directory = Path("colorized_output")
    output_directory.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_directory / f"1_original_{selected_image.name}"), original)
    cv2.imwrite(str(output_directory / f"2_annotated_{selected_image.name}"), annotated)
    cv2.imwrite(str(output_directory / f"3_colorized_{selected_image.name}"), colorized)
    
    print(f"Results saved to: {output_directory.absolute()}/")
    print(f"  - 1_original_{selected_image.name}")
    print(f"  - 2_annotated_{selected_image.name}")
    print(f"  - 3_colorized_{selected_image.name}")
    
    # Prepare display
    max_display_height = 1000
    current_height = original.shape[0]
    
    if current_height > max_display_height:
        scale = max_display_height / current_height
        display_width = int(original.shape[1] * scale)
        
        original_resized = cv2.resize(original, (display_width, max_display_height))
        annotated_resized = cv2.resize(annotated, (display_width, max_display_height))
        colorized_resized = cv2.resize(colorized, (display_width, max_display_height))
    else:
        original_resized = original
        annotated_resized = annotated
        colorized_resized = colorized
    
    def add_title_banner(img, title):
        """Add title banner to image top."""
        img_with_title = img.copy()
        cv2.rectangle(img_with_title, (0, 0), (img.shape[1], 35), (50, 50, 50), -1)
        cv2.putText(
            img_with_title, title, (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return img_with_title
    
    original_resized = add_title_banner(original_resized, "Original")
    annotated_resized = add_title_banner(
        annotated_resized, "Detection (Eyes=Cyan, Lips=Red)"
    )
    colorized_resized = add_title_banner(colorized_resized, "Natural Colorization")
    
    comparison_panel = np.hstack([original_resized, annotated_resized, colorized_resized])
    
    cv2.imshow("Colorization Results - Press any key to exit", comparison_panel)
    cv2.imwrite(str(output_directory / f"4_comparison_{selected_image.name}"), comparison_panel)
    
    print("\nDisplaying results... Press any key to close window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

