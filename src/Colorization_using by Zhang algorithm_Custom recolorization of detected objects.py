#!/usr/bin/env python
# coding: utf-8

# ## &#x2611; Interactive Object Recolorization System &darr;

# In[ ]:


"""
Author: Hadi Sarhangi Fard

*** Implements Zhang et al.'s colorization network with custom enhancements
for object-aware processing and interactive color modification

Interactive Object Recolorization System
Smart region detection with manual color override capabilities

Core Features:
    - Intelligent detection of colorizable regions (excludes glass, rubber, lights)
    - Visual map showing recolorable areas with percentage metrics
    - Manual color application restricted to valid regions only
    - Automatic colorization with Zhang model as baseline

Technical Approach:
    1. Zhang model provides initial colorization
    2. System identifies strongly colored regions using threshold analysis
    3. Only detected regions are marked as recolorable
    4. User can view and modify colors within permitted boundaries

Dependencies:
    opencv-python>=4.5.0
    numpy>=1.19.0
    torch>=1.9.0
    torchvision>=0.10.0
    ultralytics (optional, for enhanced segmentation)
"""

import numpy as np
import cv2
from pathlib import Path
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional


class AdaptiveRecolorizer:
    """
    Advanced colorization system with intelligent region detection
    and selective color override functionality.
    """
    
    def __init__(self, zhang_model_dir="model", use_yolo=False, 
                 chroma_threshold=5.0):
        """
        Initialize recolorization pipeline with smart region detection.
        
        Parameters:
            zhang_model_dir (str): Path to Zhang model files
            use_yolo (bool): Enable YOLOv8 for superior segmentation
            chroma_threshold (float): Color intensity threshold for detectability
                                     (5.0 = moderate, 3.0 = sensitive, 10.0 = strict)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chroma_threshold = chroma_threshold
        
        print(f"[INFO] Computation device: {self.device}")
        print(f"[INFO] Chroma detection threshold: {chroma_threshold}")
        
        self._initialize_zhang_network(zhang_model_dir)
        self.use_yolo = use_yolo
        self._initialize_segmentation_network()
        self._configure_color_presets()
    
    def _initialize_zhang_network(self, model_dir):
        """Load and configure Zhang colorization model."""
        model_dir = Path(model_dir)
        
        architecture_file = model_dir / "colorization_deploy_v2.prototxt"
        weights_file = model_dir / "colorization_release_v2.caffemodel"
        cluster_points = model_dir / "pts_in_hull.npy"
        
        print("[INFO] Initializing Zhang colorization network...")
        self.zhang_net = cv2.dnn.readNetFromCaffe(
            str(architecture_file), 
            str(weights_file)
        )
        
        pts = np.load(str(cluster_points))
        
        class8_layer = self.zhang_net.getLayerId("class8_ab")
        conv8_layer = self.zhang_net.getLayerId("conv8_313_rh")
        
        pts_formatted = pts.transpose().reshape(2, 313, 1, 1)
        self.zhang_net.getLayer(class8_layer).blobs = [pts_formatted.astype("float32")]
        self.zhang_net.getLayer(conv8_layer).blobs = [
            np.full([1, 313], 2.606, dtype="float32")
        ]
        
        print("  ✓ Zhang network ready")
    
    def _initialize_segmentation_network(self):
        """Setup semantic segmentation model (YOLOv8 or DeepLabV3+)."""
        print("[INFO] Loading segmentation network...")
        
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.seg_model = YOLO('yolov8x-seg.pt')
                print("  ✓ YOLOv8-X segmentation loaded")
                return
            except ImportError:
                print("[WARNING] YOLOv8 unavailable, defaulting to DeepLabV3+")
                self.use_yolo = False
        
        from torchvision.models.segmentation import deeplabv3_resnet101
        self.seg_model = deeplabv3_resnet101(pretrained=True).to(self.device)
        self.seg_model.eval()
        
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("  ✓ DeepLabV3+ segmentation loaded")
    
    def _configure_color_presets(self):
        """Define preset colors for common object categories."""
        self.color_library = {
            'person': (255, 220, 180),    # Skin tone
            'face': (255, 220, 180),      # Skin tone
            'sky': (135, 206, 250),       # Light blue
            'tree': (34, 139, 34),        # Forest green
            'grass': (124, 252, 0),       # Lawn green
            'car': (169, 169, 169),       # Gray
            'flower': (255, 182, 193),    # Pink
            'water': (0, 191, 255),       # Deep sky blue
            'building': (210, 180, 140),  # Tan
            'road': (105, 105, 105),      # Dim gray
            'default': (200, 200, 200),   # Light gray
        }
        
        # COCO dataset class names (80 categories)
        self.coco_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def execute_segmentation(self, image):
        """
        Perform semantic segmentation to identify discrete objects.
        
        Returns:
            tuple: (masks, class_names, bounding_boxes)
        """
        if self.use_yolo:
            return self._segment_with_yolo(image)
        else:
            return self._segment_with_deeplabv3(image)
    
    def _segment_with_yolo(self, image):
        """Execute YOLOv8 instance segmentation."""
        results = self.seg_model(image, verbose=False)[0]
        
        masks_list = []
        classes_list = []
        boxes_list = []
        
        if results.masks is not None:
            for idx, mask_data in enumerate(results.masks.data):
                mask_array = mask_data.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_array, 
                    (image.shape[1], image.shape[0])
                )
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                class_idx = int(results.boxes.cls[idx])
                class_label = self.coco_categories[class_idx]
                
                bbox = results.boxes.xyxy[idx].cpu().numpy().astype(int)
                
                masks_list.append(mask_binary)
                classes_list.append(class_label)
                boxes_list.append(bbox)
        
        return masks_list, classes_list, boxes_list
    
    def _segment_with_deeplabv3(self, image):
        """Execute DeepLabV3+ semantic segmentation."""
        height, width = image.shape[:2]
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.seg_transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.seg_model(input_tensor)['out'][0]
        
        segmentation_map = output.argmax(0).cpu().numpy()
        segmentation_map = cv2.resize(
            segmentation_map.astype(np.uint8), 
            (width, height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        pascal_voc_labels = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        masks_list = []
        classes_list = []
        boxes_list = []
        
        for class_id in range(1, 21):
            class_mask = (segmentation_map == class_id).astype(np.uint8)
            
            if class_mask.sum() > 100:
                coordinates = np.column_stack(np.where(class_mask > 0))
                if len(coordinates) > 0:
                    y_min, x_min = coordinates.min(axis=0)
                    y_max, x_max = coordinates.max(axis=0)
                    
                    masks_list.append(class_mask)
                    classes_list.append(pascal_voc_labels[class_id])
                    boxes_list.append([x_min, y_min, x_max, y_max])
        
        return masks_list, classes_list, boxes_list
    
    def identify_colorizable_regions(self, ab_channels):
        """
        Core algorithm: Detect regions where Zhang applied strong coloration.
        
        This identifies areas that received meaningful color from the network,
        excluding achromatic regions (glass, metal, white objects).
        
        Parameters:
            ab_channels: a and b channels from Lab color space
            
        Returns:
            Binary mask of colorizable regions
        """
        # Calculate chroma magnitude (color intensity)
        chroma_magnitude = np.sqrt(
            ab_channels[:, :, 0]**2 + ab_channels[:, :, 1]**2
        )
        
        # Threshold to identify colored regions
        colorizable_mask = (chroma_magnitude > self.chroma_threshold).astype(np.uint8)
        
        # Morphological operations to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        colorizable_mask = cv2.morphologyEx(
            colorizable_mask, cv2.MORPH_CLOSE, kernel
        )
        colorizable_mask = cv2.morphologyEx(
            colorizable_mask, cv2.MORPH_OPEN, kernel
        )
        
        return colorizable_mask
    
    def colorize_object_automatic(self, image, mask):
        """
        Apply Zhang colorization to object with region detection.
        
        Returns:
            tuple: (ab_channels, bbox_coords, valid_region_mask) or None
        """
        img_height, img_width = image.shape[:2]
        
        coordinates = np.column_stack(np.where(mask > 0))
        if len(coordinates) == 0:
            return None
        
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        
        # Add padding for context
        pad_amount = 20
        y_min = max(0, y_min - pad_amount)
        x_min = max(0, x_min - pad_amount)
        y_max = min(img_height, y_max + pad_amount)
        x_max = min(img_width, x_max + pad_amount)
        
        object_crop = image[y_min:y_max, x_min:x_max].copy()
        mask_crop = mask[y_min:y_max, x_min:x_max]
        
        # Convert to Lab and apply Zhang colorization
        normalized = object_crop.astype("float32") / 255.0
        lab_space = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        
        crop_height, crop_width = object_crop.shape[:2]
        lab_resized = cv2.resize(lab_space, (224, 224))
        L_channel = cv2.split(lab_resized)[0]
        L_channel -= 50
        
        self.zhang_net.setInput(cv2.dnn.blobFromImage(L_channel))
        ab_predicted = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_predicted = cv2.resize(
            ab_predicted, 
            (crop_width, crop_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Identify colorizable regions
        colorizable_regions = self.identify_colorizable_regions(ab_predicted)
        
        # Combine with original object mask
        valid_region_mask = mask_crop * colorizable_regions
        
        # Apply mask to ab channels
        ab_masked = ab_predicted.copy()
        ab_masked[valid_region_mask == 0] = 0
        
        return ab_masked, (y_min, x_min, y_max, x_max), valid_region_mask
    
    def colorize_object_manual(self, image, mask, rgb_color):
        """
        Apply manual color to object (respects colorizable regions only).
        
        Parameters:
            image: Source image
            mask: Object mask
            rgb_color: Desired RGB color tuple (R, G, B)
            
        Returns:
            tuple: (ab_channels, bbox_coords, mask) or None
        """
        img_height, img_width = image.shape[:2]
        
        # Convert RGB to Lab color space
        rgb_array = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb_array[0, 0] = rgb_color
        lab_color = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)[0, 0]
        
        # Extract ab values (centered at 128 in uint8 representation)
        a_target = lab_color[1] - 128
        b_target = lab_color[2] - 128
        
        coordinates = np.column_stack(np.where(mask > 0))
        if len(coordinates) == 0:
            return None
        
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        
        crop_height = y_max - y_min
        crop_width = x_max - x_min
        
        # Create uniform ab channels with target color
        ab_channels = np.zeros((crop_height, crop_width, 2), dtype=np.float32)
        ab_channels[:, :, 0] = a_target
        ab_channels[:, :, 1] = b_target
        
        # Apply object mask
        mask_crop = mask[y_min:y_max, x_min:x_max]
        ab_channels[mask_crop == 0] = 0
        
        return ab_channels, (y_min, x_min, y_max, x_max), mask_crop
    
    def generate_colorizable_visualization(self, image, region_map, class_names):
        """
        Create visual map showing colorizable regions with labels.
        
        Parameters:
            image: Original image
            region_map: Integer map where each value represents an object ID
            class_names: List of class names corresponding to IDs
            
        Returns:
            Visualization image with colored regions and labels
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visualization = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        
        # Color palette for different objects
        color_palette = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 100, 200), (255, 165, 100), (150, 200, 150),
        ]
        
        overlay = visualization.copy()
        
        # Color each detected object region
        for idx in range(1, len(class_names) + 1):
            region_mask = (region_map == idx)
            if region_mask.sum() > 0:
                color = color_palette[(idx - 1) % len(color_palette)]
                overlay[region_mask] = color
        
        # Blend overlay with base visualization
        visualization = cv2.addWeighted(visualization, 0.4, overlay, 0.6, 0)
        
        # Add text labels for each object
        for idx, class_label in enumerate(class_names):
            region_mask = (region_map == idx + 1)
            if region_mask.sum() > 0:
                coordinates = np.column_stack(np.where(region_mask > 0))
                center_y, center_x = coordinates.mean(axis=0).astype(int)
                
                color = color_palette[idx % len(color_palette)]
                label_text = f"{idx}: {class_label}"
                
                # Draw label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    visualization, 
                    (center_x - 5, center_y - text_h - 5), 
                    (center_x + text_w + 5, center_y + 5), 
                    (0, 0, 0), -1
                )
                
                # Draw label text
                cv2.putText(
                    visualization, label_text, (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        return visualization
    
    def process_interactive_colorization(self, image_input, custom_color_map=None, 
                                        generate_visualization=False):
        """
        Execute complete interactive colorization pipeline.
        
        Parameters:
            image_input: Image path or numpy array
            custom_color_map: Dict mapping object indices to RGB colors
            generate_visualization: Whether to create colorizable region map
            
        Returns:
            tuple: (colorized_image, masks, classes, region_map, visualization)
        """
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
        else:
            image = image_input
        
        img_height, img_width = image.shape[:2]
        print(f"\n[INFO] Processing image: {img_width}x{img_height}")
        
        print("[INFO] Detecting objects via segmentation...")
        object_masks, object_classes, bounding_boxes = self.execute_segmentation(image)
        
        if len(object_masks) == 0:
            print("[WARNING] No objects detected in image")
            return None
        
        unique_classes = ', '.join(set(object_classes))
        print(f"[INFO] Detected {len(object_masks)} objects: {unique_classes}")
        
        # Prepare Lab color space
        normalized = image.astype("float32") / 255.0
        lab_full = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        L_channel_full = cv2.split(lab_full)[0]
        
        # Initialize output channels
        ab_composite = np.zeros((img_height, img_width, 2), dtype=np.float32)
        coverage_counter = np.zeros((img_height, img_width), dtype=np.float32)
        colorizable_region_map = np.zeros((img_height, img_width), dtype=np.uint8)
        
        print("\n[INFO] Colorizing objects individually...")
        
        for idx, (mask, class_name) in enumerate(zip(object_masks, object_classes)):
            progress_indicator = f"[{idx + 1}/{len(object_masks)}] {class_name}..."
            print(f"  {progress_indicator}", end='')
            
            # Check for custom color override
            if custom_color_map and idx in custom_color_map:
                target_color = custom_color_map[idx]
                result = self.colorize_object_manual(image, mask, target_color)
                print(f" [Manual: RGB{target_color}]")
            else:
                result = self.colorize_object_automatic(image, mask)
                print(f" [Automatic: Zhang]")
            
            if result is not None:
                ab_object, (y1, x1, y2, x2), object_mask = result
                
                # Accumulate colorization
                ab_composite[y1:y2, x1:x2] += ab_object
                coverage_counter[y1:y2, x1:x2] += object_mask.astype(np.float32)
                colorizable_region_map[y1:y2, x1:x2][object_mask > 0] = idx + 1
        
        # Handle overlapping regions (average colors)
        overlap_regions = coverage_counter > 1
        ab_composite[overlap_regions] /= coverage_counter[overlap_regions, np.newaxis]
        
        # Fill uncovered background areas
        uncovered_pixels = coverage_counter == 0
        if uncovered_pixels.sum() > 0:
            lab_resized = cv2.resize(lab_full, (224, 224))
            L_bg = cv2.split(lab_resized)[0]
            L_bg -= 50
            
            self.zhang_net.setInput(cv2.dnn.blobFromImage(L_bg))
            ab_background = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab_background = cv2.resize(
                ab_background, (img_width, img_height), interpolation=cv2.INTER_CUBIC
            )
            ab_composite[uncovered_pixels] = ab_background[uncovered_pixels]
        
        # Reconstruct final image
        colorized_lab = np.concatenate(
            (L_channel_full[:, :, np.newaxis], ab_composite), axis=2
        )
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        colorized_bgr = (255 * colorized_bgr).astype("uint8")
        
        # Generate visualization if requested
        region_visualization = None
        if generate_visualization:
            region_visualization = self.generate_colorizable_visualization(
                image, colorizable_region_map, object_classes
            )
        
        return (colorized_bgr, object_masks, object_classes, 
                colorizable_region_map, region_visualization)


def run_interactive_session():
    """Main interactive menu for user-guided colorization."""
    print("\n" + "=" * 70)
    print(" INTELLIGENT OBJECT RECOLORIZATION SYSTEM")
    print("Only regions colored by Zhang are modifiable")
    print("=" * 70)
    
    # Discover available images
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    available_images = []
    
    for ext in supported_extensions:
        available_images.extend(list(Path('.').glob(f'*{ext}')))
    
    available_images = [
        img for img in available_images 
        if 'colorized' not in str(img)
    ]
    
    if not available_images:
        print("\n[ERROR] No images found in working directory")
        return
    
    print(f"\nDiscovered {len(available_images)} image(s):")
    for idx, img_path in enumerate(available_images, 1):
        print(f"  {idx}. {img_path.name}")
    
    user_choice = input("\nSelect image number (press Enter for first): ").strip()
    selected_image = (
        available_images[0] if not user_choice 
        else available_images[int(user_choice) - 1]
    )
    
    # Configuration options
    yolo_preference = input("Enable YOLOv8 segmentation? (y/n, default=n): ").lower() == 'y'
    
    threshold_input = input("Chroma threshold (3-10, default=5): ").strip()
    chroma_threshold = float(threshold_input) if threshold_input else 5.0
    
    # Initialize colorizer
    colorizer = AdaptiveRecolorizer(
        use_yolo=yolo_preference, 
        chroma_threshold=chroma_threshold
    )
    
    print("\n" + "=" * 70)
    print("PHASE 1: AUTOMATIC COLORIZATION")
    print("=" * 70)
    
    # Execute initial automatic colorization
    result = colorizer.process_interactive_colorization(
        selected_image, 
        generate_visualization=True
    )
    
    if result is None:
        print("[ERROR] Colorization failed")
        return
    
    (auto_colorized, masks, classes, 
     region_map, region_visualization) = result
    
    # Prepare comparison display
    original_image = cv2.imread(str(selected_image))
    grayscale_version = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_version = cv2.cvtColor(grayscale_version, cv2.COLOR_GRAY2BGR)
    
    comparison_panel = np.hstack([
        grayscale_version, 
        region_visualization, 
        auto_colorized
    ])
    
    cv2.imshow("Original | Colorizable Regions | Automatic Result", comparison_panel)
    cv2.waitKey(1)
    
    print("\n" + "=" * 70)
    print("PHASE 2: OBJECT ANALYSIS")
    print("=" * 70)
    print("\nNOTE: Only colored regions can be modified!\n")
    
    # Display colorization statistics
    for idx, class_name in enumerate(classes):
        colorizable_count = (region_map == idx + 1).sum()
        total_count = (masks[idx] > 0).sum()
        
        percentage = (
            (colorizable_count / total_count * 100) 
            if total_count > 0 else 0
        )
        
        print(f"  {idx}. {class_name:15s} - {percentage:5.1f}% colorizable "
              f"({colorizable_count:,} pixels)")
    
    # Offer customization option
    customize_prompt = input("\n Modify object colors? (y/n): ").lower()
    
    if customize_prompt == 'y':
        custom_colors = {}
        
        while True:
            object_id_input = input("\nObject ID to recolor (or 'done'): ").strip()
            
            if object_id_input.lower() == 'done':
                break
            
            try:
                object_id = int(object_id_input)
                
                if 0 <= object_id < len(classes):
                    print(f"\n Recoloring object: {classes[object_id]}")
                    print("Enter RGB values (0-255):")
                    
                    red_value = int(input("  R: "))
                    green_value = int(input("  G: "))
                    blue_value = int(input("  B: "))
                    
                    custom_colors[object_id] = (red_value, green_value, blue_value)
                    print(f" Color set to RGB({red_value}, {green_value}, {blue_value})")
                else:
                    print("Invalid object ID")
                    
            except ValueError:
                print("Invalid input - please enter numeric values")
        
        # Apply custom colors if any were specified
        if custom_colors:
            print("\n[INFO] Applying custom colors...")
            custom_result = colorizer.process_interactive_colorization(
                selected_image, 
                custom_color_map=custom_colors
            )[0]
            
            # Display comparison
            custom_comparison = np.hstack([auto_colorized, custom_result])
            cv2.imshow("Automatic vs Custom Colorization", custom_comparison)
            cv2.waitKey(1)
            
            # Save custom result
            output_directory = Path("colorized_output")
            output_directory.mkdir(exist_ok=True)
            
            custom_filename = output_directory / f"custom_{selected_image.name}"
            cv2.imwrite(str(custom_filename), custom_result)
            print(f" Saved custom colorization: {custom_filename}")
    
    # Save automatic results
    output_directory = Path("colorized_output")
    output_directory.mkdir(exist_ok=True)
    
    auto_filename = output_directory / f"auto_{selected_image.name}"
    map_filename = output_directory / f"regions_{selected_image.name}"
    
    cv2.imwrite(str(auto_filename), auto_colorized)
    cv2.imwrite(str(map_filename), region_visualization)
    
    print(f"\n Saved automatic colorization and region map")
    print(f"   Output directory: {output_directory.absolute()}")
    
    print("\n Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_interactive_session()

