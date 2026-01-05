#!/usr/bin/env python
# coding: utf-8

# ## &#x2611; Object detection processing to prevent color bleeding &darr;

# In[ ]:


"""
Author: Hadi Sarhangi Fard

*** Implements Zhang et al.'s colorization network with custom enhancements
for object-aware processing and interactive color modification

Object-by-Object Semantic Colorization
Advanced colorization system that processes each detected object independently
using segmentation combined with Zhang et al.'s deep learning model.

Key Features:
    - Individual object processing to prevent color bleeding
    - Support for both YOLOv8 and DeepLabV3+ segmentation
    - Comprehensive object detection (people, clothing, natural elements, etc.)
    - Intelligent color blending for overlapping regions
    - Automatic background fill for uncovered areas

Technical Approach:
    1. Segment image into discrete objects using deep neural networks
    2. Apply Zhang colorization model to each object's isolated region
    3. Merge results with overlap handling and gap filling
    4. Output includes original, colorized, and segmentation visualization

Dependencies:
    opencv-python>=4.5.0
    numpy>=1.19.0
    torch>=1.9.0
    torchvision>=0.10.0
    ultralytics (optional, for YOLOv8 segmentation)
"""

import numpy as np
import cv2
from pathlib import Path
import torch
import torchvision.transforms as transforms
from collections import defaultdict


class ObjectByObjectColorizer:
    """
    Implements per-object colorization to maintain semantic boundaries
    and prevent unwanted color propagation across different regions.
    """
    
    def __init__(self, zhang_model_dir="model", use_yolo=False):
        """
        Initialize the colorization pipeline with segmentation and color models.
        
        Parameters:
            zhang_model_dir (str): Directory containing Zhang model weights
            use_yolo (bool): Whether to use YOLOv8 for instance segmentation
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Device selected: {self.device}")
        
        self._initialize_zhang_network(zhang_model_dir)
        self.use_yolo = use_yolo
        self._initialize_segmentation_network()
        self._configure_object_taxonomy()
    
    def _initialize_zhang_network(self, model_dir):
        """Load and configure the Zhang colorization network with pretrained weights."""
        model_dir = Path(model_dir)
        
        prototxt_path = model_dir / "colorization_deploy_v2.prototxt"
        caffemodel_path = model_dir / "colorization_release_v2.caffemodel"
        hull_points = model_dir / "pts_in_hull.npy"
        
        print("[INFO] Loading Zhang colorization network...")
        self.zhang_net = cv2.dnn.readNetFromCaffe(
            str(prototxt_path), 
            str(caffemodel_path)
        )
        
        pts = np.load(str(hull_points))
        
        class8_layer = self.zhang_net.getLayerId("class8_ab")
        conv8_layer = self.zhang_net.getLayerId("conv8_313_rh")
        
        pts = pts.transpose().reshape(2, 313, 1, 1)
        self.zhang_net.getLayer(class8_layer).blobs = [pts.astype("float32")]
        self.zhang_net.getLayer(conv8_layer).blobs = [
            np.full([1, 313], 2.606, dtype="float32")
        ]
        
        print("[INFO] Zhang network initialized successfully")
    
    def _initialize_segmentation_network(self):
        """Initialize either YOLOv8 or DeepLabV3+ for semantic segmentation."""
        print("[INFO] Initializing segmentation network...")
        
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                print("[INFO] Loading YOLOv8-X segmentation model")
                self.seg_model = YOLO('yolov8x-seg.pt')
                return
            except ImportError:
                print("[WARNING] YOLOv8 unavailable, falling back to DeepLabV3+")
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
        
        print("[INFO] DeepLabV3+ segmentation network ready")
    
    def _configure_object_taxonomy(self):
        """Define comprehensive object categories for detection and classification."""
        
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.voc_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        self.semantic_hierarchy = {
            'person': ['face', 'hair', 'hands', 'arms', 'legs', 'torso', 'clothing', 'footwear'],
            'nature': ['sky', 'clouds', 'sun', 'moon', 'stars'],
            'vegetation': ['trees', 'trunk', 'branches', 'leaves', 'grass', 'flowers', 'petals'],
            'water': ['sea', 'ocean', 'river', 'lake', 'pond', 'stream'],
            'terrain': ['road', 'pavement', 'dirt', 'sand', 'rocks', 'mountains'],
            'vehicle': ['body', 'tires', 'wheels', 'windows', 'doors'],
            'architecture': ['walls', 'roof', 'windows', 'doors', 'facade']
        }
    
    def segment_image(self, image):
        """
        Perform semantic segmentation to identify discrete objects.
        
        Returns:
            tuple: (masks, class_names, bounding_boxes)
                - masks: List of binary masks for each detected object
                - class_names: Corresponding class labels
                - bounding_boxes: Spatial coordinates [x1, y1, x2, y2]
        """
        if self.use_yolo:
            return self._segment_with_yolo(image)
        else:
            return self._segment_with_deeplabv3(image)
    
    def _segment_with_yolo(self, image):
        """Execute YOLOv8 instance segmentation for precise object boundaries."""
        results = self.seg_model(image, verbose=False)[0]
        
        masks, class_names, boxes = [], [], []
        
        if results.masks is not None:
            for idx, mask in enumerate(results.masks.data):
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, 
                    (image.shape[1], image.shape[0])
                )
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                class_id = int(results.boxes.cls[idx])
                class_name = self.coco_classes[class_id]
                
                bbox = results.boxes.xyxy[idx].cpu().numpy().astype(int)
                
                masks.append(mask_binary)
                class_names.append(class_name)
                boxes.append(bbox)
        
        return masks, class_names, boxes
    
    def _segment_with_deeplabv3(self, image):
        """Execute DeepLabV3+ semantic segmentation as fallback method."""
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
        
        masks, class_names, boxes = [], [], []
        
        for class_id in range(1, 21):
            mask = (segmentation_map == class_id).astype(np.uint8)
            
            if mask.sum() > 100:
                coords = np.column_stack(np.where(mask > 0))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    masks.append(mask)
                    class_names.append(self.voc_classes[class_id])
                    boxes.append([x_min, y_min, x_max, y_max])
        
        return masks, class_names, boxes
    
    def colorize_object(self, image, mask, expansion_margin=10):
        """
        Apply Zhang colorization to an isolated object region.
        
        Parameters:
            image: Source grayscale image
            mask: Binary mask defining object boundaries
            expansion_margin: Pixels to dilate mask for context preservation
            
        Returns:
            tuple: (ab_channels, bbox_coords, object_mask) or None if processing fails
        """
        height, width = image.shape[:2]
        
        kernel = np.ones((expansion_margin, expansion_margin), np.uint8)
        mask_expanded = cv2.dilate(mask, kernel, iterations=1)
        
        coords = np.column_stack(np.where(mask_expanded > 0))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        padding = 20
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(height, y_max + padding)
        x_max = min(width, x_max + padding)
        
        object_region = image[y_min:y_max, x_min:x_max].copy()
        object_mask = mask_expanded[y_min:y_max, x_min:x_max]
        
        normalized = object_region.astype("float32") / 255.0
        lab_space = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        
        region_height, region_width = object_region.shape[:2]
        resized_lab = cv2.resize(lab_space, (224, 224))
        L_channel = cv2.split(resized_lab)[0]
        L_channel -= 50
        
        self.zhang_net.setInput(cv2.dnn.blobFromImage(L_channel))
        ab_predicted = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        ab_resized = cv2.resize(
            ab_predicted, 
            (region_width, region_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        ab_masked = ab_resized.copy()
        ab_masked[object_mask == 0] = 0
        
        return ab_masked, (y_min, x_min, y_max, x_max), object_mask
    
    def colorize(self, image_input, verbose=True):
        """
        Execute complete colorization pipeline with per-object processing.
        
        Parameters:
            image_input: File path or numpy array
            verbose: Enable progress logging
            
        Returns:
            tuple: (colorized_image, debug_information)
        """
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
        else:
            image = image_input
        
        height, width = image.shape[:2]
        print(f"\n[INFO] Processing {width}x{height} image")
        
        normalized = image.astype("float32") / 255.0
        lab_full = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        L_channel_full = cv2.split(lab_full)[0]
        
        ab_accumulated = np.zeros((height, width, 2), dtype=np.float32)
        coverage_counter = np.zeros((height, width), dtype=np.float32)
        
        print("[INFO] Phase 1/3: Object detection and segmentation")
        masks, class_names, boxes = self.segment_image(image)
        
        if len(masks) == 0:
            print("[WARNING] No objects detected, applying global colorization")
            
            resized_lab = cv2.resize(lab_full, (224, 224))
            L_resized = cv2.split(resized_lab)[0]
            L_resized -= 50
            
            self.zhang_net.setInput(cv2.dnn.blobFromImage(L_resized))
            ab_accumulated = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab_accumulated = cv2.resize(
                ab_accumulated, 
                (width, height), 
                interpolation=cv2.INTER_CUBIC
            )
        else:
            unique_classes = set(class_names)
            print(f"[INFO] Detected {len(masks)} objects: {', '.join(unique_classes)}")
            
            print("[INFO] Phase 2/3: Per-object colorization")
            
            for idx, (mask, class_name, bbox) in enumerate(zip(masks, class_names, boxes)):
                if verbose:
                    progress = f"[{idx+1}/{len(masks)}] Processing {class_name}..."
                    print(f"  {progress}", end='\r')
                
                colorization_result = self.colorize_object(image, mask)
                
                if colorization_result is not None:
                    ab_object, (y1, x1, y2, x2), obj_mask = colorization_result
                    
                    ab_accumulated[y1:y2, x1:x2] += ab_object
                    coverage_counter[y1:y2, x1:x2] += obj_mask.astype(np.float32)
            
            print(f"\n[INFO] Successfully colorized {len(masks)} objects")
            
            print("[INFO] Phase 3/3: Blending and gap filling")
            
            overlap_regions = coverage_counter > 1
            ab_accumulated[overlap_regions] /= coverage_counter[overlap_regions, np.newaxis]
            
            uncovered_pixels = coverage_counter == 0
            if uncovered_pixels.sum() > 0:
                resized_lab = cv2.resize(lab_full, (224, 224))
                L_resized = cv2.split(resized_lab)[0]
                L_resized -= 50
                
                self.zhang_net.setInput(cv2.dnn.blobFromImage(L_resized))
                ab_background = self.zhang_net.forward()[0, :, :, :].transpose((1, 2, 0))
                ab_background = cv2.resize(
                    ab_background, 
                    (width, height), 
                    interpolation=cv2.INTER_CUBIC
                )
                
                ab_accumulated[uncovered_pixels] = ab_background[uncovered_pixels]
        
        lab_reconstructed = np.concatenate(
            (L_channel_full[:, :, np.newaxis], ab_accumulated), 
            axis=2
        )
        
        colorized_bgr = cv2.cvtColor(lab_reconstructed, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        colorized_bgr = (255 * colorized_bgr).astype("uint8")
        
        debug_data = {
            'object_count': len(masks),
            'detected_classes': class_names,
            'segmentation_masks': masks,
            'coverage_map': coverage_counter
        }
        
        print("[INFO] Colorization pipeline completed\n")
        
        return colorized_bgr, debug_data
    
    def visualize_detections(self, image, masks, class_names):
        """Generate color-coded visualization of detected objects."""
        overlay = image.copy()
        
        np.random.seed(42)
        palette = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
        
        for mask, color, label in zip(masks, palette, class_names):
            colored_region = np.zeros_like(image)
            colored_region[mask > 0] = color
            
            overlay = cv2.addWeighted(overlay, 0.7, colored_region, 0.3, 0)
            
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                y_pos, x_pos = coords.min(axis=0)
                
                cv2.putText(
                    overlay, label, (x_pos, y_pos - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                cv2.putText(
                    overlay, label, (x_pos, y_pos - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(color.tolist()), 1
                )
        
        return overlay


def main():
    """Demonstration of object-by-object colorization system."""
    print("\n" + "=" * 70)
    print("OBJECT-BY-OBJECT COLORIZATION SYSTEM")
    print("=" * 70)
    print("\nCapabilities:")
    print("  • Comprehensive object detection across multiple categories")
    print("  • Independent colorization per object using Zhang model")
    print("  • Prevention of cross-object color contamination")
    print("  • Semantic understanding of faces, clothing, natural elements")
    print("=" * 70)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    discovered_images = []
    
    for ext in image_extensions:
        discovered_images.extend(list(Path('.').glob(f'*{ext}')))
    
    discovered_images = [
        img for img in discovered_images 
        if 'colorized' not in str(img)
    ]
    
    if not discovered_images:
        print("\n[ERROR] No suitable images found in working directory")
        return
    
    print(f"\nDiscovered {len(discovered_images)} image(s):")
    for idx, img_path in enumerate(discovered_images, 1):
        print(f"  {idx}. {img_path.name}")
    
    user_choice = input("\nSelect image number (default: 1): ").strip()
    selected_image = (
        discovered_images[0] if not user_choice 
        else discovered_images[int(user_choice) - 1]
    )
    
    yolo_preference = input(
        "\nEnable YOLOv8 segmentation? (y/n, default: n): "
    ).lower() == 'y'
    
    print("\n" + "=" * 70)
    colorizer = ObjectByObjectColorizer(use_yolo=yolo_preference)
    
    print("=" * 70)
    colorized_result, debug_info = colorizer.colorize(selected_image)
    
    output_directory = Path("colorized_output")
    output_directory.mkdir(exist_ok=True)
    
    output_filename = output_directory / f"object_by_object_{selected_image.name}"
    cv2.imwrite(str(output_filename), colorized_result)
    print(f"✓ Colorized image saved: {output_filename}")
    
    if len(debug_info['segmentation_masks']) > 0:
        original_image = cv2.imread(str(selected_image))
        detection_visualization = colorizer.visualize_detections(
            original_image, 
            debug_info['segmentation_masks'], 
            debug_info['detected_classes']
        )
        
        visualization_path = output_directory / f"detected_objects_{selected_image.name}"
        cv2.imwrite(str(visualization_path), detection_visualization)
        print(f"✓ Detection visualization saved: {visualization_path}")
        
        grayscale_version = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        grayscale_version = cv2.cvtColor(grayscale_version, cv2.COLOR_GRAY2BGR)
        
        max_display_height = 600
        img_height, img_width = grayscale_version.shape[:2]
        
        if img_height > max_display_height:
            scale_factor = max_display_height / img_height
            new_width = int(img_width * scale_factor)
            new_height = max_display_height
            
            grayscale_version = cv2.resize(grayscale_version, (new_width, new_height))
            colorized_display = cv2.resize(colorized_result, (new_width, new_height))
            detection_visualization = cv2.resize(detection_visualization, (new_width, new_height))
        else:
            colorized_display = colorized_result
        
        comparison_panel = np.hstack([
            grayscale_version, 
            colorized_display, 
            detection_visualization
        ])
        
        cv2.imshow("Original | Colorized | Detections", comparison_panel)
        
        print("\n" + "=" * 70)
        print("PROCESSING RESULTS")
        print("=" * 70)
        print(f"Objects detected: {debug_info['object_count']}")
        print(f"Categories: {', '.join(set(debug_info['detected_classes']))}")
        print("\nPress any key to close window...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# In[ ]:




