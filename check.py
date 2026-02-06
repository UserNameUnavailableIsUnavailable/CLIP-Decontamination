# for a MMSegmentation-style gray-scale mask, where each class is identified by its pixel value, count the number of classes

import numpy as np
import cv2

def count_classes_in_mask(mask_path):
    # Load the mask image in grayscale mode
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load image at {mask_path}")
    
    # Get unique pixel values in the mask
    unique_classes = np.unique(mask)
    
    # Count the number of unique classes
    num_classes = len(unique_classes)
    
    return num_classes, unique_classes.tolist()

PATH = "/home/daizhengyi/develop/Purification/data/UAVid/annotations/training/seq1_file13-2_0_1080_0_1280.png"

if __name__ == "__main__":
    num_classes, class_values = count_classes_in_mask(PATH)
    print(f"Number of classes in the mask: {num_classes}")
    print(f"Class pixel values: {class_values}")