import os
import shutil
import random

# set path
train_images_path = r"D:\yolov8_project\datasets\images\train"
train_labels_path = r"D:\yolov8_project\datasets\labels\train"
test_images_path = r"D:\yolov8_project\datasets\images\test"
test_labels_path = r"D:\yolov8_project\datasets\labels\test"

# Create a test set directory if it does not exist
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Gets a list of training set images
image_files = [f for f in os.listdir(train_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 1500 pictures were randomly selected
selected_images = random.sample(image_files, 1500)

# Move the selected image and corresponding label to the test set directory
for image_name in selected_images:
    # move picture
    src_image_path = os.path.join(train_images_path, image_name)
    dst_image_path = os.path.join(test_images_path, image_name)
    shutil.move(src_image_path, dst_image_path)

    # Get the corresponding tag file name (assuming the tag file has the same name as the image file and the extension is.txt)
    label_name = os.path.splitext(image_name)[0] + ".txt"
    src_label_path = os.path.join(train_labels_path, label_name)
    dst_label_path = os.path.join(test_labels_path, label_name)

    # Check whether the label file exists, and move it if it does
    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
    else:
        print(f"Warning: Label file {label_name} not found for image {image_name}")

print("split finished！")