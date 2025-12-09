#!/usr/bin/env python3
"""Move some stove and knife samples to validation set"""
import os
import shutil
import glob

base = '/home/pranay/projects/ECE_381_final_project/data/yolo_detection'

def find_files_with_class(labels_dir, class_id):
    """Find label files containing a specific class"""
    files = []
    for f in glob.glob(os.path.join(labels_dir, '*.txt')):
        with open(f) as fp:
            for line in fp:
                if line.startswith(f'{class_id} '):
                    files.append(f)
                    break
    return files

def move_to_val(label_files, n):
    """Move n files from train to val"""
    moved = 0
    for lbl in label_files[:n]:
        if not os.path.exists(lbl):
            continue
        base_name = os.path.splitext(os.path.basename(lbl))[0]

        # Move label
        dst_lbl = os.path.join(base, 'val', 'labels', os.path.basename(lbl))
        try:
            shutil.move(lbl, dst_lbl)
        except:
            continue

        # Move image (try different extensions)
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            src_img = os.path.join(base, 'train', 'images', base_name + ext)
            if os.path.exists(src_img):
                dst_img = os.path.join(base, 'val', 'images', base_name + ext)
                shutil.move(src_img, dst_img)
                moved += 1
                break
    return moved

# Find files with Stove (class 3) and knife (class 4)
train_labels = os.path.join(base, 'train', 'labels')

stove_files = find_files_with_class(train_labels, 3)
knife_files = find_files_with_class(train_labels, 4)

print(f"Found {len(stove_files)} files with Stove in train")
print(f"Found {len(knife_files)} files with knife in train")

# Move ~20% to val
moved_stove = move_to_val(stove_files, 15)
moved_knife = move_to_val(knife_files, 20)

print(f"\nMoved {moved_stove} Stove files to val")
print(f"Moved {moved_knife} knife files to val")

# Clear cache files
for cache in glob.glob(os.path.join(base, '**', '*.cache'), recursive=True):
    os.remove(cache)
    print(f"Removed cache: {cache}")

print("\nDone! Ready to retrain.")
