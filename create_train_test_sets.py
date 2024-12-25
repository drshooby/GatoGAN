import os
import random
import shutil

'''
Splits the dataset into training and test sets.
    - Assumes there is a main dataset folder called 'images' in the current directory.
    - The network dataloader expects the data to be in nested class folders. This will have to be done manually, or you can update the function.
    - So the dataset should be organized as follows:
        * train/*classname*/image.jpg
        * test/*classname*/image.jpg
'''

def split_dataset():
    image_folder = 'archive' # update to wherever images are being pulled from
    train_folder = 'train'
    test_folder = 'test'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    image_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))

    random.shuffle(image_files)

    # 80/20 split
    train_size = int(0.8 * len(image_files))
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]

    for train_file in train_files:
        shutil.move(train_file, os.path.join(train_folder, os.path.basename(train_file)))

    for test_file in test_files:
        shutil.move(test_file, os.path.join(test_folder, os.path.basename(test_file)))

    print(
        f"Data split complete. {len(train_files)} images in the training set, {len(test_files)} images in the test set.")