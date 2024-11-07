import os

def calculate_IW_numbers(IW_dir):
    mask_count = 0
    image_count = 0

    target_dir = os.path.join(IW_dir)
    all_files = os.listdir(IW_dir)
    