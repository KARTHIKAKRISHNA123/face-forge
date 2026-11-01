import os

from PIL import Image
from torch.utils.data import Dataset

# Why this lives in a .py file and not in the notebook:
# On Windows, DataLoader workers (num_workers > 0) are started as fresh Python
# processes. They can import classes from a .py file, but they cannot see a
# class that was defined inside a notebook cell -> "Can't get attribute
# 'ImageProcessor' on <module '__main__'>". Keeping it here lets us use
# several workers to decode JPEGs in parallel, which is the slow part.

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageProcessor(Dataset):
    def __init__(self, root_dir_path, transformations=None):
        self.root_dir_path = root_dir_path
        self.transformations = transformations

        # Only keep image files (Windows can drop Thumbs.db / desktop.ini into folders),
        # sorted so the order is the same on every run.
        self.all_img_paths = [
            os.path.join(root_dir_path, img)
            for img in sorted(os.listdir(root_dir_path))
            if img.lower().endswith(IMAGE_EXTENSIONS)
        ]

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        img_path = self.all_img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transformations:
            img = self.transformations(img)

        return img