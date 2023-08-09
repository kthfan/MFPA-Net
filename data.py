import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
import torch

# labelme segmentation
import json
import cv2

class SegmentationDatasetWrapper(torchvision.datasets.VisionDataset):
    def __init__(self, dataset, spatial_transform=None, color_transform=None):
        super().__init__(root='')
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform

    def __getitem__(self, index):
        img, mask = self.dataset[index]

        if self.spatial_transform is not None:
            is_mask_2d = len(mask.shape) == 2
            mask = mask.to(img.dtype)
            if is_mask_2d:
                mask = mask[None]
            join_tensor = torch.cat([img, mask], dim=0)
            join_tensor = self.spatial_transform(join_tensor)
            img, mask = torch.split(join_tensor, [img.shape[0], mask.shape[0]], dim=0)
            mask = mask.round().to(torch.int64)
            # if is_mask_2d:
            #     mask = mask[0]
            mask = mask[0]
        if self.color_transform is not None:
            img = self.color_transform(img)

        return img, mask

    def __len__(self):
        return len(self.dataset)


class LabelmeDataset(torchvision.datasets.VisionDataset):
    def __init__(self, json_roots, image_roots, class_names, transform=None, target_transform=None):
        if isinstance(json_roots, str):
            json_roots = [json_roots]
        if isinstance(image_roots, str):
            image_roots = [image_roots]
        super().__init__(root=image_roots[0], transform=transform, target_transform=target_transform)
        self.json_roots = json_roots
        self.image_roots = image_roots
        self.class_names = np.asarray(class_names)

        self.samples = [os.listdir(path) for path in self.json_roots]
        self.samples = [[os.path.join(root, path) for path in paths] for root, paths in
                        zip(self.json_roots, self.samples)]
        self.num_samples = [len(ls) for ls in self.samples]
        self._cumsum_num_sample = np.cumsum([0] + self.num_samples)
        self.imgs = self.samples

    def __getitem__(self, index):
        group_index = np.argmin(self._cumsum_num_sample <= index) - 1
        index = index - self._cumsum_num_sample[group_index]
        path = self.samples[group_index][index]
        with open(path, "r", encoding="utf-8") as f:  # load json
            config = json.load(f)
        masks = LabelmeDataset.extract_labelme_onehot(config, self.class_names)
        masks = np.argmax(np.pad(masks, [[1, 0], [0, 0], [0, 0]]), axis=0)
        masks = torch.from_numpy(masks)

        img_path = os.path.join(self.image_roots[group_index], config['imagePath'])
        with open(img_path, "rb") as file:
            img = Image.open(file)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            masks = self.target_transform(masks)

        return img, masks

    def __len__(self):
        return sum(self.num_samples)

    @staticmethod
    def extract_labelme_onehot(config, classnames):
        shapes = config['shapes']
        masks = np.zeros((len(classnames), config['imageHeight'], config['imageWidth']), dtype=np.float32)
        for entry in shapes:
            i = np.where(classnames == np.array(entry['label']))[0]
            if len(i) > 0:
                i = i[0]
                pts = np.array(entry['points']).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                masks[i] = cv2.fillPoly(masks[i], [pts], color=1)
        return masks


class BinarySegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(self, image_roots, mask_roots, transform=None, target_transform=None):
        if isinstance(image_roots, str):
            image_roots = [image_roots]
        if isinstance(mask_roots, str):
            mask_roots = [mask_roots]
        super().__init__(root=image_roots[0], transform=transform, target_transform=target_transform)
        self.image_roots = image_roots
        self.mask_roots = mask_roots

        self.image_paths = [os.listdir(path) for path in self.image_roots]
        self.mask_paths = [os.listdir(path) for path in self.mask_roots]
        self.image_paths = [os.path.join(root, path) for root, paths in zip(self.image_roots, self.image_paths) for path
                            in paths]
        self.mask_paths = [os.path.join(root, path) for root, paths in zip(self.mask_roots, self.mask_paths) for path in
                           paths]
        self.samples = list(zip(self.image_paths, self.mask_paths))
        self.imgs = self.samples

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        with open(img_path, "rb") as file:
            img = Image.open(file)
            img = img.convert("RGB")
        with open(mask_path, "rb") as file:
            mask = Image.open(file)
            mask = mask.convert("L")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)

def train_test_split_dataset(ds, train_size=0.7, seed=None):
    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idx, train_size=train_size, random_state=seed, shuffle=True,
                                           stratify=ds.targets)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)
    return train_ds, test_ds
