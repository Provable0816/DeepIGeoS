import torchio as tio
import numpy as np


def get_transform(split):
    if split == "train":
        transform = [
            tio.ToCanonical(),
            tio.Resample(1),
            # 如果是 unet 需要将crop大小设置为一个合适的大小，以便 unet 下采样后能恢复到原始crop的大小
            tio.CropOrPad(mask_name='crop_mask'), # crop only object region
            tio.RandomAffine(degrees=[-np.pi/8, np.pi/8], scales=[0.8, 1.25]),
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.RemapLabels({2:1, 3:1, 4:1}),
        ]
    elif split == "valid":
        transform = [
            tio.ToCanonical(), 
            tio.Resample(1),
            # 如果是 unet 需要将crop大小设置为一个合适的大小，以便 unet 下采样后能恢复到原始crop的大小
            tio.CropOrPad(mask_name='crop_mask'), # crop only object region
            tio.RemapLabels({2:1, 3:1, 4:1}),
        ]
    else:
        raise ValueError(f"split {split} is not supported")

    return tio.Compose(transform)
