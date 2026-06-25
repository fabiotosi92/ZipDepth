import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from typing import Tuple, Optional
import cv2
cv2.setNumThreads(0)


class AlbumentationsWrapper:
    """Wrapper: transform(image, depth) -> (image, depth)"""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if image.dtype not in [np.uint8, np.float32]:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze()
            h, w = image.shape[:2]
            dh, dw = depth.shape[:2]
            if (dh, dw) != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            result = self.transform(image=image, mask=depth)
            return result['image'], result['mask']
        else:
            result = self.transform(image=image)
            return result['image'], None

    def __repr__(self):
        return repr(self.transform)


def get_train_transforms(height: int = 512, width: int = 512) -> AlbumentationsWrapper:
    """
    Training transforms: random crop/resize + horizontal flip + mild color jitter.
    No aggressive augmentation — with 14M+ training images, variance comes from data diversity.
    """
    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.Compose([
                        A.SmallestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR),
                        A.RandomCrop(height=height, width=width),
                    ]),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.05),
            A.ToGray(num_output_channels=3, p=0.05),
        ],
        additional_targets={'mask': 'mask'},
    )
    return AlbumentationsWrapper(transforms)
