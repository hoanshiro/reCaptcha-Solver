import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SmallNoiseImageAugmentation:
    def __init__(self):
        # Define a combination of augmentations
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.5),
                        A.MedianBlur(blur_limit=3, p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.RandomGamma(gamma_limit=(0.8, 1.2), p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    ],
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomRain(p=0.5),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.5),
                    ],
                    p=0.5,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        # convert PIL image to numpy array
        image = np.array(image)
        augmented = self.transform(image=image)
        return augmented["image"]
