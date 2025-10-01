import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CoarseDropout(p=0.5),
        A.Normalize(mean=(0.4741, 0.4727, 0.4733), std=(0.2521, 0.2520, 0.2506)),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.Normalize(mean=(0.4741, 0.4727, 0.4733), std=(0.2521, 0.2520, 0.2506)),
        ToTensorV2(),
    ])


