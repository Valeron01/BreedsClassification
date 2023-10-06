import albumentations as al


def get_augmentations():
    return al.Sequential([
        al.ShiftScaleRotate(),
        al.Perspective(),
        al.RandomResizedCrop(224, 224, (0.4, 1)),
        al.HorizontalFlip(),
        al.RandomGamma(),
        al.RandomBrightnessContrast(),
    ])

