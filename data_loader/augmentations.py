import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------------
# TRAIN AUGMENTATIONS
# ----------------------------------
train_transforms = A.Compose([

    # ----------------------------------
    # GEOMETRIC TRANSFORMS
    # ----------------------------------
    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.2),

    A.Rotate(
        limit=20,
        p=0.5
    ),

    # ----------------------------------
    # INTENSITY TRANSFORMS
    # ----------------------------------
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    # ----------------------------------
    # DISTORTIONS
    # ----------------------------------
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        p=0.3
    ),

    A.GridDistortion(
        p=0.3
    ),

    # ----------------------------------
    # NOISE
    # ----------------------------------
    A.GaussNoise(
        p=0.2
    ),

])

# ----------------------------------
# VALIDATION TRANSFORMS
# ----------------------------------
val_transforms = A.Compose([

    # No augmentation
    # Only normalization handled earlier

])