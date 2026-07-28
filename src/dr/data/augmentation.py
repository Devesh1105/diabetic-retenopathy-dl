"""Albumentations pipelines used for class-specific augmentation.

Minority classes (Severe, Proliferate_DR in the source dataset) receive the
``intensive`` pipeline; the remaining classes receive ``moderate``. Validation
and test images are only normalised.

The transforms target the albumentations 1.4 API. Where 1.4 renamed an
argument the modern spelling is used, so the pipelines build without
deprecation warnings; see ``_std_range`` for the one conversion that is not a
plain rename.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr.config import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:  # pragma: no cover - typing only
    import albumentations as A


def _require_albumentations():
    try:
        import albumentations as A  # noqa: N812
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The albumentations backend requires 'albumentations' and "
            "'opencv-python'. Install them or set data.augmentation_backend "
            "to 'keras'."
        ) from exc
    return A, cv2


def _std_range(var_limit: tuple[float, float]) -> tuple[float, float]:
    """Convert a legacy ``GaussNoise`` variance range to a std range.

    The old ``var_limit`` was a variance in 0-255 units; ``std_range`` is a
    standard deviation expressed as a fraction of 255.
    """
    low, high = var_limit
    return (low**0.5 / 255.0, high**0.5 / 255.0)


def normalisation_only() -> "A.Compose":
    """Transform for validation/test data: ImageNet normalisation, no changes."""
    A, _ = _require_albumentations()
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def moderate_augmentation() -> "A.Compose":
    """Light augmentation for well-represented classes."""
    A, cv2 = _require_albumentations()
    return A.Compose(
        [
            A.Rotate(limit=30, p=0.6, border_mode=cv2.BORDER_REFLECT),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.85, 1.15),
                rotate=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=1,
                    ),
                ],
                p=0.6,
            ),
            A.GaussNoise(std_range=_std_range((5.0, 20.0)), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def intensive_augmentation() -> "A.Compose":
    """Aggressive augmentation for minority classes.

    Geometric, photometric, blur/noise and occlusion transforms are stacked so
    that the rare severity grades are seen under far more variation than the
    majority classes.
    """
    A, cv2 = _require_albumentations()
    return A.Compose(
        [
            # Geometry
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.9, border_mode=cv2.BORDER_REFLECT),
            A.Affine(
                translate_percent=(-0.25, 0.25),
                scale=(0.65, 1.35),
                rotate=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.Transpose(p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(p=0.4),
            A.OpticalDistortion(distort_limit=0.4, p=0.4),
            A.Perspective(scale=(0.05, 0.15), p=0.4),
            # Colour / intensity
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.4, contrast_limit=0.4, p=1
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=30,
                        sat_shift_limit=40,
                        val_shift_limit=40,
                        p=1,
                    ),
                    A.RGBShift(
                        r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1
                    ),
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1
                    ),
                ],
                p=0.9,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomGamma(gamma_limit=(60, 140), p=0.4),
            A.ChannelShuffle(p=0.3),
            A.ToGray(p=0.1),
            A.InvertImg(p=0.05),
            # Blur / noise / quality degradation
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 9), p=1),
                    A.MedianBlur(blur_limit=7, p=1),
                    A.MotionBlur(blur_limit=9, p=1),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.5), p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=_std_range((10.0, 70.0)), p=1),
                    A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.7), p=1),
                    A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1),
                ],
                p=0.6,
            ),
            A.ImageCompression(quality_range=(50, 100), p=0.4),
            A.Downscale(scale_range=(0.6, 0.9), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            # Occlusion
            A.CoarseDropout(
                num_holes_range=(5, 12),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=0,
                p=0.6,
            ),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
