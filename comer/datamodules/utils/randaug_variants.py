from comer.datamodules.utils.randaug_transforms import AutoContrast, Invert, Rotate, Equalize, Posterize, Solarize, SolarizeAdd, \
    Color, Contrast, Brightness, Sharpness, ShearX, ShearY, TranslateXabs, TranslateYabs, TranslateY, TranslateX


def fixmatch_original():
    return [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        # (CutoutAbs, 0, 40),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]


def fixmatch_modified():
    return [
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        # (Color, 0.1, 1.9),
        # (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        # (CutoutAbs, 0, 40),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
    ]