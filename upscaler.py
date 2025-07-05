import cv2
import numpy as np
import torch
from PIL import Image
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

def upscale_image(image, model_choice="RealESRGAN", scale=4):
    img = np.array(image.convert("RGB"))

    if model_choice == "Anime":
        model_name = "RealESRGAN_x4plus_anime_6B"
    else:
        model_name = "RealESRGAN_x4plus"

    model = RealESRGANer(
        scale=scale,
        model_path=None,
        model_name=model_name,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    if model_choice == "GFPGAN+RealESRGAN":
        gfpgan = GFPGANer(
            model_path=None,
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=model
        )
        _, _, output = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        _, _, output = model.enhance(img, outscale=scale)

    return Image.fromarray(output)