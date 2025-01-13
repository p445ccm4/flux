from diffusers.utils import load_image
from PIL import ImageOps, ImageFilter, Image
import largestinteriorrectangle as lir
import numpy as np
import cv2

image_path = "../ViewCrafter/outputs/panorama/d5.png"
image = load_image(image_path)
w, h = image.size

# crop the largest interior rectangle with no black pixels
# convert the image to boolean array with only black pixels as False

binary_lir = cv2.inRange(np.array(image), np.array([1, 1, 1]), np.array([255, 255, 255])).astype(bool)
lir_xywh = lir.lir(binary_lir)
pt1, pt2 = lir.pt1(lir_xywh), lir.pt2(lir_xywh)
image = image.crop((0, pt1[1], w, pt2[1]))
image.save(image_path.replace(".png", "_lir_cropped_image.png"))

# Extend and pad the image until the width-to-height ratio is 2:1
w, h = image.size
if w > h * 2:
    h = w // 2
else:
    w = h * 2
# w, h = int(w // 2), int(h // 2)
w, h = int(w), int(h)

pad_w, pad_h = (w - image.size[0])//2, (h - image.size[1])//2
image = np.array(image)
padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='mean')

padded_image = Image.fromarray(padded_image)
padded_image.save(image_path.replace(".png", f"_lir_padded_image.png"))