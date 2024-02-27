import cv2
import numpy as np

from typing import List, Dict
# import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont, ImageTk


def plot_projected_events(events, bkg_image):
    _, x, y, polarity = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    pos_ind = polarity == 1
    neg_ind = polarity == 0

    if bkg_image.ndim == 2:
        bkg_image = np.array(Image.fromarray(bkg_image).convert('RGB'))

    bkg_image[y[pos_ind], x[pos_ind]] = [255, 0, 0]
    bkg_image[y[neg_ind], x[neg_ind]] = [0, 255, 0]

    return bkg_image


def plot_detection_result(image, cat_ids: List, bboxes: List, 
                          probas: List = None, labels: List = None, 
                          colors: List = [(0, 115, 190), (218, 83, 25), (238, 178, 32), 
                                          (126, 48, 143), (119, 173, 49), (78, 191, 239)]):
    """
    https://github.com/trsvchn/coco-viewer/blob/main/cocoviewer.py
    """
    # pre-process
    colors = colors * 100

    # rescale
    height, width, _ = image.shape
    bboxes = [
        [
            x * width, 
            y * height,
            x * width + w * width, 
            y * height + h * height
        ] 
        for (x, y, w, h) in bboxes
    ]

    # create image
    image = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # display rectangular
    for k, (tl_x, tl_y, rb_x, rb_y) in enumerate(bboxes):
        label = f"{labels[cat_ids[k]]}" if labels else f""
        proba = f"{probas[cat_ids[k]]}" if probas else f""

        draw.rectangle((tl_x, tl_y, rb_x, rb_y), width=1, outline=colors[k])

        # Draw text on the image
        text = f"{label}{proba}"
        text_bbox = draw.textbbox((tl_x, tl_y), text, font=font)
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 100))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(text_bbox, fill=colors[k] + (128,))
        draw_overlay.text((tl_x, tl_y), text, fill=(255, 255, 255), font=font)
        
        # Composite the image and the overlay
        image = Image.alpha_composite(image, overlay)

    return np.array(image)
