import numpy as np

from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont


def plot_projected_events(bkg_image, events):
    # convert to pil image
    height, width, *_ = bkg_image.shape
    bkg_image = Image.fromarray(bkg_image).convert('RGBA')

    # classify based on polarity
    _, x, y, polarity = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    pos_ind = polarity == 1
    neg_ind = polarity == 0

    # project events
    evt_image = Image.new('RGBA', (width, height))
    evt_draw = ImageDraw.Draw(evt_image)
    evt_draw.point(list(zip(x[pos_ind], y[pos_ind])), fill=(255, 0, 0, 128))
    evt_draw.point(list(zip(x[neg_ind], y[neg_ind])), fill=(0, 0, 255, 128))

    # hybrid
    pil_image = Image.alpha_composite(bkg_image, evt_image)

    return np.array(pil_image)


def plot_rescaled_image(bkg_image, factor=2):
    # define rescaled function
    from scipy.ndimage import zoom
    rescale = lambda x: zoom(x, zoom=(factor, factor, 1), order=3)

    return rescale(bkg_image)


def plot_detection_result(bkg_image, bboxes: List, labels: List = None, scores: List = None,
                          categories=[f"cat-{i}" for i in range(100)],
                          colors=[(0, 115, 190), (218, 83, 25), (238, 178, 32),
                                  (126, 48, 143), (119, 173, 49), (78, 191, 239)]):
    """
    https://github.com/trsvchn/coco-viewer/blob/main/cocoviewer.py
    """
    # pre-processing
    colors = colors * 100
    height, width, *_ = bkg_image.shape

    # rescale
    bboxes = [
        [
            x * width,
            y * height,
            x * width + w * width,
            y * height + h * height
        ]
        for (x, y, w, h) in bboxes
    ]

    # create PIL image
    pil_image = Image.fromarray(bkg_image).convert("RGBA")
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    # drawing
    for k, (tl_x, tl_y, rb_x, rb_y) in enumerate(bboxes):
        # obtain element
        label = labels[k] if labels is not None else 0
        proba = scores[k] if scores is not None else None
        color = colors[label]
        category = categories[label]

        # draw rectangle
        draw.rectangle((tl_x, tl_y, rb_x, rb_y), width=2, outline=color+(200,))

        # draw text on the image
        text = f"{category}: {proba:.2f}" if proba is not None else f"{category}"
        text_bbox = draw.textbbox((tl_x, tl_y), text, font=font)
        draw.rectangle(text_bbox, fill=color+(200,))
        draw.text((tl_x, tl_y), text, fill=(255, 255, 255), font=font)

    return np.array(pil_image)

    # sample, target, output = samples[0], targets[0], outputs[0]
    # if sample['frames'] is not None:
        # image = plot_projected_events(sample['frames'].numpy(), 
        #                                 sample['events'].numpy())
        
    #     image = plot_rescaled_image(image)
        
    #     image = plot_detection_result(image, 
    #                                     bboxes=(target['bboxes']).tolist(), 
    #                                     labels=(target['labels']).tolist(),
    #                                     colors=[(0, 0, 255)])
        
    #     idn = (output['labels'] != 1)

    #     image = plot_detection_result(image, 
    #                                     bboxes=output['bboxes'][idn],
    #                                     labels=output['labels'][idn],
    #                                     scores=output['scores'][idn],
    #                                     colors=[(255, 0, 0)])

    #     import cv2
    #     cv2.imwrite('./test.png', image)
