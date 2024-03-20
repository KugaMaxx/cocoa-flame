import cv2
import torch
import numpy as np
import dv_processing as dv
import dv_toolkit as kit
from datetime import timedelta

from models.scout import flame_scout
from numpy.lib.recfunctions import structured_to_unstructured
from utils.plot import plot_detection_result, plot_projected_events


i = 0
def plot(data):
    global i
    if data['events'].isEmpty():
        return
    sample = {
        "events": torch.from_numpy(structured_to_unstructured(data['events'].numpy())) if not data['events'].isEmpty() else None,
        "frames": torch.from_numpy(data['frames'].front().image) if not data['frames'].isEmpty() else None,
    }

    model = flame_scout.init((346, 260))
    model.accept(data['events'])

    events = structured_to_unstructured(data['events'].numpy())
    frames = data['frames'].front().image if not data['frames'].isEmpty() else np.full((260, 346), 255).astype(np.uint8)
    bboxes = model.detect()

    image = plot_projected_events(frames, events)
    image = plot_detection_result(image, bboxes=bboxes, colors=[(255, 0, 0)])
    cv2.imwrite(f'./detects/detect_{i}.png', image)
    i = i + 1

# load offline data
reader = kit.io.MonoCameraReader(f"./tmp/Hybrid_02.aedat4")
data, resolution = reader.loadData(), reader.getResolution("events")

# do every 33ms (cannot modify!)
slicer = kit.MonoCameraSlicer()
slicer.doEveryTimeInterval("events", timedelta(milliseconds=11), plot)
slicer.accept(data)
