import cv2
import time
import numpy as np
import dv_processing as dv
import dv_toolkit as kit
from datetime import timedelta

import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from numpy.lib.recfunctions import structured_to_unstructured


def _visualize_events(events, size, mode="accumulate"):
    counts = np.zeros(size)
    _bins  = [size[0], size[1]]
    _range = [[0, size[0]], [0, size[1]]]

    # w/ polar
    if mode == 'polar':
        counts = np.histogram2d(events.ys(), events.xs(), 
                                weights=(-1) ** (1 + events.polarities()), 
                                bins=_bins, range=_range)[0]
        return counts
    
    # w/o polar
    elif mode == 'monopolar':
        counts = np.histogram2d(events.ys(), events.xs(), 
                                weights=(+1) ** (1 + events.polarities()), 
                                bins=_bins, range=_range)[0]
        return counts
    
    # count before polar assignment
    elif mode == 'accumulate':
        counts = np.histogram2d(events.ys(), events.xs(), 
                                weights=(+1) ** (1 + events.polarities()), 
                                bins=_bins, range=_range)[0]
        weight = np.zeros(size)
        weight[events.ys(), events.xs()] = (-1) ** (1 + events.polarities())
        return counts * weight

    return counts

# Initialize reader
reader = kit.io.MonoCameraReader("./data/DvFire/aedat/s01_v01_c001.aedat4")

# Get offline MonoCameraData
data = reader.loadData()

# Initialize slicer, it will have no jobs at this time
slicer = kit.MonoCameraSlicer()

points = structured_to_unstructured(data.events().numpy())

# tree = cKDTree(points[:, 1:])
# for i in range(len(points)):
#     indices = tree.query_ball_point(points[i, 1:], r=0, p=2.0, return_sorted=True)
#     count = len(indices) - 1

# Initialize an accumulator with some resolution
accumulator = dv.Accumulator(reader.getResolution("events"))
# Apply configuration, these values can be modified to taste
accumulator.setMinPotential(0)
accumulator.setMaxPotential(np.inf)
accumulator.setEventContribution(1.0)
accumulator.setIgnorePolarity(True)
accumulator.setDecayFunction(dv.Accumulator.Decay.NONE)

accumulator.accept(data["events"].toEventStore())
count = accumulator.getPotentialSurface()
print(np.unique(count))

# Conduct statistics
# TODO: 
# [] 统计事件输出率
# [] 计算光流
def run_statistics(data):
    accumulator.clear()
    accumulator.accept(data["events"].toEventStore())
    count = accumulator.getPotentialSurface()
    # frame = accumulator.generateFrame()
    print(np.unique(count).astype(np.int64))
    plt.imshow(count, vmin=0, vmax=count.max())
    plt.show()
    # cv2.imshow("name", frame.image)
    # cv2.waitKey(0)

# Register this method to be called every 33 millisecond of events
slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), run_statistics)

# Now push the store into the slicer, the data contents within the store
# can be arbitrary, the slicer implementation takes care of correct slicing
# algorithm and calls the previously registered callbacks accordingly.
slicer.accept(data)

# Send MonoCameraData to player, modes include:
# hybrid, 3d, 2d
player = kit.plot.OfflineMonoCameraPlayer(reader.getResolution("events"))

# View every 33 millisecond of events
player.viewPerTimeInterval(data, "events", timedelta(milliseconds=33))
