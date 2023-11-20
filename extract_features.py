import dv_processing as dv
import dv_toolkit as kit
from datetime import timedelta

# Initialize reader
reader = kit.io.MonoCameraReader("./data/DvFire/aedat/s01_v01_c001.aedat4")

# Get offline MonoCameraData
data = reader.loadData()

# Initialize slicer, it will have no jobs at this time
slicer = kit.MonoCameraSlicer()

# Conduct statistics
# TODO: 
# [] 统计事件输出率
# [] 计算光流
def run_statistics(data):
    print(data["events"])

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
