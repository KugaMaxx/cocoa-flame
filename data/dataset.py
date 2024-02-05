import os
import glob
import json
import dv_toolkit as kit
import dv_processing as dv
from torch.utils.data import Dataset

import xml.etree.ElementTree as ET 

# TODO add some data augmentation methods


class DvFire(Dataset):
    def __init__(self, partition: str) -> None:
        super().__init__()
        # support type is 'test' and 'train'
        assert partition in ['train', 'test']

        # define root directory
        self.root_dir = "/home/kuga/Workspace/aedat_to_dataset/"

        # parse xml dataset
        xml_file = ET.parse(os.path.join(self.root_dir, "shuffled_files.xml"))
        xml_selected_set = xml_file.getroot().find(partition)

        # store to elements
        self.elements = [element for element in xml_selected_set.findall('image')]

    def __getitem__(self, index):
        # get item path
        element = self.elements[index]

        # load aedat4 data
        file = os.path.join(self.root_dir, "data", element.get('name'))
        reader = kit.io.MonoCameraReader(file)
        data, resolution = reader.loadData(), reader.getResolution("events")

        # parse bounding box
        target = list()
        for sub_element in element.findall('box'):
            target.append({
                "label": sub_element.get("label"),
                "xtl": float(sub_element.get("xtl")),
                "ytl": float(sub_element.get("ytl")),
                "xbr": float(sub_element.get("xbr")),
                "ybr": float(sub_element.get("ybr"))
            })

        return data, resolution, target

    def __len__(self):
        return len(self.elements)
    

if __name__ == '__main__':
    dataset = DvFire(partition='train')
    data, resolution, target = dataset[0]
