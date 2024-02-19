import os
import dv_toolkit as kit
import xml.etree.ElementTree as ET

from numpy.lib.recfunctions import structured_to_unstructured

import torch
from torch.utils.data import Dataset

# TODO add some data augmentation methods


class DvFire(Dataset):
    def __init__(self, dataset_path: str, partition: str):
        super().__init__()
        # support type is 'test' and 'train'
        assert partition in ['train', 'test']

        # define root directory
        self.dataset_path = dataset_path

        # parse xml dataset
        xml_file = ET.parse(os.path.join(self.dataset_path, f"shuffled_{partition}.xml"))

        # mapping label to index
        labels = xml_file.getroot().find('labels')
        self.label_dict = {label.get('name'): idx for idx, label in enumerate(labels.findall('label'))}

        # store elements
        annotations = xml_file.getroot().find('annotations')
        self.elements = [element for element in annotations.findall('image')]

    def __getitem__(self, index):
        # get item path
        element = self.elements[index]
        aedat_file = os.path.join(self.dataset_path, f"data/{element.get('name')}")

        # load aedat4 data
        reader = kit.io.MonoCameraReader(aedat_file)
        data = reader.loadData()
        width, height = reader.getResolution("events")

        # parse samples
        sample = {
            'events': self._to_tensor_event(data['events']),
            'frames': self._to_tensor_frame(data['frames']),
        }
        
        # parse targets
        targets = {
            'labels': torch.tensor([self.label_dict[elem.get('label')]
                                   for elem in element.findall('box')]),
            'boxes': torch.tensor([[float(elem.get('xtl')) / width,
                                    float(elem.get('ytl')) / height,
                                    float(elem.get('xbr')) / width,
                                    float(elem.get('ybr')) / height]
                                    for elem in element.findall('box')]),
            'resolution': torch.tensor([width, height])
        }

        return sample, targets

    def __len__(self):
        return len(self.elements)

    def _to_tensor_event(self, events):
        # when empty
        if events.isEmpty():
            return None
        
        # convert to tensor
        return torch.from_numpy(structured_to_unstructured(events.numpy()))

    def _to_tensor_frame(self, frames):
        # when empty
        if frames.isEmpty():
            return None
        
        # convert to tensor
        return torch.from_numpy(frames.front().image)
