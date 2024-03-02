import os
import dv_toolkit as kit
import xml.etree.ElementTree as ET

from numpy.lib.recfunctions import structured_to_unstructured

import torch
from torch.utils.data import Dataset

from utils.plot import plot_projected_events, plot_detection_result
from datasets.dataset import DatasetBase

# TODO add some data augmentation methods


class DvFire(DatasetBase):
    def __init__(self, file_path: str, partition: str):
        super().__init__(file_path, partition)
        # parse xml dataset
        xml_file = ET.parse(self.file_path / f"shuffled_{partition}.xml")
        xml_root = xml_file.getroot()

        # mapping category to index
        self.cat_ids.update({label.get('name'): idx for idx, label \
                             in enumerate(xml_root.find('labels').findall('label'))})

        # store elements
        annotations = xml_file.getroot().find('annotations')
        self.elements = [element for element in annotations.findall('image')]

    def __getitem__(self, index):
        # get item path
        element = self.elements[index]
        aedat_file = os.path.join(self.file_path, f"data/{element.get('name')}")

        # load aedat4 data
        reader = kit.io.MonoCameraReader(aedat_file)
        data = reader.loadData()
        width, height = reader.getResolution("events")

        # parse samples
        sample = {
            'events': self._parse_events_to_tensor(data['events']),
            'frames': self._parse_frames_to_tensor(data['frames']),
        }
        
        # parse targets
        targets = {
            'file': aedat_file,
            'labels': torch.tensor([self.cat_ids[elem.get('label')]
                                   for elem in element.findall('box')]),
                                   # todo: 改成 bboxes
            'boxes': torch.tensor([[float(elem.get('xtl')) / width,
                                    float(elem.get('ytl')) / height,
                                    (float(elem.get('xbr')) - float(elem.get('xtl'))) / width,
                                    (float(elem.get('ybr')) - float(elem.get('ytl'))) / height]
                                    for elem in element.findall('box')]),
            'resolution': torch.tensor([width, height])
        }

        # if sample['frames'] is not None:
        #     image = plot_projected_events(sample['events'].numpy(), sample['frames'].numpy())
        #     plot_detection_result(image, targets['labels'].tolist(), targets['boxes'].tolist(), [0.32], ['fire'])

        return sample, targets

    def __len__(self):
        return len(self.elements)

    def _parse_events_to_tensor(self, events):
        # when empty
        if events.isEmpty():
            return None
        # convert to tensor
        return torch.from_numpy(structured_to_unstructured(events.numpy()))

    def _parse_frames_to_tensor(self, frames):
        # when empty
        if frames.isEmpty():
            return None
        # convert to tensor
        return torch.from_numpy(frames.front().image)        
