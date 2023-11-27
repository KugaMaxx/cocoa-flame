import os
import glob
import json
import dv_toolkit as kit
import dv_processing as dv
from torch.utils.data import Dataset


# TODO add some data augmentation methods


class DvFire(Dataset):
    def __init__(self, partition: str) -> None:
        """
        将会是.json文件, 每行有文件名字, 以及存在的种类和坐标
        """
        super().__init__()
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_dir = os.path.join(self.root_dir, 'DvFire')
        
        # support type is 'test' and 'train'
        assert partition in ['train', 'test']        
        with open(os.path.join(self.file_dir, f"{partition}.json")) as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        # aedat4 file
        reader = kit.io.MonoCameraReader(self.data['aedat4'][index])

        # load aedat4 data
        aedat4 = reader.loadData()

        # load resolution
        resolution = reader.getResolution("events")

        # load detection
        label = None

        return aedat4, resolution, label

    def __len__(self):
        return len(self.data['aedat4'])
    

if __name__ == '__main__':
    dataset = DvFire(partition='train')
