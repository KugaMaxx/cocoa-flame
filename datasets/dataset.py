from pathlib import Path
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, file_path: str, partition: str) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        assert self.file_path.exists(), \
            f"Dataset path ({file_path}) does not exist."

        self.partition = partition
        assert self.partition in ['train', 'test'], \
            f"Dataset partition ({partition}) does not exit."

        self.cat_ids = dict()
        self.aet_ids = dict()

    def __getitem__(self, index):
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError
