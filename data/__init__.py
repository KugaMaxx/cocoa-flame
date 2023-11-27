from .dataset import DvFire

def Datasets(name: str, **kwargs):
    # mapping
    datasets = dict(
        dvfire = DvFire
    )

    # check
    name = name.lower()
    assert name in datasets.keys()

    return datasets[name](**kwargs)
