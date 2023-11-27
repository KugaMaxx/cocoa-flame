from .point_mlp import PointMLP, PointMLPElite
from .functions import MultiCenterLoss, EntropyLoss


def Models(name: str, **kwargs):
    # mapping
    models = dict(
        pointmlp = PointMLP,
        pointmlpelite = PointMLPElite
    )

    # check
    name = name.lower()
    assert name in models.keys()

    return models[name](**kwargs)


def Criterions(name: str, **kwargs):
    # mapping
    models = dict(
        multi_center_loss = MultiCenterLoss,
        entropy_loss = EntropyLoss
    )

    # check
    name = name.lower()
    assert name in models.keys()

    return models[name](**kwargs)
