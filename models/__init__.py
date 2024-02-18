# from .point_mlp import PointMLP
# from .functions import MultiCenterLoss, EntropyLoss


# def Models(name: str, **kwargs):
#     # mapping
#     models = dict(
#         pointmlp = PointMLP,
#     )

#     # check
#     name = name.lower()
#     assert name in models.keys()

#     return models[name](**kwargs)


# def Criterions(name: str, **kwargs):
#     # mapping
#     models = dict(
#         multi_center_loss = MultiCenterLoss,
#         entropy_loss = EntropyLoss
#     )

#     # check
#     name = name.lower()
#     assert name in models.keys()

#     return models[name](**kwargs)


def build_model(name: str, args):
    candidate_model_dict = {
        "point_mlp": PointMLP
    }
    
    assert name in candidate_model_dict.keys(), "Please select one of the following models"

    model = None
    criterion = None

    return model, criterion
