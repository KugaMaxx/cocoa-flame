from .point_mlp import PointMLP, DETRLoss


def build_model(args):
    candidate_model_list = [
        "point_mlp"
    ]
    
    assert args.model_name in candidate_model_list, \
        f"Please select one of the following models: {candidate_model_list}"

    if args.model_name == "point_mlp":
        model = PointMLP()
        criterion = DETRLoss()
        # pre_processor = CocoaSampler()
        # post_processor = None

    return model, criterion
