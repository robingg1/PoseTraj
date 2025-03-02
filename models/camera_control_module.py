import torch.nn as nn

class Camera_Projection(nn.Module):
    """
    Injected Camera Control Module
    """

    def __init__(
        self,
        in_features,
        pose_dim
        
    ):
        super().__init__()

        cc_projection = nn.Linear(in_features+pose_dim, in_features)
        nn.init.eye_(list(cc_projection.parameters())[0][:in_features, :in_features])
        nn.init.zeros_(list(cc_projection.parameters())[1])

        cc_projection.requires_grad_(True)
    