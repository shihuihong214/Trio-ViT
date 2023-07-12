from typing import Optional, Dict

from EfficientViT.models.utils import load_state_dict_from_file
from EfficientViT.models.efficientvit import EfficientViTCls

__all__ = ["create_cls_model"]


REGISTERED_CLS_MODEL: Dict[str, str] = {
    ###############################################################################
    "b1-r224": "EfficientViT/checkpoints/cls/b1-r224.pt",
    "b1-r256": "EfficientViT/checkpoints/cls/b1-r256.pt",
    "b1-r288": "EfficientViT/checkpoints/cls/b1-r288.pt",
    ###############################################################################
    "b2-r224": "EfficientViT/checkpoints/cls/b2-r224.pt",
    "b2-r256": "EfficientViT/checkpoints/cls/b2-r256.pt",
    "b2-r288": "EfficientViT/checkpoints/cls/b2-r288.pt",
    ###############################################################################
    "b3-r224": "EfficientViT/checkpoints/cls/b3-r224.pt",
    "b3-r256": "EfficientViT/checkpoints/cls/b3-r256.pt",
    "b3-r288": "EfficientViT/checkpoints/cls/b3-r288.pt",
    ###############################################################################
}

def create_cls_model(name: str, pretrained=True, weight_url: Optional[str] = None, **kwargs) -> EfficientViTCls:
    from EfficientViT.models.efficientvit import efficientvit_cls_b1, efficientvit_cls_b2, efficientvit_cls_b3
    model_dict = {
        "b1": efficientvit_cls_b1,
        "b2": efficientvit_cls_b2,
        "b3": efficientvit_cls_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    
    if pretrained:
        weight_url = weight_url or REGISTERED_CLS_MODEL.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
