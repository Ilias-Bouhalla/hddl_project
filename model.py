"""
ce script utilise timm pour charger les modèles que l'on va teste:

 - EfficientNetV2S
 - ConvNeXtTiny
 - ResNet-50

"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

MODEL_ALIASES: dict[str, str] = {
    "efficientnet": "tf_efficientnetv2_s",
    "efficientnetv2s": "tf_efficientnetv2_s",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "tf_efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext": "convnext_tiny",
    "convnexttiny": "convnext_tiny",
    "convnext_tiny": "convnext_tiny",
    "resnet": "resnet50",
    "resnet50": "resnet50",
}


def resolve_model_name(model_name: str) -> str:
    """Resolve a short alias (e.g. 'resnet', 'convnext', 'efficientnet') to a timm model name."""
    normalized = model_name.strip().lower().replace("-", "").replace(" ", "")
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]

    normalized_underscore = model_name.strip().lower().replace("-", "_")
    if normalized_underscore in MODEL_ALIASES:
        return MODEL_ALIASES[normalized_underscore]

    if timm.is_model(model_name):
        return model_name

    supported = ", ".join(sorted({"efficientnet", "convnext", "resnet"}))
    raise ValueError(
        f"Modèle inconnu : '{model_name}'. Alias supportés : {supported} "
        "(ou n'importe quel nom de modèle timm valide)."
    )


def get_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int | None = None,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> nn.Module:
    """Load a classification model via timm from a string name or alias.

    Args:
        model_name: Alias ('efficientnet', 'convnext', 'resnet', 'ResNet-50', etc.)
            ou nom exact d'un modèle timm.
        pretrained: Si True, charge les poids pré-entraînés ImageNet.
        num_classes: Nombre de classes en sortie (None = tête ImageNet-1K par défaut).
        device: Périphérique cible ('cuda', 'cpu', ou None pour auto-détection).
        eval_mode: Si True, passe le modèle en mode évaluation (.eval()).
    """
    timm_name = resolve_model_name(model_name)

    create_kwargs: dict[str, object] = {"pretrained": pretrained}
    if num_classes is not None:
        create_kwargs["num_classes"] = num_classes

    model = timm.create_model(timm_name, **create_kwargs)

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(target_device)

    if eval_mode:
        model.eval()

    return model


def get_model_transforms(model: nn.Module, is_training: bool = False):
    """Return the exact torchvision/timm preprocessing transform for a given model."""
    data_config = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**data_config, is_training=is_training)
