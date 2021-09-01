"""Utility functions to construct a model."""

import torch
from torch import nn

from extensions import data_parallel
from extensions import model_refinery_wrapper
from extensions import refinery_loss
from models.efficientnet import efficientnet_b0, efficientnet_b3, Margloss
from models.efficientnet_ex import efficientnet_ex, efficientnet_exx

MODEL_NAME_MAP = {
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_ex': efficientnet_ex,
    'efficientnet_exx': efficientnet_exx,

}


def _create_single_cpu_model(model_name, state_file=None, cosln=False):
    if model_name not in MODEL_NAME_MAP:
        raise ValueError("Model {} is invalid. Pick from {}.".format(
            model_name, sorted(MODEL_NAME_MAP.keys())))
    model_class = MODEL_NAME_MAP[model_name]
    model = model_class(num_classes=100, coslinear=cosln)
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
    return model


def create_model(model_name, model_state_file=None, gpus=[0], label_refinery=None,
                 label_refinery_state_file=None, coslinear=True, scale=5.0):
    model = _create_single_cpu_model(model_name, model_state_file, coslinear)
    if label_refinery is not None:
        assert label_refinery_state_file is not None, "Refinery state is None."
        label_refinery = _create_single_cpu_model(
            label_refinery, label_refinery_state_file, coslinear)
        model = model_refinery_wrapper.ModelRefineryWrapper(model, label_refinery, scale)
        loss = refinery_loss.RefineryLoss(cosln=coslinear, scl=scale)
    else:
        if coslinear:
            print('Using other loss')
            loss = Margloss(s=scale)
        else:
            print('Using CrossEntropyLoss')
            # loss = F.cross_entropy
            loss = nn.CrossEntropyLoss()

    if len(gpus) > 0:
        model = model.cuda()
        loss = loss.cuda()
    if len(gpus) > 1:
        model = data_parallel.DataParallel(model, device_ids=gpus)
    return model, loss
