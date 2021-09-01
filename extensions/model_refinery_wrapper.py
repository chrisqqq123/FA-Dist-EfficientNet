from torch import nn
from torch.nn import functional as F
from models.efficientnet import Arctransf

class ModelRefineryWrapper(nn.Module):
    """Convenient wrapper class to train a model with a label refinery."""

    def __init__(self, model, label_refinery, scale):
        super().__init__()
        self.model = model
        self.label_refinery = label_refinery
        self.s = scale
        # Since we don't want to back-prop through the label_refinery network,
        # make the parameters of the teacher network not require gradients. This
        # saves some GPU memory.
        for param in self.label_refinery.parameters():
            param.requires_grad = False

    @property
    def LR_REGIME(self):
        # Training with label refinery does not change learing rate regime.
        # Return's wrapped model lr regime.
        return self.model.LR_REGIME

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, input):
        if self.training:
            refined_labels, refined_feature = self.label_refinery(input)
            refined_labels = Arctransf(refined_labels, self.s) # test
            refined_labels = F.softmax(refined_labels, dim=1) # test

            model_output, inner_feature = self.model(input)
            return (model_output, refined_labels, inner_feature, refined_feature)
        else:
            model_output, _ = self.model(input)
            return model_output
