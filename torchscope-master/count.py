from efficientnet import efficientnet_b0, efficientnet_b3
from efficientnet_ex import efficientnet_ex, efficientnet_exx
from torchscope import scope
import torchvision.models as models

# model = models.resnet18()
# model = efficientnet_b0()
model = efficientnet_ex()
# model = efficientnet_exx()
# print(model)
scope(model, input_size=(3, 32, 32), batch_size=2, device='cpu')