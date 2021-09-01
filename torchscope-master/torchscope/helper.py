import numpy as np
import torch.nn as nn
from efficientnet import Swish, L2norm, Arcface
from efficientnet_ex import Swish, L2norm, Arcface
__all__ = ["compute_flops", "compute_madd"]


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out) // inp.size()[0]
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out) // inp.size()[0]
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out) //inp.size()[0]
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp, out) // inp.size()[0]
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out) // inp.size()[0]
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out) // inp.size()[0]
    else:
        return 0


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    # print(inp.size())
    assert len(inp.size()) == 2 and len(out.size()) == 2
    batch_size = inp.size()[0]
    return batch_size*inp.size()[1] * out.size()[1] 


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s

    return output_elements_count


def compute_madd(module, inp, out, mul_factor = 1.0):
    # print(module)
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return compute_AdaptiveAvgPool2d_madd(module, inp, out, mul_factor)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return compute_ReLU_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.Linear):
        return compute_Linear_madd(module, inp, out, mul_factor)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_madd(module, inp[0], inp[1], out, mul_factor)
    elif isinstance(module, Swish):
        return compute_Swish_madd(module,  inp, out, mul_factor)
    elif isinstance(module, nn.Sigmoid):
        return compute_Sigmoid_madd(module,  inp, out, mul_factor)
    elif isinstance(module, L2norm):
        return compute_L2norm_madd(module,  inp, out, mul_factor)
    elif isinstance(module, Arcface):
        return compute_Arcface_madd(module,  inp, out, mul_factor)
    else:
        return 0


def compute_Conv2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    # ops per output element
    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return mul_factor*total_mul + total_add


def compute_ConvTranspose2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return mul_factor*total_mul + total_add


def compute_BatchNorm2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]

    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    return mul_factor * 2 * in_c * in_h * in_w + 2 * in_c * in_h * in_w 


def compute_MaxPool2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.MaxPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_AdaptiveAvgPool2d_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.AdaptiveAvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    kernel_size = inp.size()
    kernel_size = kernel_size[2:4]
    if isinstance(module.output_size, (tuple, list)):
        k_h, k_w = kernel_size
    else:
        k_h, k_w = kernel_size
    out_c, out_h, out_w = out.size()[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_ReLU_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, (nn.ReLU, nn.ReLU6))

    count = 1
    for i in inp.size()[1:]:
        count *= i
    return count


def compute_Softmax_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.Softmax)
    assert len(inp.size()) > 1

    count = 1
    for s in inp.size()[1:]:
        count *= s
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_Linear_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2 and len(out.size()) == 2

    num_in_features = inp.size()[1]
    num_out_features = out.size()[1]

    mul = num_in_features
    add = num_in_features - 1
    return num_out_features * (mul_factor*mul + add)


def compute_Bilinear_madd(module, inp1, inp2, out, mul_factor = 1.0):
    assert isinstance(module, nn.Bilinear)
    assert len(inp1.size()) == 2 and len(inp2.size()) == 2 and len(out.size()) == 2

    num_in_features_1 = inp1.size()[1]
    num_in_features_2 = inp2.size()[1]
    num_out_features = out.size()[1]

    mul = num_in_features_1 * num_in_features_2 + num_in_features_2
    add = num_in_features_1 * num_in_features_2 + num_in_features_2 - 1
    return num_out_features * (mul_factor*mul + add)

def compute_Swish_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, Swish)
    assert len(inp.size()) == 4 and len(out.size()) == 4

    num_in_features = inp.size()[1]
    features_size = inp.size()[2]
    total_mul = mul_factor * 3 * num_in_features*(features_size**2) + num_in_features*(features_size**2)
    return total_mul

def compute_Sigmoid_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, nn.Sigmoid)
    assert len(inp.size()) == 4 and len(out.size()) == 4
    num_in_features = inp.size()[1]
    features_size = inp.size()[2]
    total_mul = mul_factor * 2 *num_in_features*(features_size**2)+ num_in_features*(features_size**2)
    return total_mul

def compute_L2norm_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, L2norm)
    assert len(inp.size()) == 2 and len(out.size()) == 2
    num_in_features = inp.size()[1]
    # num_out_features = out.size()[1]
    norm_add = num_in_features - 1
    norm_mul = num_in_features 
    norm_div = 1
    norm_ops = norm_add + mul_factor * norm_mul + norm_div
    total_count = num_in_features * norm_ops
    return total_count

def compute_Arcface_madd(module, inp, out, mul_factor = 1.0):
    assert isinstance(module, Arcface)
    assert len(inp.size()) == 2 and len(out.size()) == 2
    num_in_features = inp.size()[1]
    num_out_features = out.size()[1]
    mat_add = num_in_features - 1
    mat_mul = num_in_features 
    mat_op = mul_factor * mat_mul+mat_add
    total_count = mat_op * num_out_features
    return total_count
