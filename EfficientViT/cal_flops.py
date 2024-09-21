import torch
import thop
from torch import nn
from EfficientViT.models.nn.ops import MatMul

def dwconv_hook(module, input, output):
    global dwconv_flops
    # print(input)
    # dwconv_flops += thop.profile(module, inputs=(input[0], ))
    kernel_height, kernel_width = module.kernel_size
    out_channels, output_height, output_width = output.size(1), output.size(2), output.size(3)
    flops = kernel_height * kernel_width * out_channels * output_height * output_width
    dwconv_flops += flops
    
def pwconv_hook(module, input, output):
    global pwconv_flops
    # pwconv_flops += thop.profile(module, inputs=(input[0], ))
    out_channels, output_height, output_width = output.size(1), output.size(2), output.size(3)
    int_channels = input[0].size(1)
    flops = int_channels * out_channels * output_height * output_width
    pwconv_flops += flops

def linear_hook(module, input, output):
    global linear_flops
    # linear_flops += thop.profile(module, inputs=(input[0], ))
    input_size, output_size = input[0].size(1), output.size(1)
    flops = input_size * output_size
    linear_flops += flops
    
def matmul_hook(module, input, output):
    global matmul_flops
    # linear_flops += thop.profile(module, inputs=(input[0], ))
    head, token, feature = input[0].size(1), input[0].size(2), input[0].size(3)
    flops = head * token * feature * output.size(3)
    matmul_flops += flops

# def dwconv_hook(module, input, output):
#     global dwconv_flops
#     dwconv_flops += thop.profile(module, inputs=(input[0], ))
    
# def pwconv_hook(module, input, output):
#     global pwconv_flops
#     pwconv_flops += thop.profile(module, inputs=(input[0], ))

# def linear_hook(module, input, output):
#     global linear_flops
#     linear_flops += thop.profile(module, inputs=(input[0], ))
    
# def matmul_hook(module, input, output):
#     global matmul_flops
#     matmul_flops += thop.profile(module, inputs=(input[0], ))

dwconv_flops = 0
pwconv_flops = 0
linear_flops = 0
matmul_flops = 0

def Cal_FLOPs(model):
    hooks = []
    # for module in profile_model.modules():
    #     if isinstance(module, nn.Conv2d):
    #         if module.out_channels == module.groups:
    #             hook = module.register_forward_hook(dwconv_hook)
    #             hooks.append(hook)
    #         elif module.kernel_size == (1,1):
    #             hook = module.register_forward_hook(pwconv_hook)
    #             hooks.append(hook)
    #     elif isinstance(module, nn.Linear):
    #         hook = module.register_forward_hook(linear_hook)
    #         hooks.append(hook)
    #     elif isinstance(module, MatMul):
    #         hook = module.register_forward_hook(matmul_hook)
    #         hooks.append(hook)
    #     else:
    #         continue
    
    def foo(model):
        childrens = list(model.children())
        if not childrens:
            if isinstance(model, nn.Conv2d):
                if model.out_channels == model.groups:
                    hook = model.register_forward_hook(dwconv_hook)
                    hooks.append(hook)
                elif model.kernel_size == (1,1) and model.groups == 1:
                    hook = model.register_forward_hook(pwconv_hook)
                    hooks.append(hook)
            # elif isinstance(model, nn.Linear):
            #     hook = model.register_forward_hook(linear_hook)
            #     hooks.append(hook)
            elif isinstance(model, MatMul):
                hook = model.register_forward_hook(matmul_hook)
                hooks.append(hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.randn(1, 3, 224, 224)
    _ = model(input)

    for hook in hooks:
        hook.remove()

    print("Total DW Convolutional FLOPs:", dwconv_flops/1e6)
    print("Total PW Convolutional FLOPs:", pwconv_flops/1e6)
    print("Total MatMul FLOPs:", (linear_flops+matmul_flops)/1e6)
    print("Overall FLOPs:", (dwconv_flops+pwconv_flops+linear_flops+matmul_flops)/1e6)
    
    # print("")
    # flops, params = thop.profile(model, inputs=(input,))
    # print(flops/1e6)
