import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough, QuantModule_Shifted
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
from EfficientViT.models.nn import DSConv, MBConv, EfficientViTBlock, IdentityLayer, ResidualBlock, LiteMSA
from EfficientViT.plot import plot_distribution

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = basic_block.relu2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        self.plot = False
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=False),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=False),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        if self.plot:
            activation = []
            activation.append(self.conv[0].out)
            activation.append(self.conv[1].out)
            activation.append(self.conv[2].out)
            plot_distribution(activation, "MBV2")
            exit()
        return out


# TODO: Support ResidualBlock in EfficientViT
class QauntMBBlock(BaseQuantBlock):
    """
    Implementation of Quantized MB Block in EfficientViT.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}, disable_act_quant=False):
        super().__init__(act_quant_params)

        # self.use_res_connect = inv_res.use_res_connect
        # self.expand_ratio = inv_res.expand_ratio
        self.inv_res = inv_res
        self.plot = False
        if isinstance(inv_res.main, DSConv):
            self.conv = nn.Sequential(
                QuantModule(inv_res.main.depth_conv.conv, weight_quant_params, act_quant_params,),
                QuantModule(inv_res.main.point_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=False),
            )
            # self.conv = nn.Sequential(
            #     QuantModule(inv_res.main.depth_conv.conv, weight_quant_params, act_quant_params,disable_act_quant=True),
            #     QuantModule_Shifted(inv_res.main.point_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=False),
            # )
            self.conv[0].activation_function = nn.Hardswish()
        
        elif isinstance(inv_res.main, MBConv):
            # self.conv = nn.Sequential(
            #     QuantModule(inv_res.main.inverted_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=disable_act_quant),
            #     QuantModule(inv_res.main.depth_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=disable_act_quant),
            #     QuantModule(inv_res.main.point_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=disable_act_quant),
            # )
            self.conv = nn.Sequential(
                QuantModule(inv_res.main.inverted_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=True),
                QuantModule_Shifted(inv_res.main.depth_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=True),
                QuantModule_Shifted(inv_res.main.point_conv.conv, weight_quant_params, act_quant_params, disable_act_quant=False),
            )
            self.conv[0].activation_function = nn.Hardswish()
            self.conv[1].activation_function = nn.Hardswish()
            self.plot = inv_res.main.plot
        
        # FIXME: quantize activation here
        elif isinstance(inv_res.main, LiteMSA):
            self.qkv = QuantModule(inv_res.main.qkv.conv, weight_quant_params, act_quant_params, disable_act_quant=True)
            
            self.aggreg = nn.ModuleList(
                [
                    nn.Sequential(
                        QuantModule(inv_res.main.aggreg[0][0], weight_quant_params, act_quant_params, disable_act_quant=True),
                        QuantModule(inv_res.main.aggreg[0][1], weight_quant_params, act_quant_params, disable_act_quant=True)
                    )
                ]
            )
            
            self.proj = QuantModule(inv_res.main.proj.conv, weight_quant_params, act_quant_params, disable_act_quant=True)
            self.kernel_func = nn.ReLU(inplace=False)
            self.dim = inv_res.main.dim
    
    
    def forward_attn(self, x):
        B, _, H, W = list(x.size())
        # print("x.shape: ", x.shape)
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0 : self.dim],
            multi_scale_qkv[..., self.dim : 2 * self.dim],
            multi_scale_qkv[..., 2 * self.dim :],
        )

        # TODO: quantize attn
        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        # print("q.shape: ", q.shape)
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        # print("out.shape: ", out.shape)
        out = torch.reshape(out, (B, -1, H, W))
        # print("out.shape: ", out.shape)
        out = self.proj(out) 
        return out   


    def forward(self, x):
        try:
            shortcut = isinstance(self.inv_res.shortcut, IdentityLayer)
        except:
            shortcut = False
            
        if isinstance(self.inv_res.main, LiteMSA):
            out = self.forward_attn(x) + x
        elif shortcut:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
                
            out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        if self.plot:
            activation = []
            activation.append(self.conv[1].shifted_input)
            activation.append(self.conv[2].shifted_input)
            plot_distribution(activation, "Shifted_EViT_Pre_MB")
            exit()
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    ResidualBlock: QauntMBBlock
}
