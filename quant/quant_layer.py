import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.autograd import Function
import numpy as np


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    shift = torch.round(torch.log(x_abs) / np.log(2))
    return shift, sign 


def round_power_of_two(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    return x_rounded


class RoundPowerOf2(Function):
    @staticmethod 
    def forward(ctx, input, stochastic=False):
        return round_power_of_two(input, stochastic)

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None


def round_power_of_2(input, stochastic=False):
    return RoundPowerOf2.apply(input, stochastic)


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    # TODO:
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        # FIXME:
        self.sym = False
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.is_act = False 

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True
        
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            # n_channels = x_clone.shape[0]
            # if len(x.shape) == 4:
            #     x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            if len(x.shape) == 4:
                if self.is_act:
                    n_channels = x_clone.shape[1]
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=0)[0]
                else:
                    n_channels = x_clone.shape[0]
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                n_channels = x_clone.shape[0]
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                # delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
                if self.is_act:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,:,:], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                # delta = delta.view(-1, 1, 1, 1)
                # zero_point = zero_point.view(-1, 1, 1, 1)
                if self.is_act:
                    delta = delta.view(1, -1, 1, 1)
                    zero_point = zero_point.view(1, -1, 1, 1)
                else:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class LogQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(LogQuantizer, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        y = torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda ())))
        out = torch.gt((x-2**y),(2**(y+1)-x))
        y += out
        # TODO:
        y = torch.clamp(y, -12, 11)
        # y[x==0] = 0
        return 2**y


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        self.out = None

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        self.input = input
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        self.out = out
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant     


class QuantModule_Shifted(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule_Shifted, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.input_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.output_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        self.shifted_input = None
        self.z = None
        self.scaling = None
        self.enbale_scale = False 


    def get_ratio(self, max, v):
        mean = torch.mean(max)
        percentile_alpha = 0.99
        try:
            cur_max = torch.quantile(v.reshape(-1), percentile_alpha)
        except:
            cur_max = torch.tensor(np.percentile(
                v.reshape(-1).cpu().detach().numpy(), percentile_alpha * 100),
                                   device=v.device,
                                   dtype=torch.float32)
        cur_max = torch.max(cur_max, mean)
        div = max/cur_max
        power = torch.clamp(torch.round(torch.div(torch.log(div),torch.log(torch.Tensor([2]).cuda ()))), 0, 5)
        return 2**power
    
    
    def forward(self, input: torch.Tensor):
        ############ shift ############
        if self.z == None:
            x_max = input.max(axis=2).values
            x_min = input.min(axis=2).values
            x_max_max = x_max.max(axis=2).values
            x_min_min = x_min.min(axis=2).values
            channel_max = x_max_max.max(axis=0).values
            channel_min = x_min_min.min(axis=0).values
            self.z = (channel_max + channel_min)/2
            # TODO:
            if self.enbale_scale:
                channel_max = (channel_max - channel_min)/2
                shifted_input = input - self.z.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(1, input.shape[1], input.shape[2], input.shape[3])
                self.scaling = self.get_ratio(channel_max, shifted_input)
        # FIXME:
        if self.use_act_quant or (self.use_weight_quant and self.enbale_scale):  
        # if True:  
            shifted_input = torch.zeros_like(input, dtype=input.dtype)
            shifted_input = input - self.z.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(1, input.shape[1], input.shape[2], input.shape[3])
            self.shifted_input = shifted_input
            if self.enbale_scale:
                shifted_input /= self.scaling.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, input.shape[1], input.shape[2], input.shape[3]).cuda()
                self.shifted_input = shifted_input
                self.z /= self.scaling
                
                if self.weight.shape[1] == 1:
                    weight = self.weight*self.scaling.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]).cuda()
                else:
                    weight = self.weight*self.scaling.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]).cuda()   
            else:
                weight = self.weight
            
            if self.use_act_quant:
                shifted_input = self.input_quantizer(shifted_input)
                   
            if self.use_weight_quant:
                weight = self.weight_quantizer(weight)
            # weight = self.weight
            bias = self.bias
            # else:
            #     weight = self.org_weight
            #     bias = self.org_bias
            shifted_bias = torch.zeros_like(bias.max(), dtype=bias.dtype)
            
            if weight.shape[1] == 1:
                diff_input = self.z.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(1, input.shape[1], input.shape[2], input.shape[3])
                out = self.fwd_func(shifted_input, weight, bias, **self.fwd_kwargs)
                diff_out = self.fwd_func(diff_input, weight, None, **self.fwd_kwargs)
                out += diff_out.expand(out.shape[0], out.shape[1], out.shape[2], out.shape[3])
            else:
                weight_sum = torch.sum(weight, dim=(2, 3))
                shifted_bias = torch.squeeze(torch.matmul(weight_sum, torch.unsqueeze(self.z, dim=1)))
                shifted_bias += bias
                out = self.fwd_func(shifted_input, weight, shifted_bias, **self.fwd_kwargs)
        
        else:
            self.shifted_input = input
            weight = self.weight
            if self.use_weight_quant:
                weight = self.weight_quantizer(weight)
            bias = self.bias
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            
            out = self.activation_function(out)
            if self.disable_act_quant:
                return out
            if self.use_act_quant:
                out = self.output_quantizer(out)
        self.out = out
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
               
class QuantModule_Scaled(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule_Scaled, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.input_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.output_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        self.scaling = None
        self.enbale_scale = False 
        self.scaled_input = False
        self.z = None
        self.shifted_input = False
        self.enbale_shift = False 

    def get_ratio(self, channel_max, channel_min, v):
        mean = torch.mean(channel_max)
        percentile_alpha = 0.99
        try:
            cur_max = torch.quantile(v.reshape(-1), percentile_alpha)
        except:
            cur_max = torch.tensor(np.percentile(
                v.reshape(-1).cpu().detach().numpy(), percentile_alpha * 100),
                                   device=v.device,
                                   dtype=torch.float32)
        # TODO:
        cur_max = torch.max(cur_max, mean)
        # cur_max = mean
        div = channel_max/cur_max
        # div[div > 64] = 64
        # div[div < -64] = -64
        # # thresh = int((cur_max/min.min()).abs())
        # thresh = 32
        # # ########### Automated Version ###########
        # best_score = 1e+10
        # for i in range(thresh):
        #     thresh_new = (i+1)
        #     div_new = div.clone()
        #     div_new[(div < 1/thresh_new) & (div > 0)] = 1/thresh_new
        #     div_new[(div > -1/thresh_new) & (div < 0)] = -1/thresh_new
        #     div_new[div == 0] = 1
        #     div_new_expand = div_new.view(1, -1, 1, 1)
        #     scaled_input = v/div_new_expand.cuda()
        #     # quantize
        #     x_min = min(scaled_input.min().item(), 0)
        #     x_max = max(scaled_input.max().item(), 0)
        #     delta = float(x_max - x_min) / (self.input_quantizer.n_levels - 1)
        #     delta = torch.tensor(delta).type_as(scaled_input)
        #     zero_point = (-x_min / delta).round()
        #     x_int = torch.round(scaled_input / delta)
        #     x_quant = torch.clamp(x_int + zero_point, 0, self.input_quantizer.n_levels - 1)
        #     x_float_q = (x_quant - zero_point) * delta
        #     # L_p norm minimization as described in LAPQ
        #     # https://arxiv.org/abs/1911.07190
        #     score = lp_loss(scaled_input, x_float_q, p=2.4, reduction='all')
        #     if score < best_score:
        #         best_score = score
        #         div_best = div_new.clone()
        # # thresh = 10
        # # print(div)
        # ########## version 1 ##########
        div[div > 64] = 64
        div[div < 1/10] = 1/10
        div[div == 0] = 1
        # ########## version 2 ##########
        # thresh = 10
        # div[div > 64] = 64
        # div[(div < 1/thresh) & (div > 0)] = 1/thresh
        # div[div == 0] = 1
        # div[(div > -1/thresh) & (div < 0)] = -1/thresh
        # ########## version 3 ##########
        # div[div < 1] = 1
        # power = torch.clamp(torch.round(torch.div(torch.log(div),torch.log(torch.Tensor([2]).cuda ()))), 0, 5)
        # return 2**power
        return div
    
    def forward(self, input: torch.Tensor):
        # ############ scaling ############
        if self.scaling == None:
            x_max = input.max(axis=2).values
            x_min = input.min(axis=2).values
            x_max_max = x_max.max(axis=2).values
            x_min_min = x_min.min(axis=2).values
            channel_max = x_max_max.max(axis=0).values
            channel_min = x_min_min.min(axis=0).values
            self.scaling = self.get_ratio(channel_max, channel_min, input)
        
        if (self.use_act_quant or self.use_weight_quant) and self.enbale_scale:  
        # if True:  
            scaled_input = input/self.scaling.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(1, input.shape[1], input.shape[2], input.shape[3]).cuda()
            if self.weight.shape[1] == 1:
                weight = self.weight*self.scaling.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]).cuda()
            else:
                weight = self.weight*self.scaling.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]).cuda()   
        else:
            weight = self.weight
            scaled_input = input
        
        self.scaled_input = scaled_input 
        
        if self.use_act_quant and (not self.enbale_shift): 
            scaled_input = self.input_quantizer(scaled_input) 
            
        if self.use_weight_quant:
            weight = self.weight_quantizer(weight)
        bias = self.bias
        
        # TODO:
        ############ shift ############
        if self.z == None:
            x_max = scaled_input.max(axis=2).values
            x_min = scaled_input.min(axis=2).values
            x_max_max = x_max.max(axis=2).values
            x_min_min = x_min.min(axis=2).values
            channel_max = x_max_max.max(axis=0).values
            channel_min = x_min_min.min(axis=0).values
            self.z = (channel_max + channel_min)/2
            
        if self.use_act_quant and self.enbale_shift:  
        # if True:
            shifted_input = torch.zeros_like(scaled_input, dtype=scaled_input.dtype)
            shifted_input = scaled_input - self.z.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(1, input.shape[1], input.shape[2], input.shape[3])
            self.shifted_input = shifted_input
        
            if self.use_act_quant: 
                shifted_input = self.input_quantizer(shifted_input)
        
            shifted_bias = torch.zeros_like(bias.max(), dtype=bias.dtype)
            if weight.shape[1] == 1:
                diff_input = self.z.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).expand(1, input.shape[1], input.shape[2], input.shape[3])
                out = self.fwd_func(shifted_input, weight, bias, **self.fwd_kwargs)
                diff_out = self.fwd_func(diff_input, weight, None, **self.fwd_kwargs)
                out += diff_out.expand(out.shape[0], out.shape[1], out.shape[2], out.shape[3])
            else:
                weight_sum = torch.sum(weight, dim=(2, 3))
                shifted_bias = torch.squeeze(torch.matmul(weight_sum, torch.unsqueeze(self.z, dim=1)))
                shifted_bias += bias
                out = self.fwd_func(shifted_input, weight, shifted_bias, **self.fwd_kwargs)
        else:    
            out = self.fwd_func(scaled_input, weight, bias, **self.fwd_kwargs)
        
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.output_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
