import math
from statistics import mean
# ####################### Model Construction #######################
# model dim
input_size = 288
expand_ratio = 4
# b1
width_list = [16, 32, 64, 128, 256]
depth_list=[1, 2, 3, 3, 4]
dim = 16
# b2
# width_list = [24, 48, 96, 192, 384]
# depth_list=[1, 3, 4, 4, 6]
# dim = 32
# type: Conv, DSConv, MBConv, MSA
block_list = []
# FIXME:
# input stem
block_list.append({"type": "Conv", "kernel_size": 3, "input_H": input_size, "input_C": 3, "output_E": input_size//2, "output_M": width_list[0]})
input_size = input_size//2
for i in range(depth_list[0]):
    block_list.append({"type": "DSConv", "kernel_size": 3, "input_H": input_size, "input_C": width_list[0], "output_E": input_size, "output_M": width_list[0]})
    in_channels = width_list[0]

# MBConv stages
for w, d in zip(width_list[1:3], depth_list[1:3]):
    for i in range(d):
        stride = 2 if i == 0 else 1
        if stride == 2:
           output_size = input_size//2
        else:
            output_size = input_size
        block_list.append({"type": "MBConv", "kernel_size": 3, "input_H": input_size, "input_C": in_channels, "output_E": output_size, "output_M": w})
        in_channels = w
        input_size = output_size

# EfficientViT Block stages
for w, d in zip(width_list[3:], depth_list[3:]):
    # MBConv
    stride = 2
    output_size = input_size//2
    block_list.append({"type": "MBConv", "kernel_size": 3, "input_H": input_size, "input_C": in_channels, "output_E": output_size, "output_M": w})
    in_channels = w
    input_size = output_size
    # EfficientViT Block
    for _ in range(d):
        # Lightweight MSA
        block_list.append({"type": "MSA", "kernel_size": 5, "input_H": input_size, "input_C": in_channels, "output_E": input_size, "output_M": in_channels, "dim": dim})
        # MBConv
        block_list.append({"type": "MBConv", "kernel_size": 3, "input_H": input_size, "input_C": in_channels, "output_E": input_size, "output_M": in_channels})
 
# calculate FLOPs
# type: Conv, DSConv, MBConv, MSA
Overall_FLOPs = 0
Stem_FLOPs = 0
PW_FLOPs = 0
DW_FLOPs = 0
MatMul_FLOPs = 0
for block in block_list:
    if block["type"] == "Conv":
        Stem_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["kernel_size"]*block["kernel_size"]*block["input_C"]
    elif block["type"] == "DSConv":
        DW_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*block["kernel_size"]*block["kernel_size"]
        PW_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]
    elif block["type"] == "MBConv":
        PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*expand_ratio*block["input_C"]
        DW_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*expand_ratio*block["kernel_size"]*block["kernel_size"]
        PW_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]*expand_ratio
    elif block["type"] == "MSA":   
        # proj
        PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["input_C"]
        # aggregation
        DW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["kernel_size"]*block["kernel_size"]
        Overall_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["dim"]
        # MSA:      FLOPs(A=K^T*V & Q*A)*num_head*2
        # MSA: 2*token_dim*feature_dim*feature_dim*num_head*2
        MatMul_FLOPs += 2*block["input_H"]*block["input_H"]*block["dim"]*block["dim"]*(block["input_C"]//block["dim"])*2
        # proj
        PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*2*block["input_C"]
        
Overall_FLOPs = PW_FLOPs + DW_FLOPs + MatMul_FLOPs + Stem_FLOPs   
print("FLOPs of Stem are: ", Stem_FLOPs/1e6, Stem_FLOPs/Overall_FLOPs*100)
print("FLOPs of PWConv are: ", PW_FLOPs/1e6, PW_FLOPs/Overall_FLOPs*100)
print("FLOPs of DWConv are: ", DW_FLOPs/1e6, DW_FLOPs/Overall_FLOPs*100)
print("FLOPs of MatMul are: ", MatMul_FLOPs/1e6, MatMul_FLOPs/Overall_FLOPs*100)
print("FLOPs of Overall are: ", Overall_FLOPs/1e6, Overall_FLOPs/Overall_FLOPs*100)
print("")

# ######################## HW Simulation ########################
# HW Config
# Multiplier-Adder-Tree: Input Broadcast
# PW: inner PE lane: Cin Paral/ among PE lane: Cout Paral
num_MAT_PE = 8
num_MAT_lane = 8
# Reconfig-PE: Support both self-acc and down-forward acc
# PW: inner PE lane: Cin Parall/ among PE lane: Cout Parall
# DW: inner PE lane: Cout Parall/ among PE lane: Output Pixel (feature map) Parall
num_Reg_PE = 8
num_Reg_lane = 8
stored_cout_pw = 32
# FIXME:
utilization_thresh = 0.9
total_MAT_PE = num_MAT_PE*num_MAT_lane
total_Reg_PE = num_Reg_PE*num_Reg_lane
total_all_PE = total_MAT_PE + total_Reg_PE
PE_util = []
Flag = False
total_cycles = 0


def use_all_PEs(total_cycles, PE_util, out_channel, in_channel, remaining, type):
    cycles = 0
    for _Cout in range(math.ceil(out_channel/(num_Reg_lane+num_MAT_lane))):
        for _Cin in range(math.ceil(in_channel/num_MAT_PE)):
            cycles += 1
    total_cycles += cycles * remaining
    PE_util.append((in_channel*out_channel)/(cycles*total_all_PE))
    if PE_util[-1] < utilization_thresh:
        print("*******Warning: Low utilization in PW/Conv/GConv/MatMul: ", PE_util[-1])
        print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
        print("")
    elif PE_util[-1] > 1:
        print("*******Warning: Error utilization in PW/Conv/GConv/MatMul: ", PE_util[-1])
        print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
        print("")
    # print(cycles * remaining)
    return total_cycles, PE_util
        
        
def dw_pw_fuse(total_cycles, PE_util, output_E, in_channel, out_channel, kernel_size, type, gconv_in_channel=None):
    # print("gconv_in_channel ", gconv_in_channel)
    Flag = False 
    stored_cout_pw = num_MAT_lane + num_Reg_lane
    # dw
    cycles = 0
    for _fm in range(math.ceil(output_E/num_Reg_lane)):
        for _Cout in range(math.ceil(in_channel/num_Reg_PE)):
            cycles += 1
    PE_util.append((output_E*in_channel)/(cycles*total_Reg_PE))
    if PE_util[-1] < utilization_thresh:
        print("*******Warning: Low utilization in DW: ", PE_util[-1])
        print("{type, output_E, out_channel}: ", type, output_E, in_channel)
        print("")
    elif PE_util[-1] > 1:
        print("*******Warning: Error utilization in DW: ", PE_util[-1])
        print("{type, output_E, out_channel}: ", type, output_E, in_channel)
        print("")
    DW_cycles = cycles * output_E*kernel_size*kernel_size
    setup_cycles = math.ceil(in_channel/num_Reg_PE)*kernel_size*kernel_size

    # PW
    if gconv_in_channel:
        in_channel = gconv_in_channel
    stored_cout_pw = min(stored_cout_pw, out_channel)
    computed_pixels = (DW_cycles-setup_cycles)//(math.ceil(in_channel/num_MAT_PE))*num_MAT_lane
    # print("{type, out_channel, in_channel, output_E, output_E}: ", type, out_channel, in_channel, output_E, output_E)
    # print("computed_pixels: ", computed_pixels)
    # if gconv_in_channel:
    #     print("DW_cycles-setup_cycles:", DW_cycles-setup_cycles)
    #     print('computed_pixels:', computed_pixels)
    #     print("output_E*output_E:", output_E*output_E)
    computed_cout = computed_pixels//(2*num_MAT_lane*output_E*output_E)*stored_cout_pw
    # print("computed_cout: ", computed_cout)
    if not computed_cout < out_channel:
        print("Already finished computations of PW before layer fusion!!!!!!")
        print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
        print("")
        PE_util.append((in_channel*out_channel*output_E*output_E)/(num_MAT_PE*num_MAT_lane*DW_cycles))
        PW_cycles = 0
        if PE_util[-1] < utilization_thresh:
            print("*******Warning: Low utilization in PW after fusion (1st): ", PE_util[-1])
            print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
            print("")
        elif PE_util[-1] > 1:
            print("*******Warning: Error utilization in PW after fusion (1st): ", PE_util[-1])
            print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
            print("")
    else:
        stored_cout_pw_original = stored_cout_pw
        remianing_cout = out_channel - computed_cout
        # print("remianing_cout: ", remianing_cout)
        stored_cout_pw = min(stored_cout_pw, remianing_cout)
        remaining_pixels = output_E*output_E - (computed_pixels - computed_cout*output_E*output_E)//stored_cout_pw_original
        # print("remaining_pixels: ", remaining_pixels)
        # The computation of DW ends, then use all PEs to compute PW
        cycles = 0
        for _Cout in range(math.ceil(stored_cout_pw_original/(num_MAT_lane+num_Reg_lane))):
            for _Cin in range(math.ceil(in_channel/num_MAT_PE)):
                cycles += 1
        PW_cycles = cycles*remaining_pixels
        PE_util.append((in_channel*stored_cout_pw_original)/(cycles*total_all_PE))
        if PE_util[-1] < utilization_thresh:
            print("*******Warning: Low utilization in PW after fusion (1st): ", PE_util[-1])
            print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
            print("")
        elif PE_util[-1] > 1:
            print("*******Warning: Error utilization in PW after fusion (1st): ", PE_util[-1])
            print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
            print("")
        remianing_cout -= stored_cout_pw_original
        if remianing_cout > 0:
            cycles = 0
            for _Cout in range(math.ceil(remianing_cout/(num_MAT_lane+num_Reg_lane))):
                for _Cin in range(math.ceil(in_channel/num_MAT_PE)):
                    cycles += 1
            PW_cycles += cycles*(output_E*output_E)
            PE_util.append((in_channel*remianing_cout)/(cycles*total_all_PE))
            if PE_util[-1] < utilization_thresh:
                print("*******Warning: Low utilization in PW after fusion (2st): ", PE_util[-1])
                print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
                print("")
            elif PE_util[-1] > 1:
                print("*******Warning: Error utilization in PW after fusion (2st): ", PE_util[-1])
                print("{type, out_channel, in_channel}: ", type, out_channel, in_channel)
                print("")
        
        PW_cycles += (DW_cycles - setup_cycles)
    
    total_cycles += max(DW_cycles, PW_cycles + setup_cycles)
    # print(DW_cycles, PW_cycles, setup_cycles)
    # print(max(DW_cycles, PW_cycles + setup_cycles))
    return total_cycles, PE_util
        
        
for block in block_list:
    if block["type"] == "Conv":
        # use all PEs to simultaneously compute
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["kernel_size"]*block["kernel_size"]*block["input_C"]
        total_cycles, PE_util = use_all_PEs(total_cycles, PE_util, block["output_M"], block["input_C"], block["output_E"]*block["output_E"]*block["kernel_size"]*block["kernel_size"], block["type"])
                   
    elif block["type"] == "DSConv":
        # Fuse DW and PW
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*block["kernel_size"]*block["kernel_size"]
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]
        total_cycles, PE_util = dw_pw_fuse(total_cycles, PE_util, block["output_E"], block["input_C"], block["output_M"], block["kernel_size"], block["type"])
        
    elif block["type"] == "MBConv":
        # Use all PEs
        # FIXME:
        # Overall_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*expand_ratio*block["input_C"]
        total_cycles, PE_util = use_all_PEs(total_cycles, PE_util, block["input_C"]*expand_ratio, block["input_C"], block["input_H"]*block["input_H"], block["type"])
        
        # Fuse DW and PW
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*expand_ratio*block["kernel_size"]*block["kernel_size"]
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]*expand_ratio
        total_cycles, PE_util = dw_pw_fuse(total_cycles, PE_util, block["output_E"], block["input_C"]*expand_ratio, block["output_M"], block["kernel_size"], block["type"])
        
    elif block["type"] == "MSA":   
        # proj: use all PE
        # PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["input_C"]
        total_cycles, PE_util = use_all_PEs(total_cycles, PE_util, block["input_C"]*3, block["input_C"], block["input_H"]*block["input_H"], block["type"])
            
        # aggregation: DW and PW Fusion
        # DW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["kernel_size"]*block["kernel_size"]
        # Overall_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["dim"]
        total_cycles, PE_util = dw_pw_fuse(total_cycles, PE_util, block["input_H"], block["input_C"]*3, block["output_M"], block["kernel_size"], block["type"], gconv_in_channel=block["dim"])
        
        # MSA: use all PE
        # MSA:      FLOPs(A=K^T*V & Q*A)*num_head*2
        # MSA: 2*token_dim*feature_dim*feature_dim*num_head*2
        # MatMul_FLOPs += 2*block["input_H"]*block["input_H"]*block["dim"]*block["dim"]*(block["input_C"]//block["dim"])*2
        cycles = 0
        # different heads map tp different PE Arrays
        for _head in range(((block["input_C"]//block["dim"])*2)//2):
            for _Cout in range(math.ceil(block["dim"]/num_Reg_lane)):
                for _Cin in range(math.ceil(block["input_H"]*block["input_H"]/num_MAT_PE)):
                    cycles += 1
        total_cycles += cycles * block["dim"]
        PE_util.append((block["dim"]*block["input_H"]*block["input_H"])/((_Cout+1)*(_Cin+1)*total_MAT_PE))
        if PE_util[-1] < utilization_thresh:
            print("*******Warning: Low utilization in MatMal (1st): ", PE_util[-1])
            print("{type, output_H, output_W}: ", type, block["input_H"]*block["input_H"], block["dim"])
            print("")
        elif PE_util[-1] > 1:
            print("*******Warning: Error utilization in MatMal (1st): ", PE_util[-1])
            print("{type, output_H, output_W}: ", type, block["input_H"]*block["input_H"], block["dim"])
            print("")
            
        cycles = 0
        for _head in range(((block["input_C"]//block["dim"])*2)//2):
            for _Cout in range(math.ceil(block["dim"]/num_Reg_lane)):
                for _Cin in range(math.ceil(block["dim"]/num_MAT_PE)):
                    cycles += 1
        total_cycles += cycles * block["input_H"]*block["input_H"]
        PE_util.append((block["dim"]*block["dim"])/((_Cout+1)*(_Cin+1)*total_MAT_PE))
        if PE_util[-1] < utilization_thresh:
            print("*******Warning: Low utilization in MatMal (2st): ", PE_util[-1])
            print("{type, output_H, output_W}: ", type, block["dim"], block["dim"])
            print("")
        elif PE_util[-1] > 1:
            print("*******Warning: Error utilization in MatMal (2st): ", PE_util[-1])
            print("{type, output_H, output_W}: ", type, block["dim"], block["dim"])
            print("")
            
        # proj: use all PE
        # PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*2*block["input_C"]
        total_cycles, PE_util = use_all_PEs(total_cycles, PE_util, block["input_C"]*2, block["input_C"], block["input_H"]*block["input_H"], block["type"])
    
    # print("Average PE Utilization is: ",(Overall_FLOPs/(total_cycles*total_all_PE)))
    # print('total_cycles: ', total_cycles)
    # print('Overall_FLOPs: ', Overall_FLOPs)
        
print("")        
print("PE Utilization is: ", PE_util)
# print("Average PE Utilization is: ", mean(PE_util))
print("Average PE Utilization is: ",(Overall_FLOPs/(total_cycles*total_all_PE)))
print("Overall Cycles is: ", total_cycles)