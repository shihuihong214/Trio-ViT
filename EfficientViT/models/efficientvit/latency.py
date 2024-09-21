import math
from statistics import mean
# ####################### Model Construction #######################
# model dim
input_size = 224
expand_ratio = 4
# b1
width_list = [16, 32, 64, 128, 256]
depth_list=[1, 2, 3, 3, 4]
dim = 16
# type: Conv, DSConv, MBConv, MSA
block_list = []
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
PW_FLOPs = 0
DW_FLOPs = 0
MatMul_FLOPs = 0
for block in block_list:
    if block["type"] == "Conv":
        Overall_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["kernel_size"]*block["kernel_size"]*block["input_C"]
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
    
print("FLOPs of PWConv are: ", PW_FLOPs/1e6)
print("FLOPs of DWConv are: ", DW_FLOPs/1e6)
print("FLOPs of MatMul are: ", MatMul_FLOPs/1e6)
print("FLOPs of Overall are: ", (PW_FLOPs+DW_FLOPs+MatMul_FLOPs+Overall_FLOPs)/1e6)

# ######################## HW Simulation ########################
# HW Config
# Multiplier-Adder-Tree: Input Broadcast
# PW: inner PE lane: Cin Paral/ among PE lane: Cout Paral
num_MAT_PE = 16
num_MAT_lane = 8
# Reconfig-PE: Support both self-acc and down-forward acc
# PW: inner PE lane: Cin Parall/ among PE lane: Cout Parall
# DW: inner PE lane: Cout Parall/ among PE lane: Output Pixel (feature map) Parall
num_Reg_PE = 16
num_Reg_lane = 8

total_MAT_PE = num_MAT_PE*num_MAT_lane
total_Reg_PE = num_Reg_PE*num_Reg_lane
total_all_PE = total_MAT_PE + total_Reg_PE
PE_util = []
Flag = False
total_cycles = 0
for block in block_list:
    if block["type"] == "Conv":
        # use all PEs to simultaneously compute
        # Overall_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["kernel_size"]*block["kernel_size"]*block["input_C"]
        cycles = 0
        for _Cout in range(math.ceil(block["output_M"]/(num_Reg_lane+num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]/num_MAT_PE)):
               cycles += 1
        total_cycles += cycles * block["output_E"]*block["output_E"]*block["kernel_size"]*block["kernel_size"]
        PE_util.append((block["input_C"]*block["output_M"])/(cycles*total_all_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
            
              
    elif block["type"] == "DSConv":
        # Fuse DW and PW
        # DW_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*block["kernel_size"]*block["kernel_size"]
        cycles = 0
        for _fm in range(math.ceil(block["output_E"]/num_Reg_lane)):
            for _Cout in range(math.ceil(block["input_C"]/num_Reg_PE)):
               cycles += 1
        PE_util.append((block["output_E"]*block["input_C"])/(cycles*total_Reg_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
        DW_cycles = cycles * block["output_E"]*block["kernel_size"]*block["kernel_size"]
        setup_cycles = math.ceil(block["input_C"]/num_Reg_PE)*block["kernel_size"]*block["kernel_size"]
        
        # PW_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]
        cycles = 0
        for _Cout in range(math.ceil(block["output_M"]/(num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]/num_MAT_PE)):
               cycles += 1
            # The computation of DW ends, then use all PEs to compute PW
            if not cycles < (DW_cycles - setup_cycles):
                PE_util.append(1)
                Flag = True
                break
        if Flag == True:
            remaining_Cout = block["output_M"]-((_Cout+1)*num_MAT_lane)
            for _Cout in range(math.ceil(remaining_Cout/(num_MAT_lane+num_Reg_lane))):
                for _Cin in range(math.ceil(block["input_C"]/num_MAT_PE)):
                    cycles += 1   
            PE_util.append((block["input_C"]*remaining_Cout)/((_Cout+1)*(_Cin+1)*total_all_PE)) 
        else:
            PE_util.append((block["input_C"]*block["output_M"])/(cycles*total_MAT_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
        
        Flag = False 
        PW_cycles = cycles * block["output_E"]*block["output_E"]
        total_cycles += max(DW_cycles, PW_cycles + setup_cycles)
        
    elif block["type"] == "MBConv":
        # Use all PEs
        # PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*expand_ratio*block["input_C"]
        cycles = 0
        for _Cout in range(math.ceil(block["input_C"]*expand_ratio/(num_Reg_lane+num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]/num_MAT_PE)):
               cycles += 1
        total_cycles += cycles * block["input_H"]*block["input_H"]
        PE_util.append((block["input_C"]*expand_ratio*block["input_C"])/(cycles*total_all_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
        
        # Fuse DW and PW
        # DW_FLOPs += block["output_E"]*block["output_E"]*block["input_C"]*expand_ratio*block["kernel_size"]*block["kernel_size"]
        cycles = 0
        for _fm in range(math.ceil(block["output_E"]/num_Reg_lane)):
            for _Cout in range(math.ceil(block["input_C"]*expand_ratio/num_Reg_PE)):
               cycles += 1
        PE_util.append((block["output_E"]*block["input_C"]*expand_ratio)/(cycles*total_Reg_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
        DW_cycles = cycles * block["output_E"]*block["kernel_size"]*block["kernel_size"]
        setup_cycles = math.ceil(block["input_C"]*expand_ratio/num_Reg_PE)*block["kernel_size"]*block["kernel_size"]
        
        # PW_FLOPs += block["output_E"]*block["output_E"]*block["output_M"]*block["input_C"]*expand_ratio
        cycles = 0
        for _Cout in range(math.ceil(block["output_M"]/(num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]*expand_ratio/num_MAT_PE)):
               cycles += 1
            # The computation of DW ends, then use all PEs to compute PW
            if not cycles < (DW_cycles - setup_cycles):
                PE_util.append(1)
                Flag = True
                break
        if Flag == True:
            remaining_Cout = block["output_M"]-((_Cout+1)*num_MAT_lane)
            for _Cout in range(math.ceil(remaining_Cout/(num_MAT_lane+num_Reg_lane))):
                for _Cin in range(math.ceil(block["input_C"]*expand_ratio/num_MAT_PE)):
                    cycles += 1   
            PE_util.append((block["input_C"]*expand_ratio*remaining_Cout)/((_Cout+1)*(_Cin+1)*total_all_PE)) 
        else:
            PE_util.append((block["input_C"]*expand_ratio*block["output_M"])/(cycles*total_MAT_PE)) 
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization: ", PE_util[-1])
            print(block)
            print("")
            
        Flag = False 
        PW_cycles = cycles * block["output_E"]*block["output_E"]
        total_cycles += max(DW_cycles, PW_cycles + setup_cycles)
        
    elif block["type"] == "MSA":   
        # proj: use all PE
        # PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["input_C"]
        cycles = 0
        for _Cout in range(math.ceil(block["input_C"]*3/(num_Reg_lane+num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]/num_MAT_PE)):
               cycles += 1
        total_cycles += cycles * block["input_H"]*block["input_H"]
        PE_util.append((block["input_C"]*3*block["input_C"])/(cycles*total_all_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in Proj: ", PE_util[-1])
            print(block)
            print("")
            
        # aggregation: DW and PW Fusion
        # DW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["kernel_size"]*block["kernel_size"]
        cycles = 0
        for _fm in range(math.ceil(block["input_H"]/num_Reg_lane)):
            for _Cout in range(math.ceil(block["input_C"]*3/num_Reg_PE)):
               cycles += 1
        PE_util.append((block["input_H"]*block["input_C"]*3)/(cycles*total_Reg_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in DW: ", PE_util[-1])
            print(block)
            print("")
        DW_cycles = cycles * block["input_H"]*block["kernel_size"]*block["kernel_size"]
        setup_cycles = math.ceil(block["input_C"]*3/num_Reg_PE)*block["kernel_size"]*block["kernel_size"]
        
        # Overall_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*3*block["dim"]
        cycles = 0
        for _Cout in range(math.ceil(block["input_C"]*3/(num_MAT_lane))):
            for _Cin in range(math.ceil(block["dim"]/num_MAT_PE)):
               cycles += 1
            # The computation of DW ends, then use all PEs to compute PW
            if not cycles < (DW_cycles - setup_cycles):
                PE_util.append(1)
                Flag = True
                break
        if Flag == True:
            remaining_Cout = block["input_C"]*3-((_Cout+1)*num_MAT_lane)
            for _Cout in range(math.ceil(remaining_Cout/(num_MAT_lane+num_Reg_lane))):
                for _Cin in range(math.ceil(block["dim"]/num_MAT_PE)):
                    cycles += 1   
            PE_util.append((block["dim"]*remaining_Cout)/((_Cout+1)*(_Cin+1)*total_all_PE)) 
        else:
            PE_util.append((block["input_C"]*3*block["dim"])/(cycles*total_MAT_PE)) 
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in GConv: ", PE_util[-1])
            print(block)
            print("")
            
        Flag = False 
        PW_cycles = cycles * block["input_H"]*block["input_H"]
        total_cycles += max(DW_cycles, PW_cycles + setup_cycles)
        
        # MSA: use all PE
        # MSA:      FLOPs(A=K^T*V & Q*A)*num_head*2
        # MSA: 2*token_dim*feature_dim*feature_dim*num_head*2
        MatMul_FLOPs += 2*block["input_H"]*block["input_H"]*block["dim"]*block["dim"]*(block["input_C"]//block["dim"])*2
        cycles = 0
        # different heads map tp different PE Arrays
        # FIXME: low utilization
        for _head in range(((block["input_C"]//block["dim"])*2)//2):
            for _Cout in range(math.ceil(block["dim"]/num_Reg_lane)):
                for _Cin in range(math.ceil(block["input_H"]*block["input_H"]/num_MAT_PE)):
                    cycles += 1
        total_cycles += cycles * block["dim"]
        PE_util.append((block["dim"]*block["input_H"]*block["input_H"])/((_Cout+1)*(_Cin+1)*total_MAT_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in K^T*V: ", PE_util[-1])
            print(block["input_H"]*block["input_H"])
            print(block["dim"])
            print(block)
            print("")
            
        cycles = 0
        for _head in range(((block["input_C"]//block["dim"])*2)//2):
            for _Cout in range(math.ceil(block["dim"]/num_Reg_lane)):
                for _Cin in range(math.ceil(block["dim"]/num_MAT_PE)):
                    cycles += 1
        total_cycles += cycles * block["input_H"]*block["input_H"]
        PE_util.append((block["dim"]*block["dim"])/((_Cout+1)*(_Cin+1)*total_MAT_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in QA: ", PE_util[-1])
            print(block)
            print("")
            
        # proj: use all PE
        # PW_FLOPs += block["input_H"]*block["input_H"]*block["input_C"]*2*block["input_C"]
        cycles = 0
        for _Cout in range(math.ceil(block["input_C"]/(num_Reg_lane+num_MAT_lane))):
            for _Cin in range(math.ceil(block["input_C"]*2/num_MAT_PE)):
               cycles += 1
        total_cycles += cycles * block["input_H"]*block["input_H"]
        PE_util.append((block["input_C"]*block["input_C"]*2)/(cycles*total_all_PE))
        if PE_util[-1] < 0.75:
            print("*******Warning: Low utilization in last Proj: ", PE_util[-1])
            print(block)
            print("")
print("")        
print("PE Utilization is: ", PE_util)
print("Average PE Utilization is: ", mean(PE_util))
print("Overall Cycles is: ", total_cycles)
        
    