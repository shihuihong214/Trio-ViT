# CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --data_path /home/shared_data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.1 --n_bits_a 8 --act_quant --model b1-r224 --test_before_calibration

gpu=(0 3 2 3)
weight_bits=(4)
act_bits=(10 6 8 16)
weight=(0.5)
for i in 0 
do
    CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u main_imagenet.py --data_path /home/shared_data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 --test_before_calibration  --disable_8bit_head_stem  --n_bits_a 8 --act_quant > logs/MB/w8_a8_shifted_2 2>&1 &    
done

# /data/imagenet/

CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --data_path /data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.1