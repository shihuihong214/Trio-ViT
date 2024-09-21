# CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --data_path /home/shared_data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.1 --n_bits_a 8 --act_quant --model b3-r224 

# gpu=(0 1 2 3 3)
# weight_bits=(4)
# act_bits=(4 6 8 12 16)
# weight=(0.5)
# for i in 3
# do
#     CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u main_imagenet.py --data_path /data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r256  --disable_8bit_head_stem  --n_bits_a 8  --act_quant --input_size 288 --test_before_calibration > logs/B1/Sym_8bit_256 2>&1 &    
# done

# /data/imagenet/
# /home/shared_data/imagenet/

CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --data_path /data/imagenet/ --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --input_size 224 --test_before_calibration > logs/B1_New/testing_correctness 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python -m main_imagenet_dist --data_path /data/imagenet --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.5 --model b2-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --test_before_calibration --dist-url tcp://127.0.0.1:3458 --batch_size 32 --num_samples 512 --input_size 224  > logs/B2/224_CW 2>&1 &


# CUDA_VISIBLE_DEVICES=1,2,3 python -m main_imagenet_dist --data_path /data/imagenet --arch mobilenetv2 --n_bits_w 8 --channel_wise --weight 0.5 --model b2-r224 --disable_8bit_head_stem  --n_bits_a 8  --act_quant --test_before_calibration --dist-url tcp://127.0.0.1:3459