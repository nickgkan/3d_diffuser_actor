main_dir=Actor_18Peract_20Demo_10GNFactortask

dataset=data/peract/Peract_packaged/train
valset=data/peract/Peract_packaged/val

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=8
C=120
ngpus=6
max_episodes_per_task=20
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --tasks close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/peract/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --gripper_loc_bounds_buffer 0.08 \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 4000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 600 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr\
    --num_history $num_history \
    --cameras front\
    --max_episodes_per_task $max_episodes_per_task \
    --quaternion_format $quaternion_format \
    --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps
