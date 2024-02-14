main_dir=Act3d_18Peract_100Demo_multitask

dataset=data/peract/Peract_packaged/train
valset=data/peract/Peract_packaged/val

lr=1e-4
num_ghost_points=1000
num_ghost_points_val=10000
B=8
C=120
ngpus=6

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_keypose.py \
    --tasks place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/peract/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --action_dim 8 \
    --use_instruction 1 \
    --weight_tying 1 \
    --gp_emb_tying 1 \
    --val_freq 4000 \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 1 \
    --cache_size 600 \
    --cache_size_val 0 \
    --variations {0..199} \
    --num_ghost_points $num_ghost_points\
    --num_ghost_points_val $num_ghost_points_val\
    --symmetric_rotation_loss 0 \
    --regress_position_offset 0 \
    --num_sampling_level 3 \
    --lr $lr\
    --position_loss_coeff 1 \
    --cameras left_shoulder right_shoulder wrist front\
    --max_episodes_per_task -1 \
    --run_log_dir act3d_multitask-C$C-B$B-lr$lr