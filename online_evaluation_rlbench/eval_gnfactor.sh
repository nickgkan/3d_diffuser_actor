exp=3d_diffuser_actor_gnfactor

tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
)
data_dir=./data/peract/raw/test/
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
checkpoint=train_logs/diffuser_actor_gnfactor.pth
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 20 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.08 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1
done

