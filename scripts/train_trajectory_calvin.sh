main_dir=Planner_Calvin

dataset=./data/calvin/packaged_ABC_D/training
valset=./data/calvin/packaged_ABC_D/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=3
diffusion_timesteps=25
B=30
C=192
ngpus=6
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz

run_log_dir=diffusion_taskABC_D-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-backbone$backbone-S$image_size-R$relative_action-wd$wd

export PYTHONPATH=`pwd`:$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory_calvin.py \
    --tasks A B C D\
    --backbone $backbone \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/calvin_task_ABC_D/ \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --image_size $image_size \
    --num_workers 4 \
    --max_episode_length 30 \
    --train_iters 600000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq $val_freq \
    --val_iters 2 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 3 \
    --cache_size 0 \
    --cache_size_val 0 \
    --keypose_only 0 \
    --variations {0..0} \
    --lr $lr\
    --wd $wd \
    --num_history $num_history \
    --cameras front wrist \
    --max_episodes_per_task -1 \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --quaternion_format $quaternion_format \
    --run_log_dir $run_log_dir


torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path calvin/dataset/task_ABC_D \
    --calvin_model_path calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds calvin/dataset/task_ABC_D/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/${run_log_dir}/eval_logs/ \
    --quaternion_format $quaternion_format \
    --checkpoint train_logs/${main_dir}/${run_log_dir}/last.pth
