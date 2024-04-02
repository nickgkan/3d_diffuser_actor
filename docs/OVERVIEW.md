# Model overview
In this code base, we provide our implementation of [3D Diffuser Actor](../model/trajectory_optimization/diffuser_actor.py) and [Act3D](../model/keypose_optimization/act3d.py).  We provide an overview of input and output of both models.

## Common input format
Both models take the following inputs:

1. `RGB observations`: a tensor of shape (batch_size, num_cameras, 3, H, W).  The pixel values are in the range of [0, 1]
2. `Point cloud observation`: a tensor of shape (batch_size, num_cameras, 3, H, W).
3. `Instruction encodings`: a tensor of shape (batch_size, max_instruction_length, C).  In this code base, the embedding dimension `C` is set to 512.






## 3D Diffuser Actor
3D Diffuser Actor is a diffusion model that takes proprioception history into account.  3D Diffuser Actor is flexible to predict either keyposes or trajectories.  This model uses continuous `6D` rotation representations by default.

### Additional inputs
* `curr_gripper`: a tensor of shape (batch_size, history_length, 7), where the last channel denotes xyz-action (3D) and quarternion (4D).
* `trajectory_mask`: a tensor of shape (batch_size, trajectory_length), which is only used to indicate the length of each trajectory.  To predict keyposes, we just need to set its shape to (batch_size, 1).
* `gt_trajectory`: a tensor of shape (batch_size, trajectory_length, 7), where the last channel denotes xyz-action (3D) and quarternion (4D).  The input is only used during training, you can safely set it to `None` otherwise.

### Output
The model returns the diffusion loss, when `run_inference=False`, otherwise, it returns pose trajectories of shape (batch_size, trajectory_length, 8) when `run_inference=True`.

### Usage 
For training, forward 3D Diffuser Actor with `run_inference=False`
```
> loss = model.forward(gt_trajectory,
                       trajectory_mask,
                       rgb_obs,
                       pcd_obs,
                       instruction,
                       curr_gripper,
                       run_inference=False)
```

For evaluation, forward 3D Diffuser Actor with `run_inference=True`
```
> fake_gt_trajectory = None
> trajectory_mask = torch.full((1, trajectory_length), False).to(device)
> trajectory = model.forward(fake_gt_trajectory,
                             trajectory_mask,
                             rgb_obs,
                             pcd_obs,
                             instruction,
                             curr_gripper,
                             run_inference=True)
```


## Act3D
Act3D does not consider proprioception history and only predicts keyposes.   The model uses `quarternion` as the rotation representation by default.

### Input
* `curr_gripper`: a tensor of shape (batch_size, 8), where the last channel denotes xyz-action (3D), quarternion (4D), and end-effector openess (1D). 

### Output

* `position`: a tensor of shape (batch_size, 3)
* `rotation`: a tensor of shape (batch_size, 4)
* `gripper`: a tensor of shape (batch_size, 1)

### Usage 
Forward Act3D, the model returns a dictionary

```
> out_dict = Act3D.forward(rgb_obs,
                           pcd_obs,
                           instruction,
                           curr_gripper)
```

