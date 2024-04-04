# Prepare data on RLBench

## PerAct setup

We use exactly the same train/test set on RLBench as [PerAct](https://github.com/peract/peract).  We re-render the same episodes with higher camera resolution ($256 \times 256$ compared $128 \times 128$ of PerAct).

### Prepare for testing episodes

1. Download the testing episodes from [PerAct](https://github.com/peract/peract?tab=readme-ov-file#pre-generated-datasets) repo.  Extract the zip files to `./data/peract/raw/test`
2. Rearrange episodes by their variantions
```
# For each task, separate episodes in all_variations/ to variations0/ ... variationsN/
> python data_preprocessing/rearrange_rlbench_demos.py --root_dir $(pwd)/data/peract/raw/test
```


### Prepare for training/validation episodes

1. Download our packaged demonstrations for training from [here](https://huggingface.co/katefgroup/3d_diffuser_actor/blob/main/Peract_packaged.zip).  Extract the zip file to `./data/peract/`


### Optional: Re-render training/validation episodes in higher resolution

1. Download the training/validation episodes from [PerAct](https://github.com/peract/peract?tab=readme-ov-file#pre-generated-datasets) repo.  Extract the zip files to `./data/peract/raw/train` or `./data/peract/raw/val`
2. Run this bashscript for re-rendering and packaging them into `.dat` files
```
# set SPLIT=train for training episodes
# set SPLIT=val for validation episodes
> bash scripts/rerender_highres_cameraview.sh
```


### Expected directory layout
```
./data/peract
         |------ raw/test
         |             |------ close_jar/
         |             |           |------ variation0/
         |             |           |            |------ variation_descriptions.pkl
         |             |           |            |------ episodes/
         |             |           |                      |------ episode0/
         |             |           |                      |         |------ low_dim_obs.pkl
         |             |           |                      |         |------ front_depth/
         |             |           |                      |         |------ front_rgb/
         |             |           |                      |         |------ wrist_depth/
         |             |           |                      |         |------ wrist_rgb/
         |             |           |                      |         |------ left_shoulder_depth/
         |             |           |                      |         |------ left_shoulder_rgb/
         |             |           |                      |         |------ right_shoulder_depth/
         |             |           |                      |         |------ right_shoulder_rgb/
         |             |           |                      |
         |             |           |                      |------ episode0/...
         |             |           |------ variation1/...
         |             |------ push_buttons/a
         |
         |------ Peract_packaged/
                      |------ train/
                      |          |------ close_jar+0/
                      |          |            |------ ep0.dat
                      |          |            |------ ep1.dat
                      |          |
                      |          |------ close_jar+0/...
                      |
                      |------ val/...
```
