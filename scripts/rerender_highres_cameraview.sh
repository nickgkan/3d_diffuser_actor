#!/usr/bin/bash

SPLIT=train
DEMO_ROOT=./data/peract/raw/${SPLIT}
RAW_SAVE_PATH=./data/peract/raw_highres/${SPLIT}
PACKAGE_SAVE_PATH=./data/peract/packaged_highres/${SPLIT}

export PYTHONPATH=$(pwd):$PYTHONPATH

# Re-render high-resolution camera views
for task in place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
do
    python data_preprocessing/rerender_highres_rlbench.py \
        --tasks=$task \
        --save_path=$RAW_SAVE_PATH \
        --demo_path=$DEMO_ROOT \
        --image_size=256,256\
        --renderer=opengl \
        --processes=1 \
        --all_variations=True

done

# Re-arrange directory
python data_preprocessing/rearrange_rlbench_demos.py --root_dir $(pwd)/data/peract/raw_highres/${SPLIT}

# Package episodes into .dat fiels
for task in place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
do
    python data_preprocessing/package_rlbench.py \
        --data_dir=$RAW_SAVE_PATH \
        --tasks=$task \
        --output=$PACKAGE_SAVE_PATH \
        --store_intermediate_actions=1
done
