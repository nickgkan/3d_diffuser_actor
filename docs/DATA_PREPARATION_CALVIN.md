# Prepare data on CALVIN

* Download the play demonstrations from [Calvin](https://github.com/mees/calvin) repo.
```
> cd calvin/dataset
> sh download_data.sh ABC
```

* Package the demonstrations for training
```
> python data_preprocessing/package_calvin.py --split training
> python data_preprocessing/package_calvin.py --split validation
```

### Expected directory layout
```
./calvin/dataset/task_ABC_D
                     |------- training/
                     |------- validation/

./data/calvin/packaged_ABC_D
                     |------- training/
                     |            |------- A+0/
                     |            |          |------- ann_1.dat
                     |            |          |------- ...
                     |            |
                     |            |------- B+0/
                     |            |------- C+0/
                     |
                     |------- validation/
                                  |------- D+0/
```
