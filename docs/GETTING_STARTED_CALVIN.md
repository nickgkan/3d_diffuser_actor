# Getting Started with Calvin

### Step 0: Install CALVIN
```
> git clone --recurse-submodules https://github.com/mees/calvin.git
> export CALVIN_ROOT=$(pwd)/calvin
> cd calvin
> cd calvin_env; git checkout -b main --track origin/main
> cd ..
> ./install.sh
```

### Step 1: Prepare data on CALVIN
See [Preparing CALVIN dataset](./DATA_PREPARATION_CALVIN.md)

### Step 2: Train the policy

* Train and test 3D Diffuser Actor on CALVIN

```
> bash scripts/train_trajectory_calvin.sh
```
