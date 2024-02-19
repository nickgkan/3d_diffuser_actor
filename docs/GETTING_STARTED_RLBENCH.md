# Getting Started with RLBench

There are three simulation setups in RLBench: 1) [PerAct](https://github.com/peract/peract), 2) [GNFactor](https://github.com/YanjieZe/GNFactor), and 3) [Hiveformer](https://github.com/vlc-robot/hiveformer).  GNFactor uses exactly the same setup as PerAct.  Both have different succes conditions and 3D object models than Hiveformer.

Before training/testing on each setup, please install the RLBench library correspondingly.

## Train and evaluate on RLBench with the Peract/GNFactor setup

### Step 0: Prepare data on RLBench
See [Preparing RLBench dataset](./DATA_PREPARATION_RLBENCH.md)

### Step 1: Install RLBench with the PerAct setup
```
> git clone https://github.com/MohitShridhar/RLBench.git
> git checkout -b peract --track origin/peract
> pip install -r requirements.txt
> pip install -e .
```

Remember to modify the success condition of `close_jar` task in RLBench, as the original condition is incorrect.  See this [pull request](https://github.com/MohitShridhar/RLBench/pull/1) for more detail.  

### Step 2: Train the policy

* Train 3D Diffuser Actor with the PerAct setup

```
> bash scripts/train_keypose_peract.sh
```

* Train 3D Diffuser Actor with the GNFactor setup

```
> bash scripts/train_keypose_gnfactor.sh
```

We also provide training scripts for [Act3D](https://arxiv.org/abs/2306.17817).

* Train Act3D with the PerAct setup

```
> bash scripts/train_act3d_peract.sh
```

* Train Act3D with the GNFactor setup

```
> bash scripts/train_act3d_gnfactor.sh
```

### Step 3: Test the policy

* Test 3D Diffuser Actor with the PerAct setup

```
> bash online_evaluation_rlbench/eval_peract.sh
```

* Test 3D Diffuser Actor with the GNFactor setup

```
> bash online_evaluation_rlbench/eval_gnfactor.sh
```

We also provide testing scripts for [Act3D](https://arxiv.org/abs/2306.17817).

* Test Act3D with the PerAct setup

```
> bash online_evaluation_rlbench/eval_act3d_peract.sh
```

* Test Act3D with the GNFactor setup

```
> bash online_evaluation_rlbench/eval_act3d_gnfactor.sh
```
