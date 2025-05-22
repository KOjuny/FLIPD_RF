# Reproducing the FMNIST Experiments

1.  Create and activate the conda environment first:
    ```bash
    conda env create -f env.yaml
    conda activate dgm_geometry
    ```

2.  Train an FMNIST model:
    ```bash
    export CUDA_VISIBLE_DEVICES=0
    python scripts/train.py dataset=fmnist +experiment=train_diffusion_greyscale
    ```

3.  Monitor the training samples using mlflow:
    ```bash
    cd outputs
    mlflow ui
    ```
    Open up the mlflow UI using the link provided. Go to `Training` and then select the last run. Move to artifacts and monitor `sample_grids` as the samples are being generated. Whenever the samples look visually reasonable, you may stop the training and then access the checkpoint from the `lightning_logs/version_0/checkpoints/epoch=<epoch-num>-step=<step-num>.ckpt` directory.
4.  Now create (if not created) a `.env` file in the root directory and add the following line:
    ```bash
    DIFFUSION_FMNIST_MLP_CHECKPOINT_N_EPOCH=<epoch-num>+2 # add the epoch-num of the checkpoint you want to use with 2 (TODO: fix this)
    DIFFUSION_FMNIST_MLP_CHECKPOINT=<path-to-checkpoint>
    ```
    Replace `<path-to-checkpoint>` with the path to the checkpoint you just saved.
4.  Now run the following script which will create a run that contains the plots as artifact for 50 different values of the t_0 hyper-parameter ranging from `0` to `1.0`.
    ```bash
    # This setting will run in a faster setting where the number of Hutchinson samples are 50
    python scripts/train.py dataset=fmnist +experiment=train_diffusion_greyscale +checkpoint=diffusion_fmnist_mlp +callbacks@all_callbacks.lid=flipd_curve all_callbacks.lid.frequency=1 all_callbacks.lid.lid_estimation_args.method=hutchinson_gaussian
    # This setting will run in a slower setting where the trace term is computed detereministically
    python scripts/train.py dataset=fmnist +experiment=train_diffusion_greyscale +checkpoint=diffusion_fmnist_mlp +callbacks@all_callbacks.lid=flipd_curve all_callbacks.lid.frequency=1 all_callbacks.lid.lid_estimation_args.method=deterministic
    # increase the number of samples
    python scripts/train.py dataset=fmnist +experiment=train_diffusion_greyscale +checkpoint=diffusion_fmnist_mlp +callbacks@all_callbacks.lid=flipd_curve all_callbacks.lid.frequency=1 all_callbacks.lid.lid_estimation_args.subsample_size=4096
    ```
    For more fine-grained control over the run, please look at `conf/callbacks/flipd_curve.yaml` and change the arguments accordingly.
5.  After running these, you will get access to some new training runs on the mlflow logs. Go to the new run's artifacts and into the `FlipdEstimatorCurve/trends/` directory. There, you will find `trend_epoch=<num-epoch>.png` which will visualize the curve of the LID estimator w.r.t. the time hyperparameter. You can also get full access to each datapoint and it's correspond curve: all of the samples for which the curve is computed can be found in `FlipdEstimatorCurve/samples`, the range of which the hyperparameter has been sweeped on can be found in `FlipdEstimatorCurve/trends/sweeping_range.csv`, and finally, the raw value of the curve can be found in `trends/trend_epoch=<num-epoch>.csv` file: each row corresponds to a datapoint indexed in the `FlipdEstimatorCurve/samples` file and the columns correspond to the estimate evaluated at a specific value of the hyperparameter in the `FlipdEstimatorCurve/trends/sweeping_range.csv`.

