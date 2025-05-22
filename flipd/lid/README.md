# LID estimation

LID estimation methods in this repository are either based on a DGM model, or they are traditional model-free estimators. We handle them separately, with the former being handled as a callback in the training loop, and the latter being handled as a standalone script. Here, we will briefly describe how to run each of them.

## Model-free LID Estimators

Common model-free LID estimators are lPCA, ESS, and MLE. We have also implemented our own closed-form diffusion model (CFDM) LID estimator in this repository. To run model-free estimators, setup your mlflow and run scripts in the following format:
```bash
python scripts/model_free_lid.py dataset=<dataset> lid_method=<lid-method> +experiment=<lid_greyscale | lid_rgb | lid_tabular> subsample_size=<subsample-size>
```
Explanation of the arguments:
1. `dataset`: The dataset to run the LID estimator on. The available datasets include `lollipop`, `swiss_roll`, `cifar10`, `mnist`, to name a few. We essentially cover everything in the [`conf/datasets`](../conf/dataset) directory.
2. `lid_method`: The LID estimator to run. The available LID estimators include `lpca`, `ess`, `mle`, and `cfdm`.
3. `experiment_dir`: The directory to save the experiment outputs. The available directories include `lid_greyscale`, `lid_rgb`, and `lid_tabular` for grayscale, RGB, and tabular datasets, respectively.
4. `subsample_size`: This performs a subsampling of the dataset to the specified size. This is useful for large datasets like CIFAR-10 and MNIST.

Example runs:
```bash
# Lollipop runs:
python scripts/model_free_lid.py dataset=lollipop lid_method=ess +experiment=lid_tabular subsample_size=10000
python scripts/model_free_lid.py dataset=lollipop lid_method=lpca +experiment=lid_tabular subsample_size=10000
python scripts/model_free_lid.py dataset=lollipop lid_method=cfdm +experiment=lid_tabular subsample_size=10000
# Image Greyscale runs
python scripts/model_free_lid.py dataset=mnist lid_method=ess +experiment=lid_greyscale subsample_size=4096
python scripts/model_free_lid.py dataset=fmnist lid_method=lpca +experiment=lid_greyscale subsample_size=4096
# Image RGB runs
python scripts/model_free_lid.py dataset=cifar10 lid_method=ess +experiment=lid_rgb subsample_size=4096
```

### How to interpret the output

You should first setup your Mlflow server. Then, you can visualize the outputs for each of these runs. The outputs include the following:
1. **Estimation heatmap**: For a holistic view, these scripts will log the UMAP embedding of all the dataset and for each point, they will log the estimated value of LID in the `lid_image/heatmap_pred.png` file in the artifacts and the true LIDs (if available) will be logged `lid_image/heatmap_gt.png` file in the artifacts. For example, the images will not have a ground truth files. You can also set the run to not log the heatmap by setting `visualize_manifold=null`.
2. **Detailed evaluation**: A detailed evaluation of the LID estimator is logged in the `results.yaml` file in the artifacts. This includes the mean absolute error, mean squared error, concordance index, and many other summary statistics to show how well the LID estimator performed. In cases where the ground truth is not available, you can check the average estimated LID values to see how well the LID estimator performed. You can also see how well the LID estimator performs on each individual submanifold for datasets with multiple known submanifolds (such as lollipop).
3. **Raw prediction**: To perform any additional analysis, the raw predictions are also logged in the `prediction.csv` file in the artifacts that you can download and analyze.

### How to tweak the lid method setting

The configurations above are the default settings for each of the LID estimators. However, you can tweak the settings just like you do with Hydra normally. For example, you can set the time hyperparameter of cfdm to another value that works better for the FMNIST dataset:
```bash
python scripts/model_free_lid.py dataset=fmnist lid_method=cfdm +experiment=lid_greyscale subsample_size=10000 lid_method.estimation_args.t=0.43
```
For a more detailed look, you can take a look at the appropriate configurations in the [`conf/`](../conf) directory.

## Model-based LID Estimators

Model-based LID estimators are based on a DGM model. Therefore, to monitor their performance we have implemented appropriate callbacks that you can be hooked on to the training loop and run the training script. The available mode-based LID estimators include the normal bundles estimator, LIDL, our Fokker-Planck estimator with and without regression. 

All of these callbacks has an argument called `all_callbacks.monitor_lid.subsample_size` that you can set to a number to see the samples. These samples are reproducible and they are made to monitor the LID for a specific subset of the dataset if you choose not to monitor the entire dataset. You can also set a `frequency` argument to control how often you want to monitor the LID during training.

Before we dive into the details of each estimator, let's first cover some of the basic things that *all* of these callbacks log:

1. **Data summary**: The callback logs the data summary in the `lid_logs_{estimator_name}/manifold_info.csv` file in the artifacts directory. This will include all the subsampled datapoints information. If for example, these are from synthetic data where the manifold structure is known, then the columns `lid` and `submanifold` will contain the true LID values and the submanifold labels respectively. If not, then these columns will be filled by `-1` and `0` respectively; this for example happens with image datasets.

2. **Seeing samples**:  All the samples will be logged in `lid_logs_{estimator_name}/samples/` directory. It will be either a `.csv` file if the data is tabular and a set of `.png` and `.npy` files if the data is an image. In addition to that, each sample has a transformed and an untransformed version. This is because the DGM also has a set of post-hoc transformations that are applied after the DGM generates the samples.

3. **Evaluating**: In all of the monitoring callbacks where a csv file is stored that contains information about all the datapoints, the index of the data (the row it resides in) is consistent with the row in the `manifold_info.csv`. Thus, you can join tables and evaluate the performance of LID estimators on a per-sample basis if you want to.

Now let's dive into each individual estimator:

### Normal Bundles Estimator

This estimator is based on a study by [Stanczuk et al.](https://arxiv.org/abs/2212.12611) where the score function is sampled around the manifold and the LID is estimated based on the normal bundles of the manifold. To run this estimator, for example on the lollipop dataset, run the following command:

```bash
# you can change the frequency to control how often during training you want to monitor LID
python scripts/train.py dataset=lollipop +experiment=train_diffusion_tabular +callbacks@all_callbacks.lid=normal_bundle_lid_curve all_callbacks.lid.frequency=1
```


*Note*: This estimator takes long to calculate as it requires computing SVD of matrices.

Additional outputs include:

**The trend of LID**: After computing the SVD decomposition, a threshold is applied to the singular values to estimate the LID. The trend of the LID is logged in the `lid_logs_{estimator_name}/trends/trend_lid_epoch={epoch_idx}.csv` and `lid_logs_{estimator_name}/trends/trend_lid_epoch={epoch_idx}.png` which shows how changing the threshold changes the LID. These trends start off from ambient dimension and go down to 0 with (hopefully) a plateau in between which shows the intrinsic dimensionality of the manifold. To access the raw values of the thresholds, you can check `lid_logs_{estimator_name}/trends/sweeping_range.csv` and the `.csv` file.

### Fokker Planck Estimator

This is our novel LID estimator and it is the fastest among the LID estimators in the codebase. To run this estimator, for example on the lollipop dataset, run the following command:

```bash
python scripts/train.py dataset=lollipop +experiment=train_diffusion_tabular +callbacks@all_callbacks.lid=flipd_curve all_callbacks.lid.frequency=1
```

Additional outputs include:

**The trend of LID**: Similar to normal bundles, we also have a trend which is the FLIPD estimator evaluated at different values of `t` in `[0, 1]`. The trend of the LID is logged in the `lid_logs_{estimator_name}/trends/trend_lid_epoch={epoch_idx}.csv` and `lid_logs_{estimator_name}/trends/trend_lid_epoch={epoch_idx}.png` which shows how changing the evaluation value `t` changes the LID. These trends start off from ambient dimension and go down to 0 with (hopefully) a *kink* in between which shows the intrinsic dimensionality of the manifold. To access the raw values of the timesteps, you can check `lid_logs_{estimator_name}/trends/sweeping_range.csv` and the `.csv` file.

### LIDL Estimator

This is a method proposed by [Tempczyk et al.](https://arxiv.org/pdf/2206.14882). Unlike the other estimators, this one trains an ensemble of density estimation models (typically normalizing flows). Therefore, the LID monitoring for this estimator is a bit different. To run this estimator, for example on the lollipop dataset, run the following command to train an ensemble of 8 models:

```bash
python scripts/train.py dataset=lollipop +experiment=train_lidl_tabular dataset.train.size=4096 dataset.val.size=128 +callbacks@all_callbacks.umap=umap all_callbacks.umap.frequency=1
```
**Note**: Here, the umap callback will actually generate multiple umap embeddings for each of the models in the ensemble.

Additional outputs include:
1. **log likelihood of perturbed data**: Every once in a while, the callback will compute the log_prob of all the datapoints for the different models in the ensemble. This is logged in the `lid_logs_{estimator_name}/trends/likelihood_trend_epoch={epoch}.png` and `lid_logs_{estimator_name}/trends/likelihood_trend_epoch={epoch}.csv` file where the x-axis are the different noise scales (the logarithm of the standard deviation of the Gaussian) and each trend represents a datapoint with the `y-value` being the log_prob of the datapoint for the model associated with a specific noise scale.
2. **Regression**: The result of doing a regression on the log_prob of the datapoints is logged in the `lid_logs_{estimator_name}/predictions/estimates_{epoch}.csv`. As always, the row numbers are consistent with the ones in the `manifold_info.csv` file. Thus, you can use that to evaluate the performance of LIDL.

### Image datasets

Image datasets are also supported here. The only difference is that the samples are saved as images and numpy arrays. For example, to run the FLIPD estimator on the FMNIST dataset, run the following command:

```bash
# Set some of the LID estimation hyperaparameters to make the run extremely lightweight for the sake of the example
python scripts/train.py dataset=fmnist +experiment=train_diffusion_greyscale +callbacks@all_callbacks.lid=flipd_curve all_callbacks.lid.frequency=1 all_callbacks.lid.lid_estimation_args.method=hutchinson_gaussian +all_callbacks.lid.lid_estimation_args.hutchinson_sample_count=1
```
In the samples directory now there will be multiple images alongside their numpy array representations.

### Stacking the LID Estimator Callbacks

You can also stack the LID monitoring using different methods. This is ideal for comparing the performance of different LID estimators. For example, to stack the FLIPD and normal bundles, run the following command:

```bash
python scripts/train.py dataset=lollipop +experiment=train_diffusion_tabular +callbacks@all_callbacks.lid1=flipd_curve all_callbacks.lid1.frequency=1 +callbacks@all_callbacks.lid2=normal_bundle_lid_curve all_callbacks.lid2.frequency=1
```
Note that you cannot stack the LIDL estimator with other estimators as it requires a different setup.

## Tracking
Experiments with MLFlow outputs should generate a subdirectory in `./outputs`. To visualize these outputs, run `mlflow` in the `./outputs` directory:
```
cd outputs
mlflow ui
```
