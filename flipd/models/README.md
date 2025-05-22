# Models
This directory contains network architectures, models, and training code.

This repo uses Hydra and MLFlow for running experiments. See `conf/train.yaml` for sample hyperparameters and model info. The subdirectories of `conf/` contain different configurations for different experiments:

1. `conf/callbacks/`: This directory contains all the different callbacks that can be added to the training loop. For example, `checkpoint.yaml` ensures that the model is saved and the checkpoints are stored. As another example, `sample_grid.yaml` is a callbacks that show an 8 by 8 grid of generated samples.
2. `conf/dataset/`: This directory contains all the different datasets that you can train your models on. For example, `cifar10.yaml` and `mnist.yaml` contain examples for RGB and greyscale image datasets and `lollipop.yaml` is a simple synthetic dataset with 2 columns with each row representing a point on a lollipop.
3. `conf/data_transforms/`: This directory contains all the different data transforms that you can perform before giving it to a generative model. For example, for training flows on image data, one should dequantize the image which is available in the `dequantize.yaml` file. On the other hand, for diffusion models one should normalize and zero center the data which is available in the `zero_center_normalize_greyscale.yaml` and `zero_center_normalize_rgb.yaml` file.
4. `conf/dgm/`: This directory contains the actual architecture specifications for the generative models that are being used alongside the default arguments used for sampling or estimating log_prob using them.
5. `conf/lightning_dgm/`: This directory contains the training module configurations, for example, the optimizer, the learning rate scheduler, etc.


## Test Runs
To test the training functionality without logging, try
```bash
python scripts/train.py dataset=cifar10 +experiment=train_diffusion_rgb dev_run=true train.trainer.callbacks=null
```
You can also simply get the configuration that is being used by running:
```bash
python scripts/train.py ... --help --resolve
```
## Preset Experiments

A number of preset experiments are available in the `conf/experiment/` directory. To run one of these, use the following command:
```bash
# to train a greyscale diffusion, run the following! You can for example replace the dataset argument with mnist or fmnist
python scripts/train.py dataset=<grayscale-data> +experiment=train_diffusion_greyscale
# to train an RGB diffusion, run the following! You can for example replace the dataset argument with cifar10
python scripts/train.py dataset=<rgb-data> +experiment=train_diffusion_rgb
# to train a greyscale flow, run the following! You can for example replace the dataset argument with mnist or fmnist
python scripts/train.py dataset=<grayscale-data> +experiment=train_flow_greyscale
# to train an RGB flow, run the following! You can for example replace the dataset argument with cifar10
python scripts/train.py dataset=<rgb-data> +experiment=train_flow_rgb
```

**NOTE**: Running through preset experiments will also automatically generate appropriate tags for your runs in Mlflow.

## Customizing Callbacks

You can choose to add any number of callbacks to the training loop that are already available in the [config/callbacks](../conf/callbacks/) directory.

```bash
python scripts/train.py ... +callbacks@all_callbacks.<callback-name>=<callback-yaml-file>
```
For example, to add a callback for visualizing the UMAP of generated vs. real data, and then also edit the callback to run after every single epoch, you can run the following:
```bash
python scripts/train.py +callbacks@all_callbacks.umap=umap all_callbacks.umap.frequency=1
```
You can even mix and match it with the experiments:
```bash
python scripts/train.py dataset=<grayscale-data> +experiment=train_flow_greyscale +callbacks@all_callbacks.umap=umap all_callbacks.umap.frequency=1 +tags.umap=true
```

## (Optional) Customizing Data Transforms 

**Note**: If you use the preset experiment configurations, there's no need to add data transforms manually; the preset configurations already include the necessary data transforms. With image flows being passed through a dequantization and logit transform, and image diffusions being passed through a zero-centering and normalization transform. However, if you want to specify your own data transforms, you can do so using the following instructions. The list of all the implemented transforms are available in the [conf/data_transforms](../conf/data_transforms/) directory. You can also choose to implement your own data transforms similar to the quantize/dequantize transforms that are available in [here](../conf/data_transforms/quantize_image.yaml) and [here](../conf/data_transforms/dequantize_image.yaml). 

Every dataset goes through a set of data transforms before being fed to the model which are all specified within the `all_data_transforms` attribute in the configurations. These transforms are ordered: the first transform is applied first, the second transform is applied next, and so on.
To ensure proper ordering, `all_data_transforms` contains keys of type `t{idx}` where `idx` is an integer that specifies the order of the transform. For example,

```yaml
all_data_transforms:
  t0: resize
  t1: to_tensor
  t2: zero_center_normalize_rgb
```
would apply the `resize` transform first, followed by the `to_tensor` transform, and finally the `zero_center_normalize_rgb` transform.

Similar to callbacks, you can add a new data transform to the training loop using the following command:
```bash
python scripts/train.py ... +all_data_transforms.clear=true +data_transforms@all_data_transforms.append.t<idx>=<data-transform-yaml-file> ...
```
This will clear up all the existing transforms and add the new transform at the specified index. The following command, for example, changes the transforms to `resize`, then `greyscale`, then `to_tensor`, then `dequantize`, followed by a `logit_transform`:
```bash
python scripts/train.py ... +all_data_transforms.clear=true +data_transforms@all_data_transforms.append.t0=resize +data_transforms@all_data_transforms.append.t1=greyscale +data_transforms@all_data_transforms.append.t2=dequantize  +data_transforms@all_data_transforms.append.t3=logit_transform ...
```

Apart from data transforms, we can also control the sampling transforms. These are transforms that are applied to the output of a generative model to convert it into a form that can be visualized. For example, for a diffusion model, we can apply a `clamp` transform to ensure that the output is in the range [0, 1]. Or for a flow model, you would have to perform a sigmoid transform (which is the inverse of the logit_transforms) to convert the output back to the range [0, 1]. The syntax of controlling sampling transforms is similar to that of data transforms, except for the fact that all of them are stored in `all_sampling_transforms` rather than `all_data_transforms`. The following command, for example, performs a sigmoid transform, and then a quantization transform, followed by a clamp:

```bash
python scripts/train.py ... +all_sampling_transforms.clear=true +data_transforms@all_sampling_transforms.append.t0=sigmoid_transform +data_transforms@all_sampling_transforms.append.t1=quantize_image +data_transforms@all_sampling_transforms.append.t2=clamp ...
```
You can also delete an already available transform using the `+all_data_transforms.delete=t<idx>` or `+all_sampling_transforms.delete=t<idx>` command. For example, if you want to train an mnist flow model but don't want to perform the logit transform for a greyscale flow, you can do the following:
```bash
python scripts/train.py dataset=mnist +experiment=train_flow_greyscale +all_data_transforms.delete=t4 +all_sampling_transforms.delete=t0
```
This is because originally the logit transform was the 4th transform in the data transforms and the 0th transform in the sampling transforms.

## Sweeps
To test the sweep functionality, try the following, which initiates a run for each of 2 different values of `layers_per_block`:

```bash
python scripts/train.py -m dev_run=true dataset=cifar10 +experiment=train_diffusion_rgb train.trainer.callbacks=null 'dgm.architecture.score_net.layers_per_block=1,2,3'
```


Sweeps can also be configured directly using configs; one example is the config `confs/hydra/cifar10_mem_sweep.yaml`:
```bash
python scripts/train.py hydra=cifar10_mem_sweep
```

## Tracking
Experiments with MLFlow outputs should generate a subdirectory in `./outputs`. To visualize these outputs, run `mlflow` in the `./outputs` directory:
```bash
cd outputs
mlflow ui
```

## Loading checkpoints (Optional)

Some configurations may use variables that are defined in your environment. For example, the directory in which your checkpoints are stored. To set these variables, we use dotenv. To run image diffusions using our checkpoints, you can run the following commands to store the checkpoint directories:

```bash
dotenv set DIFFUSION_MNIST_CHECKPOINT <your-mnist-checkpoint-dir>
dotenv set DIFFUSION_MNIST_CHECKPOINT_N_EPOCH <your-mnist-checkpoint-epoch> # set to: 455
dotenv set DIFFUSION_FMNIST_CHECKPOINT <your-fmnist-checkpoint-dir>
dotenv set DIFFUSION_FMNIST_CHECKPOINT_N_EPOCH <your-fmnist-checkpoint-epoch> # set to: 542
dotenv set DIFFUSION_CIFAR10_CHECKPOINT <your-cifar10-checkpoint-dir>
dotenv set DIFFUSION_CIFAR10_CHECKPOINT_N_EPOCH <your-cifar10-checkpoint-epoch> # set to: 458
dotenv set DIFFUSION_SVHN_CHECKPOINT <your-svhn-checkpoint-dir>
dotenv set DIFFUSION_SVHN_CHECKPOINT_N_EPOCH <your-svhn-checkpoint-epoch> # set to 612
```

**Note**: This only applies to the image complexity sweep for now and it is not needed for other experiments.