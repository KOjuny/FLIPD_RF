"""
This script automatically downloads all the resources and checkpoints that are required
for reproducing the results. 

We also set up the appropriate dotenv environment variables to a default value for the 
user that they can modify if they wish to later.
"""

import os
import shutil
from pathlib import Path

import dotenv
import gdown

if os.environ.get("IS_TESTING", False):
    from scripts import tools
else:
    import tools


def main(download_files: bool = True):
    """When download_files is set to False, it would avoid downloading everything and just proceed with some additional setup."""
    dotenv_file = dotenv.find_dotenv()
    # if the dotenv file does not exist, create it
    if dotenv_file == "":
        # make an empty dotenv file
        dotenv_file = os.path.join(os.getcwd(), ".env")
        with open(dotenv_file, "w") as f:
            f.write("")

    # create a copy of dotenv_file
    mex = 0
    while os.path.exists(f".env_backup{mex}"):
        mex += 1
    shutil.copy(dotenv_file, f".env_backup{mex}")

    dotenv.load_dotenv(dotenv_file, override=True)

    # get the current absolute path
    project_path = Path(os.getcwd()).absolute()

    # (1) download the checkpoints for the diffusion models and set the dotenv variables
    if download_files:
        gdown.download_folder(
            "https://drive.google.com/drive/u/1/folders/11jxRW5hPuM8mgMSzdER773PB4tv3orri",
            output="outputs/downloads/",
        )

    print("Setting environment variables for diffusion checkpoints ...")
    for checkpoint_key, checkpoint_path in [
        ("DIFFUSION_CIFAR10_CHECKPOINT", os.path.join("cifar10", "epoch=456-step=160864.ckpt")),
        ("DIFFUSION_SVHN_CHECKPOINT", os.path.join("svhn", "epoch=610-step=350103.ckpt")),
        ("DIFFUSION_FMNIST_CHECKPOINT", os.path.join("fmnist", "epoch=540-step=253729.ckpt")),
        ("DIFFUSION_MNIST_CHECKPOINT", os.path.join("mnist", "epoch=453-step=212926.ckpt")),
        (
            "DIFFUSION_CIFAR10_MLP_CHECKPOINT",
            os.path.join("cifar10-mlp", "epoch=1192-step=466463.ckpt"),
        ),
        ("DIFFUSION_SVHN_MLP_CHECKPOINT", os.path.join("svhn-mlp", "epoch=1045-step=509450.ckpt")),
        ("DIFFUSION_MNIST_MLP_CHECKPOINT", os.path.join("mnist-mlp", "epoch=780-step=366289.ckpt")),
        (
            "DIFFUSION_FMNIST_MLP_CHECKPOINT",
            os.path.join("fmnist-mlp", "epoch=995-step=467124.ckpt"),
        ),
    ]:
        os.environ[checkpoint_key] = os.path.join(
            project_path,
            "outputs",
            "downloads",
            "dgm-geometry",
            "checkpoints",
            "diffusions",
            checkpoint_path,
        )

        # check if the os.environ[checkpoint_key] file exists
        if not os.path.exists(os.environ[checkpoint_key]):
            raise FileNotFoundError(
                f"Could not find the checkpoint file at {os.environ[checkpoint_key]}"
            )

        dotenv.set_key(dotenv_file, checkpoint_key, os.environ[checkpoint_key])

        # parse the checkpoint path and find a epoch=?:
        epoch = int(checkpoint_path.split("epoch=")[1].split("-")[0]) + 2
        os.environ[checkpoint_key + "_N_EPOCH"] = str(epoch)
        dotenv.set_key(
            dotenv_file, checkpoint_key + "_N_EPOCH", os.environ[checkpoint_key + "_N_EPOCH"]
        )
    print("done!")

    # (2) download the flow-related checkpoints
    print("Setting environment variables for diffusion checkpoints ...")
    for checkpoint_key, checkpoint_path in [
        ("FLOW_MNIST_CHECKPOINT", os.path.join("mnist", "epoch=196-step=92393.ckpt")),
        ("FLOW_FMNIST_CHECKPOINT", os.path.join("fmnist", "epoch=216-step=101773.ckpt")),
    ]:
        os.environ[checkpoint_key] = os.path.join(
            project_path,
            "outputs",
            "downloads",
            "dgm-geometry",
            "checkpoints",
            "flows",
            checkpoint_path,
        )

        # check if the os.environ[checkpoint_key] file exists
        if not os.path.exists(os.environ[checkpoint_key]):
            raise FileNotFoundError(
                f"Could not find the checkpoint file at {os.environ[checkpoint_key]}"
            )

        dotenv.set_key(dotenv_file, checkpoint_key, os.environ[checkpoint_key])

        # parse the checkpoint path and find a epoch=?:
        epoch = int(checkpoint_path.split("epoch=")[1].split("-")[0]) + 2
        os.environ[checkpoint_key + "_N_EPOCH"] = str(epoch)
        dotenv.set_key(
            dotenv_file, checkpoint_key + "_N_EPOCH", os.environ[checkpoint_key + "_N_EPOCH"]
        )
    print("done!")


DISCLAIMER = """
[Disclaimer]
\tThis script will download all the resources and checkpoints required for the project, 
\tand will replace your existing .env file with the new one. We recommend you only run 
\tthis once to setup the project.

\tNote that this script will also backup your existing .env file to .env_backup{mex} 
\twhere mex is the number of backups already present in case you need to revert back.
"""

if __name__ == "__main__":
    # setting up root to be the root of the project
    tools.setup_root()
    print(DISCLAIMER)
    resp = input("Do you wish to continue? (y/n)\nYou can press 'n' to exit ... ")
    setup = resp.lower()[0] == "y"
    if not setup:
        print("Exiting ...")
        exit(0)
    print("Continuing ...")

    if not os.environ.get("IS_TESTING", False):
        resp = input(
            "Do you wish to download everything? (y/n)\nYou can press 'n' to skip download ... "
        )
        download_files = resp.lower()[0] == "y"
        main(download_files)
    else:
        main()
