import os
import cv2
import time
import torch
import warnings
import argparse

from params import LOG_PATH, DATA_PATH
from data.preparation import prepare_vindr_data
from utils.torch import init_distributed
from utils.logger import create_logger, save_config, prepare_log_folder

cv2.setNumThreads(0)


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """

    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Images
    img_folder = "vindr_yolox_1536_1024/"
    aug_strength = 2
    resize = None

    use_cbis = False
    use_cmmd = False
    use_pasm = False
    use_pl = False

    # k-fold
    k = 4
    folds_file = ""
    selected_folds = [0]

    # Model
    name = "nextvit_base"
    pretrained_weights = None
    num_classes = 5
    num_classes_aux = 0
    n_channels = 3
    reduce_stride = False
    drop_rate = 0.
    drop_path_rate = 0.
    use_gem = False
    syncbn = False

    # Training
    loss_config = {
        "name": "ce",
        "smoothing": 0.1,  # 0.01
        "activation": "softmax",  # "sigmoid", "softmax"
        "aux_loss_weight": 0.0,
        "pos_weight": None,
        "activation_aux": "softmax",
        "gmic": "gmic" in name,
    }

    data_config = {
        "batch_size": 8,
        "val_bs": 8,
        "mix": "mixup",
        "mix_proba": 0.0,
        "mix_alpha": 4.0,
        "additive_mix": False,
        "use_len_sampler": False,
        "use_balanced_sampler": False,
        "use_weighted_sampler": False,
        "sampler_weights": [1, 1, 1, 1],  # pos, birads 0, 1, 2
        "use_custom_collate": False,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 1e-4,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.0,
        "weight_decay": 0,  # 1e-2,
    }

    epochs = 5

    use_fp16 = True

    verbose = 1
    verbose_eval = 100

    fullfit = False


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = args.fold if args.fold > -1 else args.device
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    if not log_folder:
        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH + "pretrain/")

    if config.local_rank == 0:
        create_logger(directory=log_folder, name="logs.txt")
        save_config(config, log_folder + "config.json")

    if args.model:
        config.name = args.model
        config.loss_config['gmic'] = "gmic" in config.name

    if args.epochs:
        config.epochs = args.epochs

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    df = prepare_vindr_data(DATA_PATH, DATA_PATH + Config.img_folder)

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model  {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )

        print("\n -> Training\n")

    from training.main import k_fold

    k_fold(Config, df, log_folder=log_folder)

    if config.local_rank == 0:
        print("\nDone !")
