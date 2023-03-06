# SETTINGS

# import os
# os.environ["PYTORCH_JIT"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["GOMP_CPU_AFFINITY"] = "0-31"

# import ctypes
# _libcudart = ctypes.CDLL('libcudart.so')
# # Set device limit on the current device
# # cudaLimitMaxL2FetchGranularity = 0x05
# pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
# _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
# _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
# assert pValue.contents.value == 128

# CODE

# import torch_performance_linter

import os
import cv2
import time
import torch
import warnings
import argparse

from data.preparation import prepare_data
from params import DATA_PATH, PRETRAINED_WEIGHTS
from utils.torch import init_distributed   # , map_cpu sync_across_gpus,
from utils.logger import create_logger, save_config, prepare_log_folder, init_neptune

cv2.setNumThreads(0)


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold number",
    )
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
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
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
#     img_folder = "yolox_1536_1024/"
    img_folder = "crops_512_512/"

    window = img_folder.endswith("_w/")
    aug_strength = 0
    resize = None

    use_cbis = False
    use_cmmd = False
    use_pasm = False
    use_pl = False

    # k-fold
    k = 4
    folds_file = "../input/folds_mpware.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "eca_nfnet_l2"  # "tf_efficientnetv2_s" "eca_nfnet_l1"
    pretrained_weights = "../logs/2023-02-16/5/"  # PRETRAINED_WEIGHTS.get(name, None)
    num_classes = 1
    num_classes_aux = 0
    n_channels = 3
    reduce_stride = False
    drop_rate = 0.1
    drop_path_rate = 0.1
    use_gem = True
    syncbn = False

    # Training
    loss_config = {
        "name": "bce",
        "smoothing": 0.0,
        "activation": "sigmoid",
        "aux_loss_weight": 0.,
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
        "name": "Ranger",
        "lr": 5e-5,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.0,
        "weight_decay": 0,  # 1e-2,
    }

    epochs = 5
    use_fp16 = True

    verbose = 1
    verbose_eval = 200

    fullfit = True
    n_fullfit = 1


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
        from params import LOG_PATH

        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH)

    if args.model:
        config.name = args.model
        config.loss_config["gmic"] = "gmic" in config.name
        if config.pretrained_weights is not None:
            config.pretrained_weights = PRETRAINED_WEIGHTS.get(args.model, None)

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    df = prepare_data(DATA_PATH, config.img_folder)

    try:
        print(torch_performance_linter)  # noqa
        if config.local_rank == 0:
            print("Using TPL\n")
        run = None
        config.epochs = 1
        log_folder = None
        df = df.head(10000)
    except Exception:
        run = None
        if config.local_rank == 0:
            run = init_neptune(Config, log_folder)

            if args.fold > -1:
                config.selected_folds = [args.fold]
                create_logger(directory=log_folder, name=f"logs_{args.fold}.txt")
            else:
                create_logger(directory=log_folder, name="logs.txt")

            save_config(config, log_folder + "config.json")

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model  {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    from training.main import k_fold

    k_fold(Config, df, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
