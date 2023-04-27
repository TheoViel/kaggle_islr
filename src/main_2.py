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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import torch
import warnings
import argparse

from data.preparation import prepare_data, prepare_wsasl
from params import DATA_PATH
from utils.torch import init_distributed
from utils.logger import create_logger, save_config, prepare_log_folder, init_neptune


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
    parser.add_argument(
        "--mt-ema-decay",
        type=float,
        default=0,
        help="Mean teacher EMA decay",
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

    # Data
    processed_folder = "torch_12/"
    max_len = 25
    resize_mode = "pad"
    aug_strength = 3
    use_extra_data = False
    n_landmarks = 100

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "mlp_bert_3"  # mlp_bert_skip
    pretrained_weights = None  # "../logs/pretrain/2023-04-08/2/mlp_bert_3_0.pt"
    syncbn = False
    num_classes = 250
    num_classes_aux = 0

    transfo_layers = 3
    embed_dim = 16
    dense_dim = 256  # 192 256
    transfo_dim = 1024  # 768 1024
    transfo_heads = 16
    drop_rate = 0.05

    # Training
    loss_config = {
        "name": "ce",  # ce
        "smoothing": 0.3,
        "activation": "softmax",
        "aux_loss_weight": 0.,
        "activation_aux": "softmax",
        "ousm_k": 3,
        "use_embed": False,
    }

    data_config = {
        "batch_size": 32,
        "val_bs": 1024,
        "use_len_sampler": False,
        "mix_proba": 0.5,
        "mix_alpha": 0.4,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 3e-4,
        "warmup_prop": 0.25,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.,
        "weight_decay": 0.4,
#         # AWP
#         "use_awp": True,
#         "awp_start_step": 1,
#         "awp_lr": 1e-3,
#         "awp_eps": 1e-3,
#         "awp_period": 1,
    }

    mt_config = {
        "distill": True,
        "ema_decay": 0.97,
        "consistency_weight": 5,
        "rampup_prop": 0.25,
        "aux_loss_weight": 0.,
    }

    epochs = 100

    use_fp16 = True
    model_soup = True

    verbose = 1
    verbose_eval = 250

    fullfit = len(selected_folds) == 4
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

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr
        
    if args.mt_ema_decay:
        config.mt_config["ema_decay"] = args.mt_ema_decay

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    df = prepare_data(DATA_PATH, config.processed_folder)

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

        print(f"- Model {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    from training.main import k_fold

    df_extra = None
    if config.use_extra_data:
        df_extra = prepare_wsasl(DATA_PATH, config.processed_folder[:-1] + "_wlasl/")

    #     df = df.head(10000).reset_index(drop=True)

    k_fold(Config, df, df_extra=df_extra, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
