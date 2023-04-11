import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import torch
import warnings
import argparse
import pandas as pd

from data.preparation import prepare_data
from params import DATA_PATH, LOG_PATH
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
    processed_folder = "torch_11_wlasl/"
    max_len = 30
    resize_mode = "pad"
    aug_strength = 3
    use_extra_data = False
    n_landmarks = 100

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "mlp_bert_3"
#     name = "cnn_bert"
#     name = "bi_bert"
    pretrained_weights = None
    syncbn = False
    num_classes = 2000
    num_classes_aux = 0

    transfo_layers = 3
    embed_dim = 16
    dense_dim = 256
    transfo_dim = 1024  # 288
    transfo_heads = 16
    drop_rate = 0.05

    # Training
    loss_config = {
        "name": "ce",  # ce
        "smoothing": 0.3,
        "activation": "softmax",
        "aux_loss_weight": 0.,
        "activation_aux": "softmax",
    }

    data_config = {
        "batch_size": 32,
        "val_bs": 1024,
        "use_len_sampler": False,  # trimming is still slower, fix ?
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 2e-4,
        "warmup_prop": 0.1,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.,
    }

    epochs = 120

    use_fp16 = True
    model_soup = False

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

    log_folder = None
    if config.local_rank == 0:
        log_folder = prepare_log_folder(LOG_PATH  + "pretrain/")

    if args.model:
        config.name = args.model
        if config.pretrained_weights is not None:
            config.pretrained_weights = PRETRAINED_WEIGHTS.get(args.model, None)

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    
    run = None
    if config.local_rank == 0:
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

    from training.main import train

#     df = df.head(10000).reset_index(drop=True)
    df = pd.read_csv(DATA_PATH + 'df_wsasl.csv').sample(
        frac=1, random_state=config.seed
    ).reset_index(drop=True)

    df['processed_path'] = DATA_PATH + config.processed_folder + df['processed_path']

    train(config, df.head(len(df) - 1000), df.tail(1000), 0, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
