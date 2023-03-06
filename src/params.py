# Hardcoded stuff, paths are to adapt to your setup
import numpy as np

NUM_WORKERS = 8

DATA_PATH = "../input/"
# TRAIN_DCM_PATH = "/raid/train_images/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["cancer"]
NUM_CLASSES = 250

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/RSNA-Breast-Cancer-Detection"
