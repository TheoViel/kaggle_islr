import omegaconf
import sys
sys.path.append('../openhands')

import time
import torch
import torch.nn as nn
import pytorch_lightning as pl

from openhands.core.data import DataModule
from openhands.models.loader import get_model


class OpenHandModel(pl.LightningModule):
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)

        self.model = self.create_model(cfg.model)
        if stage == "test":
            self.model.to('cpu').eval()
    
    def create_model(self, cfg):
        return get_model(cfg, self.datamodule.in_channels, self.datamodule.num_class)
    
    def forward(self, x):
        return self.model(x)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu"), verbose=0):
        assert "pretrained" in self.cfg.keys()

        ckpt_path = self.cfg["pretrained"]
        if verbose:
            print(f"-> Loading weights from {ckpt_path}\n")

        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt
    
    def test_inference(self):
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        preds, truths = [], []
        for batch in tqdm(dataloader):
#             return batch, 0
            start_time = time.time()
            with torch.no_grad():
                y_hat = self.model(batch["frames"])
                preds.append(y_hat.detach().cpu().numpy())
                truths.append(batch['labels'].detach().cpu().numpy())
                
            break

        return np.concatenate(preds), np.concatenate(truths)
        

class SLGCN(nn.Module):
    def __init__(self, num_classes=None, cfg_path="../input/weights/st_gcn/config.yaml", verbose=0, **kwargs):
        super().__init__()
        self.num_classes_aux = 0
        self.num_classes = num_classes
        cfg = omegaconf.OmegaConf.load(cfg_path)
        model = OpenHandModel(cfg=cfg)
        model.init_from_checkpoint_if_available(verbose=verbose)

        self.encoder = model.model.encoder
        
        if num_classes is None:
            self.decoder = model.model.decoder
        else:
            self.decoder = nn.Linear(256, num_classes)
            
        self.ids = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
        
    def forward(self, data):
        # TODO : reformat properly
        x = torch.stack([data['x'], data['y']], 1)
        x = x.T[self.ids].T

        fts = self.encoder(x)  # expects BS x 2 x n_frames x 27
        y = self.decoder(fts)

        return y, torch.zeros(1)
        