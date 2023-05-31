import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from .imeshsegnet import iMeshSegNet
from collections import OrderedDict

class LitModule(pl.LightningModule):
    def __init__(self,cfg: OmegaConf):
        super(LitModule, self).__init__()
        self.cfg = cfg
        self.cfg_train = cfg.train
        self.cfg_model = cfg.model
        
        self.model = iMeshSegNet(num_classes=self.cfg_model.num_classes, 
                                num_channels=self.cfg_model.num_channels, 
                                with_dropout=self.cfg_model.with_dropout, 
                                dropout_p=self.cfg_model.dropout_p)
    
        self.hparams.learning_rate = cfg.train.learning_rate
        self.save_hyperparameters()
        
    def forward(self, X):
        outputs = self.model(X['cells'], X['KG_12'], X['KG_6'])
        return outputs
        
    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            name = k[6:] # remove the 'model.' in ckpt['state_dict']['model.*']
            new_state_dict[name] = v
        epoch, global_step = ckpt['epoch'], ckpt['global_step']
        print(f'loading checkpoint with epoch:{epoch} and global step: {global_step}')
        self.model.load_state_dict(new_state_dict, strict=True)