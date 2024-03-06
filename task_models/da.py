from typing import Any
import lightning as pl





class DomainApatationModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = self.get_model(cfg)
        
    def debatch(self, batch):
        x, y = batch
        return x, y
    
    def general_step(self, batch, batch_idx, mode):
        x, y = self.debatch(batch)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f'{mode}/loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.cfg)
        return optimizer
    
    def get_model(self, cfg):
        raise NotImplementedError
    
    def get_optimizer(self, cfg):
        raise NotImplementedError