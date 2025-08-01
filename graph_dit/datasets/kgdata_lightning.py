from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datasets.kgdata import TransductiveKnowledgeGraph



class TransductiveDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.datadir = cfg.dataset.datadir 
        self.batch_size =  cfg.train.batch_size 
        self.test_batch_size =  cfg.test.batch_size
        self.num_workers = cfg.dataset.num_workers
    
        self.data_object = TransductiveKnowledgeGraph(cfg)
        self.num_relation = self.data_object.num_relation
        
    def train_dataloader(self):
        return DataLoader(self.data_object.train_data, 
                          collate_fn=self.data_object.train_collate_fn,
                          shuffle=True, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_object.valid_data, 
                           collate_fn=self.data_object.valid_collate_fn,
                          shuffle=False, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_object.test_data, 
                           collate_fn=self.data_object.test_collate_fn,
                          shuffle=False, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
