import torch
import pytorch_lightning as pl

from joint_bert import JointBert

class BertForTokenClassification_pl(pl.LightningModule):
    '''
    PyTorch Lightningのモデル
    '''
    def __init__(self, model_name, num_intent_labels, num_slot_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = JointBert(
            model_name,
            num_intent_labels = num_intent_labels,
            num_slot_labels   = num_slot_labels
        )
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.bert_tc(**batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.bert_tc(**batch)
        self.log('val_loss', val_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
