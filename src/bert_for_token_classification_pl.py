import torch
import pytorch_lightning as pl

from ner_tokenizer_bio import NER_tokenizer_BIO
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
        output = self.bert_tc(**batch)
        loss = output[0]
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output[0]
        self.log('val_loss', val_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
