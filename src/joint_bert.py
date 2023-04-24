import torch
from transformers import BertPreTrainedModel, BertModel, BertConfig, PreTrainedModel

from module import IntentClassifier, SlotClassifier

class JointBert(PreTrainedModel):
    def __init__(self, model_name, num_intent_labels, num_slot_labels):
        super().__init__(BertConfig())
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels   = num_slot_labels
        self.bert              = BertModel.from_pretrained(model_name)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(self.bert.config.hidden_size, self.num_intent_labels, 0.5)
        self.slot_classifier   = SlotClassifier(  self.bert.config.hidden_size, self.num_slot_labels,   0.5)
        self.slot_loss_coef    = 1.0

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label = None, slot_labels = None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # output[0]: sequence_output(last_hidden_state)
        # output[1]: pooled_output(pooler_output)
        # output[2]: hidden_states
        # output[3]: attentions
        sequence_output = outputs[0]
        pooled_output   = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits   = self.slot_classifier(sequence_output)

        total_loss = torch.tensor(0.0)

        # 1. Intent Softmax
        if intent_label is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = torch.nn.MSELoss()
                intent_loss     = intent_loss_fct(intent_logits.view(-1), intent_label.view(-1))
            else:
                intent_loss_fct = torch.nn.CrossEntropyLoss()
                intent_loss     = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.view(-1))
            total_loss = total_loss + intent_loss

        # 2. Slot Softmax
        if slot_labels is not None:
            slot_loss_fct = torch.nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss   = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss     = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))
            total_loss = total_loss + self.slot_loss_coef * slot_loss

        return total_loss, intent_logits, slot_logits

