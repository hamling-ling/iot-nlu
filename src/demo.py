import sys
import datetime

import torch

# Local modules
from ner_tokenizer_bio import NER_tokenizer_BIO
from bert_for_token_classification_pl import BertForTokenClassification_pl

import nluconfig as config
import demo_tool as tool


def infer(bert, tokenizer, text):
    start_time = datetime.datetime.now()
    encoding, spans = tokenizer.encode_plus_untagged(
        text, max_length=128, return_tensors='pt'
    )
    encoding = { k: v.cuda() for k, v in encoding.items() } 

    with torch.no_grad():
        total_loss, logits_intent, logits_slot = bert(**encoding)
        scores_intent = logits_intent.cpu().numpy()
        scores_slots  = logits_slot[0].cpu().numpy().tolist()

    # Intent 分類スコアを Intent に変換する
    intent = scores_intent.argmax(-1)[0]
    # Slot 分類スコアを固有表現に変換する
    entities = tokenizer.convert_bert_output_to_entities(
        text, scores_slots, spans
    )

    end_time = datetime.datetime.now()
    print(f"took {(end_time-start_time).total_seconds() * 1000} ms")

    return intent, entities


def main():
    # Load model
    print(f"loading {config.BEST_MODEL_PATH}")
    model   = BertForTokenClassification_pl.load_from_checkpoint(config.BEST_MODEL_PATH)
    bert_tc = model.bert_tc.cuda()

    # Load Tokenizer
    print(f"loading {config.MODEL_NAME}")
    tokenizer = NER_tokenizer_BIO.from_pretrained(
        config.MODEL_NAME,
        num_entity_type=config.NUM_ENTITY_TYPE
    )

    print("type \'q\' to exit")
    print('input> ', end='', flush=True)
    for line in sys.stdin:
        text = line.rstrip()
        if text in ['q', 'quit', 'exit', 'bye']:
            print('exit')
            return
        if len(text) == 0:
            continue
        
        # Infer
        intent, entities = infer(bert_tc, tokenizer, text)

        # Show result
        tool.print_output(text, intent, entities)


if __name__ == "__main__": 
    main()
 