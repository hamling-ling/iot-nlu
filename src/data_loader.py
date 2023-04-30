import random
import json
from tqdm import tqdm
import unicodedata
import numpy as np

from torch.utils.data import DataLoader
from ner_tokenizer_bio import NER_tokenizer_BIO

# tokenizer を保存した場所
TOKENIZER_PATH      = './model/iot-nlu-tokenizer'
# 最大シーケンス長
MAX_LENGTH = 128
# インテントの種類数 (None=0, LED_ON=1, LED_OFF=2, READ_THERMO=3, OPEN=4, CLOSE=5, SET_TEMP=6)
NUM_INTENT_LABELS = 7
# スロットの種類数 (COL=1, COLLTDEV=2, LOC=3, ONOFFDEV=4, OPENABLE=5, TEMPDEV=6, TEMPERTURE_NUM=7, THMDEV=8)
NUM_ENTITY_TYPE   = 8

class DataLoaderContext:
    
    def __init__(self, tokenizer):

        # データのロード
        dataset = json.load(open('data/nlp_data.json','r'))

        # カテゴリーをラベルに変更、文字列の正規化する。
        for sample in dataset:
            sample['text'] = unicodedata.normalize('NFKC', sample['text'])

        # データセットの分割
        random.shuffle(dataset)
        dataset = dataset[:10000]
        n       = len(dataset)
        n_train = int(n*0.6)
        n_val   = int(n*0.2)
        dataset_train = dataset[:n_train]
        dataset_val   = dataset[n_train:n_train+n_val]
        dataset_test  = dataset[n_train+n_val:]

        self.dataset_train = self.format_dataset(tokenizer, dataset_train, MAX_LENGTH)
        self.dataset_val   = self.format_dataset(tokenizer, dataset_val,   MAX_LENGTH)
        self.dataset_test  = self.format_dataset(tokenizer, dataset_test,  MAX_LENGTH)

        self.loader_train = DataLoader(self.dataset_train, batch_size=1)
        self.loader_val   = DataLoader(self.dataset_val,   batch_size=1)
        self.loader_test  = DataLoader(self.dataset_test,  batch_size=1)

    def format_dataset(self, tokenizer, dataset, max_length):
        """
        データセットをデータローダに入力できる形に整形。
        """
        dataset_for_loader = []
        for sample in dataset:
            text = sample['text']
            entities = sample['entities']
            encoding = tokenizer.encode_plus_tagged(
                text, entities, max_length=max_length
            )
            encoding['intent_label']   = sample['intent']
            encoding['input_ids']      = np.array(encoding['input_ids'], np.int32)
            encoding['attention_mask'] = np.array(encoding['attention_mask'], np.int32)
            encoding['token_type_ids'] = np.array(encoding['token_type_ids'], np.int32)

            encoding = { k: np.array(v, np.int32) for k, v in encoding.items() }
            dataset_for_loader.append(encoding)
        return dataset_for_loader


def load_data():
    tokenizer = NER_tokenizer_BIO.from_pretrained(
        TOKENIZER_PATH,
        num_entity_type=NUM_ENTITY_TYPE
    )

    ctx = DataLoaderContext(tokenizer)

    for encoding in ctx.loader_train:
        inputs = {
            "input_ids"      : encoding["input_ids"     ].numpy().astype(np.int32),
            "attention_mask" : encoding["attention_mask"].numpy().astype(np.int32),
            "token_type_ids" : encoding["token_type_ids"].numpy().astype(np.int32)
        }
        yield inputs
