# Japanese pretrained model by Tohoku University
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# Path to best checkpoint file
BEST_MODEL_PATH='./model/epoch=4-step=660.ckpt'

# Path to TensorRT engine file
ENGINE_PATH    = "model/iot-nlu-sim-fp16.engine"

# インテントの種類数 (None=0, LED_ON=1, LED_OFF=2, READ_THERMO=3, OPEN=4, CLOSE=5, SET_TEMP=6)
NUM_INTENT_LABELS = 7
INTENT_MEANINGS = ['なし', '点灯したい', '消灯したい', '数値が知りたい', '開けたい', '閉めたい', '数値設定したい']

# スロットの種類数 (COL=1, COLLTDEV=2, LOC=3, ONOFFDEV=4, OPENABLE=5, TEMPDEV=6, TEMPERTURE_NUM=7, THMDEV=8)
NUM_ENTITY_TYPE = 8
ENTITY_MEANINGS = ['色', '照明', '設置場所', 'オンオフできる物', '開閉する物', '温度調節できる物', '温度', '温度計']
