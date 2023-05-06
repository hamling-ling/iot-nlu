import sys
import datetime
import numpy as np
import unicodedata
import json

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# Local modules
from ner_tokenizer_bio import NER_tokenizer_BIO

import nluconfig as config
import demo_tool as tool


class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        print(severity, msg)


class IOVariables():
    def __init__(self):
        # Compute memory size for inputs/outputs
        nbytes_input_ids      = trt.volume((1,128)) * trt.int32.itemsize
        nbytes_attention_mask = trt.volume((1,128)) * trt.int32.itemsize
        nbytes_token_type_ids = trt.volume((1,128)) * trt.int32.itemsize
        nbytes_logits_intent  = trt.volume([1, config.NUM_INTENT_LABELS]) * trt.float32.itemsize
        nbytes_logits_slot    = trt.volume([1, 128, 17])                  * trt.float32.itemsize
        nbytes_total_loss     = trt.volume([1,1])                         * trt.float32.itemsize

        # memory allocation for inputs
        self.d_input_ids      = cuda.mem_alloc(nbytes_input_ids)
        self.d_attention_mask = cuda.mem_alloc(nbytes_attention_mask)
        self.d_token_type_ids = cuda.mem_alloc(nbytes_token_type_ids)

        # memory allocation for outputs
        self.d_logits_intent  = cuda.mem_alloc(nbytes_logits_intent)
        self.d_logits_slot    = cuda.mem_alloc(nbytes_logits_slot)
        self.d_total_loss     = cuda.mem_alloc(nbytes_total_loss)


def load_engine(runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


def infer(tokenizer, context, io_val, text):

    start_time = datetime.datetime.now()

    text = unicodedata.normalize('NFKC', text)
    encoding, spans = tokenizer.encode_plus_untagged(
        text, max_length=128
    )

    input_ids      = np.array(encoding["input_ids"],      np.int32)
    attention_mask = np.array(encoding["attention_mask"], np.int32)
    token_type_ids = np.array(encoding["token_type_ids"], np.int32)

    # Prepare output variables
    total_loss    = np.zeros([1,1],                         np.float32)
    logits_intent = np.zeros([1, config.NUM_INTENT_LABELS], np.float32)
    logits_slot   = np.zeros([1, 128, 17],                  np.float32)

    # Transfer input data from python buffers to device(GPU)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(io_val.d_input_ids,      input_ids,      stream)
    cuda.memcpy_htod_async(io_val.d_attention_mask, attention_mask, stream)
    cuda.memcpy_htod_async(io_val.d_token_type_ids, token_type_ids, stream)

    # Run the model
    bindings = [
        int(io_val.d_input_ids),     int(io_val.d_attention_mask), int(io_val.d_token_type_ids),
        int(io_val.d_logits_intent), int(io_val.d_logits_slot),    int(io_val.d_total_loss)
    ]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy output from GPU to host
    d_logits_intent = io_val.d_logits_intent
    d_logits_slot   = io_val.d_logits_slot
    d_total_loss    = io_val.d_total_loss

    cuda.memcpy_dtoh_async(logits_intent, io_val.d_logits_intent, stream)
    cuda.memcpy_dtoh_async(logits_slot,   io_val.d_logits_slot,   stream)
    cuda.memcpy_dtoh_async(total_loss,    io_val.d_total_loss,    stream)
    stream.synchronize()

    intent       = logits_intent.argmax(-1)[0]
    scores_slots = logits_slot[0]
    entities      = tokenizer.convert_bert_output_to_entities(text, scores_slots, spans)

    end_time = datetime.datetime.now()
    print(f"took {(end_time-start_time).total_seconds() * 1000} ms")

    return intent, entities


def input_loop(execution_context):

        # Load Tokenizer
    print(f"loading {config.MODEL_NAME}")
    tokenizer = NER_tokenizer_BIO.from_pretrained(
        config.MODEL_NAME,
        num_entity_type=config.NUM_ENTITY_TYPE
    )

    io_val = IOVariables()

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
        intent, entities = infer(tokenizer, execution_context, io_val, text)

        # Show result
        tool.print_output(text, intent, entities)


def main():
    with trt.Runtime(MyLogger()) as runtime:
        with load_engine(runtime, config.ENGINE_PATH) as engine:
            with engine.create_execution_context() as execution_context:
                input_loop(execution_context)



if __name__ == "__main__": 
    main()
 