
import os
import pickle

import numpy as np
from tensorflow import keras

import transformers
from transformers import AutoTokenizer

def load_roberta_squadv1(pretrained_model_str, max_seq_length):
    bert_hf_layer = transformers.TFRobertaModel.from_pretrained(
        pretrained_model_str, output_attentions=True)

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    sequence_output = bert_hf_layer(input_ids=input_word_ids, attention_mask=input_mask,
                                    token_type_ids=input_type_ids).last_hidden_state

    start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = tf.keras.layers.Flatten(name="flatten_start")(start_logits)

    end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = tf.keras.layers.Flatten(name="flatten_end")(end_logits)

    start_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name="softmax_start")(start_logits)
    end_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name="softmax_end")(end_logits)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                        outputs=[start_probs, end_probs],
                        name="BERT_QA")

    os.system("wget https://api.wandb.ai/files/buio/SQUAD/184b7gum/model-best.h5")
    model.load_weights("model-best.h5")
    return model

def custom_inference(model, tokenizer, tokenized_passages, tokenized_question, max_seq_length):
    prob = []
    candidate_ans = []
    for tokenized_passage in tokenized_passages:
        input_ids = tokenized_passage + tokenized_question[1:]
        token_type_ids = [0] * len(tokenized_passage) + [1] * len(tokenized_question[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        input_word_ids = np.array(input_ids)
        input_mask = np.array(attention_mask)
        input_type_ids = np.array(token_type_ids)
        predictions = model.predict([np.expand_dims(input_word_ids, axis =0),
                                    np.expand_dims(input_mask, axis = 0),
                                    np.expand_dims(input_type_ids,axis=0)])
        start, end = list(np.argmax(predictions, axis=-1).squeeze())
        if start > end:
            continue
        else:
            prob_start,prob_end = list(np.max(predictions, axis=-1).squeeze())
            prob_sum = prob_start+prob_end
            predicted_ans = tokenizer.decode(tokenized_passage[start : end+1])
            if predicted_ans != '':
                candidate_ans.append(predicted_ans)
                prob.append(prob_sum)

    print(*zip(prob, candidate_ans), sep='\n')
    ans = candidate_ans[np.argmax(prob)]
    return ans

class MyPredictor(object):

    def __init__(self, model, preprocessor):
        self.pretrained_model_str = "roberta-base"
        self.max_seq_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_str)
        self.model = load_roberta_squadv1(self.pretrained_model_str, self.max_seq_length)
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        print(instances)
        preprocessed_inputs = self._preprocessor.preprocess(instances)
        proposal_passages = self.preprocessed_inputs[0]
        tokenized_question = self.preprocessed_inputs[1]
        outputs = custom_inference(self.model, self.tokenizer, proposal_passages, tokenized_question, self.max_seq_length)
        return outputs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.h5')
        model = keras.models.load_model(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
