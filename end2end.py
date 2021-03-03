import os
import re
import json
import math
import random
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

def init():
    pd.set_option('display.max_colwidth', -1)

    # fix random seeds
    seed_value = 42 #@param {type:"integer"}

    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    tf.compat.v1.set_random_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    os.system("gcloud config set project feisty-mechanic-221914")

import wikipediaapi

def fetch_category(category="Tom Cruise filmography"):
    # Get all links in a wiki page
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(category)
    def get_links(page):
        links = page.links
        for title in sorted(links.keys()):
            if links[title].ns != 0 or "Unauthorized" in title:
                links.pop(title)
        return links

    links = get_links(page_py)

    pages_text = {}
    for title in links:
        page_py = links[title]
        pages_text[title] = page_py.text

    pages_text

    for title in pages_text:
        stop_index = pages_text[title].rfind("References")
        pages_text[title] = pages_text[title][:stop_index]

    return pages_text


import nltk
nltk.download('punkt')

def preprocess_ir(text):
    REPLACE_WITH_SPACE = re.compile(r"\n")
    text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]
    # we don't remove symbols, but just put a space before and after them. We did this because we noticed that Glove contains an embedding also for
    # them, so, in this way, we are able to split these symbols from the text when computing sentence tokens
    text = [re.sub(r"([(.;:!\'ˈ~?,\"(\[\])\\\/\-–\t```<>_#$€@%*+—°′″“”×’^₤₹‘])", r'', line) for line in text]
    # we noticed that in the text sometimes we find numbers and the following word merged together (ex: 1980february),
    # so we put a space between the number and the word
    text = [re.sub(r"(\d+)([a-z]+)", r'\1 \2', line) for line in text]
    text = [re.sub('\s{2,}', ' ', line.strip()) for line in text]   # replacing more than one consecutive blank spaces with only one of them
    return text

from sklearn.feature_extraction.text import TfidfVectorizer

def load_tfidf_squad_v1():
    os.system("gsutil cp gs://squad_squad/tfidf_squadv1.pkl ./tfidf_squadv1.pkl")
    with open('tfidf_squadv1.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_best_title(pages_vectorized, titles, question_vectorized):
    k=1
    tree = NearestNeighbors(n_neighbors=k, metric='cosine')
    tree.fit(pages_vectorized)

    results = tree.kneighbors(question_vectorized, n_neighbors=k, return_distance=False)

    titles = np.array(list(titles))
    title = titles[results[0]][0]
    return title

import transformers

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

import nltk.data
from transformers import AutoTokenizer

def preprocess_and_split_qa(tokenizer, pages_dict, max_seq_length):
    passages_dict = {}
    max_question_len = 20 # token

    for k in pages_dict.keys():
        context = pages_dict[k]
        tokenizer_nltk = nltk.data.load('tokenizers/punkt/english.pickle')
        context_sentences = tokenizer_nltk.tokenize(context)
        preprocessed_context = [" ".join(str(line).split()) for line in context_sentences]
        tokenized_sentences = [tokenizer(preprocessed_line, return_offsets_mapping=True).input_ids for preprocessed_line in preprocessed_context]
        sentence_index = 0
        tokenized_passages = []
        txt_passages = []

        while sentence_index < len(tokenized_sentences):
            start = sentence_index
            len_count = max_question_len
            while len_count <= max_seq_length and sentence_index < len(tokenized_sentences):
                len_count += len(tokenized_sentences[sentence_index])
                sentence_index += 1
            end = sentence_index -1
            tokenized_passages.append(tokenizer(" ".join(preprocessed_context[start:end]), return_offsets_mapping=True).input_ids)
            txt_passages.append(' '.join(preprocessed_context[start:end]))

        passages_dict[k] = tokenized_passages

    return passages_dict

def tokenize_question_qa(question, tokenizer):
    preprocessed_question = " ".join(str(question).split())
    tokenized_question = tokenizer(preprocessed_question, return_offsets_mapping=True).input_ids
    return tokenized_question

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


def main():
    init()
    print("[Debug] Initialized\n\n")

    pretrained_model_str = "roberta-base"
    max_seq_length = 512

    vectorizer = load_tfidf_squad_v1()
    qa_model = load_roberta_squadv1(pretrained_model_str, max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_str)
    print("[Debug] Model loaded\n\n")

    pages_dict = fetch_category()
    pages_titles = pages_dict.keys()
    pages_text = pages_dict.values()
    print("[Debug] Wikipedia resources fetched\n\n")

    pages_text_preprocessed = preprocess_ir(pages_text)
    pages_vectorized = vectorizer.transform(pages_text_preprocessed)
    passages_dict = preprocess_and_split_qa(tokenizer, pages_dict, max_seq_length)
    print("[Debug] Pages preprocessed\n\n")

    question = "Which aircraft drives Maverick?"
    question_vectorized = vectorizer.transform([question])
    question_tokenized = tokenize_question_qa(question, tokenizer)
    print("[Debug] Question encoded\n\n")

    best_title = get_best_title(pages_vectorized, pages_titles, question_vectorized)
    proposal_passages = passages_dict[best_title]
    print("[Debug] IR selected title:", best_title, "\n\n")
    answer = custom_inference(qa_model, tokenizer, proposal_passages, question_tokenized, max_seq_length)
    print("Answer:", answer, "\n\n")

main()
