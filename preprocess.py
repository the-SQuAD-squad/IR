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


class ZeroCenterer(object):

    def __init__(self):
        init()
        print("[Debug] Initialized\n\n")
        self.pretrained_model_str = "roberta-base"
        self.max_seq_length = 512
        self.vectorizer = load_tfidf_squad_v1()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_str)
        print("[Debug] Model loaded\n\n")

        self.pages_dict = fetch_category()
        self.pages_titles = self.pages_dict.keys()
        self.pages_text = self.pages_dict.values()
        print("[Debug] Wikipedia resources fetched\n\n")

        self.pages_text_preprocessed = preprocess_ir(self.pages_text)
        self.pages_vectorized = self.vectorizer.transform(self.pages_text_preprocessed)
        self.passages_dict = preprocess_and_split_qa(self.tokenizer, self.pages_dict, self.max_seq_length)
        print("[Debug] Pages preprocessed\n\n")


    def preprocess(self, question):
        print(question)
        question_vectorized = self.vectorizer.transform([question])
        question_tokenized = tokenize_question_qa(question, self.tokenizer)
        print("[Debug] Question encoded\n\n")

        best_title = get_best_title(self.pages_vectorized, self.pages_titles, question_vectorized)
        proposal_passages = self.passages_dict[best_title]
        return proposal_passages
