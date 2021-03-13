import os
import re
import json
import math
import random
import string
import pickle
import logging
logging.basicConfig(filename='/var/log/query_preprocessor/logs.txt', filemode='a', level=logging.INFO)
import numpy as np

import socket
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import nltk.data
from transformers import AutoTokenizer

def init():
    seed_value = 42 #@param {type:"integer"}
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    os.system("gcloud config set project feisty-mechanic-221914")
    nltk.download('punkt')

def load_pages():
    with open('/home/daniele.veri.96/pages.pkl', 'rb') as handle:
        pages_text = pickle.load(handle)
    return pages_text

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

def get_best_title(pages_vectorized, titles, question_vectorized):
    k=1
    tree = NearestNeighbors(n_neighbors=k, metric='cosine')
    tree.fit(pages_vectorized)

    results = tree.kneighbors(question_vectorized, n_neighbors=k, return_distance=False)

    titles = np.array(list(titles))
    title = titles[results[0]][0]
    return title

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

def mask_and_pad(tokenizer, tokenized_passages, tokenized_question, max_seq_length):
    instances=[]
    for tokenized_passage in tokenized_passages:
        input_ids = tokenized_passage + tokenized_question[1:]
        token_type_ids = [0] * len(tokenized_passage) + [1] * len(tokenized_question[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        instances.append({
            "input_word_ids": input_ids,
            "input_type_ids": token_type_ids,
            "input_mask": attention_mask
        })

    return instances

# env: GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
def discovery_service(project, region, model, version):
    socket.setdefaulttimeout(300)
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options, static_discovery=False)
    name = 'projects/{}/models/{}/versions/{}'.format(project, model,version)
    return service, name

def sample_predictions(predictions, tokenizer, tokenized_passage):
    candidate_ans = []
    confidence = []
    for i,p in enumerate(predictions):
        current_passage = tokenized_passage[i]
        softmax_start = np.array(p['softmax_start'])
        softmax_end = np.array(p['softmax_end'])
        start_idx = np.argmax(softmax_start)
        start_prob = np.max(softmax_start)
        end_idx = np.argmax(softmax_end)
        end_prob = np.max(softmax_end)

        if start_idx > end_idx:
            continue
        else:
            prob_sum = start_prob + end_prob
            predicted_ans = tokenizer.decode(current_passage[start_idx : end_idx+1])
            if predicted_ans != '' and predicted_ans != "<s>":
                candidate_ans.append(predicted_ans)
                confidence.append(prob_sum)
    if confidence == []:
        return "No answer"
    logging.info("Probabilities: {}".format(*zip(confidence, candidate_ans)))
    answer = candidate_ans[np.argmax(confidence)]
    return answer

class Preprocessor(object):
    def __init__(self):
        init()
        self.pretrained_model_str = "roberta-base"
        self.max_seq_length = 512
        self.vectorizer = TfidfVectorizer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_str)
        logging.info("Initialized\n\n")

        self.pages_dict = load_pages()
        self.pages_titles = self.pages_dict.keys()
        self.pages_text = self.pages_dict.values()
        logging.info("Wikipedia resources fetched\n\n")

        self.qa_mdl_service, self.selfservice_name = discovery_service("feisty-mechanic-221914", "us-central1", "cruiseNet", "v1")
        logging.info("QA model service found\n\n")

        self.pages_text_preprocessed = preprocess_ir(self.pages_text)
        self.pages_vectorized = self.vectorizer.fit_transform(self.pages_text_preprocessed)
        self.passages_dict = preprocess_and_split_qa(self.tokenizer, self.pages_dict, self.max_seq_length)
        logging.info("Pages preprocessed\n\n")

    def process(self, question):
        try:
            logging.info("Received question: {}".format(question))
            question_vectorized = self.vectorizer.transform([question])
            question_tokenized = tokenize_question_qa(question, self.tokenizer)
            best_title = get_best_title(self.pages_vectorized, self.pages_titles, question_vectorized)
            logging.info("IR model selected page: {}\n\n".format(best_title))
            proposal_passages = self.passages_dict[best_title]
            instances = mask_and_pad(self.tokenizer, proposal_passages, question_tokenized, self.max_seq_length)
            logging.info("Input ready, querying QA model ...\n\n")
            response = self.qa_mdl_service.projects().predict(name=self.selfservice_name, body={'instances': instances}).execute()  ### num_retries=3 ?
            logging.info("QA model response\n\n")
            if 'error' in response:
                raise Exception(response['error'])
            answer = sample_predictions(response['predictions'], self.tokenizer, proposal_passages)
            return answer
        except Exception as e:
            logging.exception(e)
            print(e)
            return "Unexpected error. Retry."
