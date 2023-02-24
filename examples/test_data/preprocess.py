import json
import os
import random
from collections import Counter, defaultdict

import datasets
import numpy as np
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

import spacy
import unidecode
from spacy import displacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, HYPHENS, LIST_ELLIPSES, LIST_ICONS
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, minibatch
from transformers.models.gpt2.modeling_gpt2 import NYT_RULE


# customize spacy tokenizer
nlp = spacy.load("en_core_web_lg")

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        #         r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        #         r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        #         r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer


def load_data(src_dir=".", file_name="train.json"):
    raw_json = os.path.join(src_dir, file_name)
    file = open(raw_json, "r")
    sentences = file.readlines()
    return [json.loads(line) for line in sentences]


def process(exp, res, sentid, only_ent=False, only_re=False):
    exp_text = exp["sentText"]
    text_doc = nlp(exp_text)
    sent_res = []
    re_count = 0
    exp_tokens = [unidecode.unidecode(tok.text) for tok in text_doc if tok.text != "\r\n"]

    for ins_id, NER in enumerate(exp["entityMentions"]):
        query, ner_tag = NER["text"], NER["label"]
        query_doc = nlp(query)
        query_tokens = [tok.text for tok in query_doc]
        try:
            query_id = exp_tokens.index(query_tokens[0])
        except:
            query_id = exp_tokens.index(unidecode.unidecode(query_tokens[0]))

        tokens = np.array(["O"] * len(exp_tokens), dtype="object")
        tokens[query_id] = "B-" + ner_tag
        tokens[query_id + 1 : query_id + len(query_tokens)] = "I-" + ner_tag

        for RE in exp["relationMentions"]:
            target, re_tag = RE["em2Text"], RE["label"]
            if RE["em1Text"] == query and re_tag != "None":
                target_doc = nlp(target)
                target_tokens = [tok.text for tok in target_doc]
                try:
                    target_id = exp_tokens.index(target_tokens[0])
                except:
                    target_id = exp_tokens.index(unidecode.unidecode(target_tokens[0]))

                tokens[target_id] = "B-" + re_tag
                tokens[target_id + 1 : target_id + len(target_tokens)] = "I-" + re_tag
                re_count += 1

        ins_ID = len(res) + len(sent_res)
        sent_res.append(
            {
                "tokens": exp_tokens,
                "ner_tags": tokens.tolist(),
                "query_ids": query_id,
                "sentID": sentid,
                "instanceID": ins_ID,
            }
        )

    if only_ent and only_re:
        raise ValueError("Cannot do only entity and only relation at the same time!")

    if only_ent:
        if re_count == 0:  # not contain relations
            res += sent_res
    elif only_re:
        if re_count != 0:  # contain relations
            res += sent_res
    else:
        res += sent_res


def process_data(dataset, only_ent=False, only_re=False):
    res = []
    for i, instance in tqdm(enumerate(dataset)):
        try:
            process(instance, res, i, only_ent=only_ent, only_re=only_re)
        except:
            pass
    return res


def saving(f_path, res):
    with open(f_path, "w") as f:
        for value in res:
            f.write(json.dumps(value))
            f.write("\n")


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


# train_json = "new_train.json"
# # test_json = "new_test.json"
# test_json = "dev_part.json"
# train_file = 'joint_train_NYT_revised.json'
# test_file = 'new_joint_test_NYT.json'
# test_file = 'joint_test_NYT_revised.json'

# # check if processed train file/test file already exists
# if os.path.exists(train_file):
#     train_processed_data = load_dataset('json', data_files=train_file)['train']
# else:
#     train_data = load_data(file_name=train_json)
#     # remove repetition
#     unique_train = {each['sentText']: each for each in train_data}.values()
#     unique_train = list(unique_train)
#     # process data into query-level instances
#     train_processed_data = process_data(unique_train)
#     # saving processed file
#     saving(train_file, train_processed_data)

# if os.path.exists(test_file):
#     test_processed_data = load_dataset('json', data_files=test_file)['train']
# else:
#     test_data = load_data(file_name=test_json)
#     # remove repetition
#     unique_test = {each['sentText']: each for each in test_data}.values()
#     unique_test = list(unique_test)
#     # process data into query-level instances
#     test_processed_data = process_data(unique_test)
#     # saving processed file
#     saving(test_file, test_processed_data)

train_file_filtered = "train_NYT_preprocessed.json"
processed_data_filtered = load_dataset("json", data_files=train_file_filtered)
train_processed_data_filtered = processed_data_filtered["train"]


# # saving a sample train subset for quick training (according to sentID)
# if not isinstance(train_processed_data_filtered, datasets.Dataset):
#     train_processed_data = load_dataset('json', data_files=train_file_filtered)['train']

num_sent = len(set(train_processed_data_filtered["sentID"]))  # number of sentences
# 1/5, 1/4, 1/3, 1/2

for k in range(2, 6):  # half, triple, quarter, fitth split
    sentID_split = partition(list(range(num_sent)), k)  # split into k subtrain datasets
    ID_list = sentID_split[0]
    sub_train = train_processed_data_filtered.filter(
        lambda exp: exp["sentID"] in ID_list,
        num_proc=None,
        load_from_cache_file=False,
    )
    print("subtrain {} sent_num {} data size {}".format(k, len(ID_list), len(sub_train)))
    sub_train.to_json("train_NYT_1over{}_preprocessed.json".format(k))

# src_file = 'test_part.json'
# # trg_file = 'joint_test_NYT(entity).json'
# trg_file = 'joint_test_NYT(relation).json'

# test_data = load_data(file_name=src_file)
# # test_ent_data = process_data(test_data, only_ent=True)
# # saving(trg_file, test_ent_data)
# test_re_data = process_data(test_data, only_re=True)
# saving(trg_file, test_re_data)
