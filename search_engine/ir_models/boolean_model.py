import re

import numpy as np
from nltk import word_tokenize
WORD_DICT = {}
TERM_VECTORS = []


def bool_representation(terms, word_dicts):
    bool_dict = {}
    for key in word_dicts:
        bool_list = [term in word_dicts[key] for term in terms]
        bool_dict[key] = bool_list
    return bool_dict


def clean_text(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()

    return text


def evaluate_expression(t1, t2, op):
    if op == "and" or op == "AND":
        return np.array(t1).astype(int) & np.array(t2).astype(int)
    if op == "or" or op == "AND":
        return np.array(t1).astype(int) | np.array(t2).astype(int)


def evaluate_query(query, word_dict, term_vectors):
    operators = ['and', 'or', 'not', "AND", "OR", "NOT"]

    modified_query = query.split(' ')

    for i, term in enumerate(modified_query):
        term = term.lower()

        if term in operators:
            continue

        elif term in term_vectors:
            modified_query[i] = np.array(term_vectors[term])

        elif term not in operators:
            modified_query[i] = len(word_dict) * [0]

        if term in term_vectors and i > 0 and (modified_query[i - 1] == "not" or modified_query[i - 1] == "NOT"):
            vector = np.array(term_vectors[term])
            modified_query[i] = np.where((vector == 0) | (vector == 1), vector ^ 1, vector)
            modified_query.pop(i - 1)
        if term not in term_vectors and i > 0 and (modified_query[i - 1] == "not" or modified_query[i - 1] == "NOT"):
            modified_query[i] = len(word_dict) * [1]
            modified_query.pop(i - 1)

    res = None
    current_op = None
    for element in modified_query:
        if element not in operators and res is None:
            res = element
        elif element not in operators and current_op is not None:
            res = evaluate_expression(element, res, current_op)
        elif element in operators:
            current_op = element
    return res


def preprocess_text(text):
    text = clean_text(text)
    words = word_tokenize(text)
    words = [word for word in words if len(word) > 1]
    return words


def create_index(terms, word_dict):
    term_vectors = {}
    for term in terms:
        term_vector = [int(term in word_dict[document_name]) for document_name in word_dict]
        term_vectors[term] =term_vector
    return term_vectors


def calculate_distance(t1, t2, op):
    if op == "and" or op == "AND":
        return 1 - ( np.sqrt((1 - np.array(t1).astype(int)) ** 2 + (1 - np.array(t2).astype(int)) ** 2) / 2 )
    if op == "or" or op == "AND":
        return np.sqrt((np.array(t1).astype(int)) ** 2 + (np.array(t2).astype(int)) ** 2) / 2


def get_document_vectors(terms, word_dict):
    document_vectors = {}
    for document_name in word_dict:
        document_vectors[document_name] = [int(term in word_dict[document_name]) for term in terms]
    return document_vectors


def evaluate_extended_boolean(query, word_dict, term_vectors):
    operators = ['and', 'or', 'not', "AND", "OR", "NOT"]

    modified_query = query.split(' ')

    for i, term in enumerate(modified_query):
        term = term.lower()

        if term in operators:
            continue

        elif term in term_vectors:
            modified_query[i] = np.array(term_vectors[term])

        elif term not in operators:
            modified_query[i] = len(word_dict) * [0]

        if term in term_vectors and i > 0 and (modified_query[i - 1] == "not" or modified_query[i - 1] == "NOT"):
            vector = np.array(term_vectors[term])
            modified_query[i] = np.where((vector == 0) | (vector == 1), vector ^ 1, vector)
            modified_query.pop(i - 1)
        if term not in term_vectors and i > 0 and (modified_query[i - 1] == "not" or modified_query[i - 1] == "NOT"):
            modified_query[i] = len(word_dict) * [1]
            modified_query.pop(i - 1)

    res = None
    current_op = None
    for element in modified_query:
        if element not in operators and res is None:
            res = element
        elif element not in operators and current_op is not None:
            res = calculate_distance(element, res, current_op)
        elif element in operators:
            current_op = element
    return res