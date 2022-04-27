import numpy as np
from collections import OrderedDict


def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}

    for word in vocab:
        for doc_id, doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs


def wordDocFre(vocab, word_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in word_dict.values():
            if word in doc:
                frq = frq + 1
        df[word] = frq
    return df

def inverseDocFre(vocab,doc_fre,length):
    idf= {}
    for word in vocab:
        idf[word] = np.log2((length+1) / doc_fre[word])
    return idf


def tfidf(vocab,tf,idf_scr,doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr


def vectorSpaceModel(query, doc_dict, tfidf_scr):
    query_vocab = []
    for word in query.split():
        if word not in query_vocab:
            query_vocab.append(word)

    query_wc = {}
    for word in query_vocab:
        query_wc[word] = query.lower().split().count(word)

    relevance_scores = {}
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab:
            try:
                score += query_wc[word] * tfidf_scr[doc_id][word]
            except:
                pass
        relevance_scores[doc_id] = score
    sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    top_5 = {k: sorted_value[k] for k in list(sorted_value)[:5]}
    return top_5


def get_tf_idf(docs_dict, vocab, df_dict):
    number_of_documents = len(docs_dict)
    tf_dict = termFrequencyInDoc(vocab, docs_dict)
    idf_dict = inverseDocFre(vocab, df_dict, number_of_documents)

    tf_idf = tfidf(vocab, tf_dict, idf_dict, docs_dict)

    return tf_idf
