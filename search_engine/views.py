import pdb

from django.core.cache import cache
from rest_framework.exceptions import APIException

from rest_framework.response import Response
from rest_framework.views import APIView

from information_retrieval.settings import BASE_DIR
import re
import os
import docx2txt

from search_engine.ir_models.boolean_model import preprocess_text, create_index, evaluate_query, \
    evaluate_extended_boolean
from search_engine.ir_models.vector_model import  vectorSpaceModel


def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex,'',text)
    return text_returned


class Search(APIView):
    def get(self, request):
        pass


class BooleanQuery(APIView):
    def get(self, request):
        query = request.GET.get('query')
        if not query:
            raise APIException("request must contain query")
        word_dict = cache.get('word_dict')
        term_vectors = cache.get('term_vectors')
        result = evaluate_query(query, word_dict, term_vectors)
        documents = word_dict.keys()
        resulting_documents = []
        for i, document_name in enumerate(documents):
            if result[i] == 1:
                resulting_documents.append({'document_name': document_name, 'similarity': result[i]})
        return Response({"result": resulting_documents})


class ExtendedBooleanQuery(APIView):
    def get(self, request):
        query = request.GET.get('query')
        if not query:
            raise APIException("request must contain query")
        word_dict = cache.get('word_dict')
        term_vectors = cache.get('term_vectors')
        result = evaluate_extended_boolean(query, word_dict, term_vectors)
        documents = word_dict.keys()
        resulting_documents = []
        for i, document_name in enumerate(documents):
            if result[i] != 0:
                resulting_documents.append({'document_name': document_name, 'similarity': result[i]})
        return Response({"result": resulting_documents})


class SetFolder(APIView):
    def post(self, request):
        data_dir = os.path.join(BASE_DIR, 'data')
        word_dict = {}
        terms = []
        doc_dict = {}

        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            text = docx2txt.process(file_path)
            words = preprocess_text(text)
            word_dict[file_name] = words
            terms.extend(words)
            doc_dict[file_name] = text

        terms = list(set(terms))

        term_vectors = create_index(terms, word_dict)

        cache.set('doc_dict', doc_dict)
        cache.set('vocab', terms)
        cache.set('word_dict', word_dict)
        cache.set('term_vectors', term_vectors)
        return Response({})


class ViewDocument(APIView):
    def post(self, request):
        if 'document_name' not in request.data:
            raise APIException("request body must have document_name")
        data_dir = os.path.join(BASE_DIR, 'data')
        document_name = request.data['document_name']
        try:
            file_path = os.path.join(data_dir, document_name)
            text = " ".join(docx2txt.process(file_path).split())
        except:
            raise APIException("file doesn't exist")
        return Response({'document_text': text})


class VectorSpaceQuery(APIView):
    def get(self, request):
        query = request.GET.get('query')
        if not query:
            raise APIException("request must contain query")
        docs = cache.get('doc_dict')
        tf_idf = cache.get('tf_idf')
        top = vectorSpaceModel(query, docs, tf_idf)
        result = []
        for key in top:
            result.append({'document_name': key, 'similarity': top[key]})
        return Response({'result': result})

