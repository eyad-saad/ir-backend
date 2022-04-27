"""information_retrieval URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os
import pdb

from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from information_retrieval import settings
from search_engine import views

from django.core.cache import cache
from information_retrieval.settings import BASE_DIR
import os
import docx2txt

from search_engine.ir_models.boolean_model import preprocess_text, create_index, get_document_vectors
from search_engine.ir_models.vector_model import wordDocFre, get_tf_idf

data_dir = os.path.join(BASE_DIR, 'data')
word_dict = {}
terms = []
doc_dict = {}
doc_frequency = {}


for file_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file_name)
    text = docx2txt.process(file_path)
    words = preprocess_text(text)
    word_dict[file_name] = words
    terms.extend(words)
    doc_dict[file_name] = text

df_dict = wordDocFre(terms, word_dict)
tf_idf = get_tf_idf(doc_dict, terms, df_dict)
terms = list(set(terms))

term_vectors = create_index(terms, word_dict)

cache.set('tf_idf', tf_idf, 99999999)
cache.set('doc_dict', doc_dict, 99999999)
cache.set('vocab', terms, 99999999)
cache.set('word_dict', word_dict, 99999999)
cache.set('term_vectors', term_vectors, 99999999)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('boolean-search', views.BooleanQuery.as_view()),
    path('set-folder', views.SetFolder.as_view()),
    path('view-document', views.ViewDocument.as_view()),
    path('vsm-search', views.VectorSpaceQuery.as_view()),
    path('extended-boolean-search', views.ExtendedBooleanQuery.as_view()),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
