import os
import io
import sys
import pandas as pd
import copy
import numpy as np
import html2text
import matplotlib.pyplot as plt

import gensim

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn import manifold

g_seed = np.random.RandomState(seed=3)
g_source_path = 'data/gap-html/'
g_gen_data_path = 'data/gen/' 
g_html_converter = html2text.HTML2Text()
g_page_break_marker = 'JPW_PAGE_BRAKE'
g_doc_desription = {
    'gap_aLcWAAAAQAAJ':'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE By EDWARD GIBBON',
    'gap_dIkBAAAAQAAJ':'THE HISTORY OF ROME BY THE REV. WILLIAM P. DICKSON',
    'gap_MEoWAAAAYAAJ':'THE ANNALS OF TACITUS BY ARTHUR MURPHY',
    'gap_Bdw_AAAAYAAJ':'THE HISTORY OF ROME, BY TITUS LIVIUS 1797',
    'gap_TgpMAAAAYAAJ':'ANTIQUITIES OF THE JEWS, THE LIFE OF JOSEPHUS, WRITTEN BY HIMSELF',
    'gap_m_6B1DkImIoC':'TITUS LIVIUS ROMAN HISTORY - TRANSLATED INTO ENGLISH BY WILLIAM GORDON',
    'gap_2X5KAAAAYAAJ':'THE WORKS OF CORNELIUS TACITUS BY ARTHUR MURPHY',
    'gap_DhULAAAAYAAJ':'THE DESCRIPTION OF GREECE, BY PAUS ANIAS',
    'gap_fnAMAAAAYAAJ':'THE HISTORY OF THE PELOPONNESIAN WAR, BY THUCYDIDES',
    'gap_-C0BAAAAQAAJ':'DICTIONARY GREEK AND ROMAN GEOGRAPHY BY WILLIAM SMITH',
    'gap_XmqHlMECi6kC':'THE HISTORY DECLINE AND FALL ROMAN EMPIRE. BY EDWARD GIBBON, ESQ',
    'gap_y-AvAAAAYAAJ':'THE PARKMAN COLLECTION. BEQUEATHED BY Fran',
    'gap_CnnUAAAAMAAJ':'THE WHOLE GENUINE WORKS OF FLAVIUS JOSEPHUS',
    'gap_pX5KAAAAYAAJ':'THE WORKS OF CORNELIUS TACITUS 1805',
    'gap_9ksIAAAAQAAJ':'THE HISTORY OF THE PELOPONNESIAN WAR, TRANSLATED FROM THE GREEK',
    'gap_IlUMAQAAMAAJ':'GIBBONS HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE, By THOMAS BOWDLER - sensored',
    'gap_GIt0HMhqjRgC':'GIBBONS HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE By THOMAS BOWDLER 1826',
    'gap_ogsNAAAAIAAJ':'THE WORKS Of JOSEPH US, WITH A LIFE WRITTEN BY HIMSELF',
    'gap_DqQNAAAAYAAJ':'HISTORY OF ROME. TRANSLATED BY GEORGE BAKER',
    'gap_RqMNAAAAYAAJ':'HISTORY OF ROME, TRANSLATED BY GEORGE BAKER - THE FIFTH VOLUME',
    'gap_WORMAAAAYAAJ':'THE HISTORIES CAIUS COBNELIUS TACITUS: PROFESSOR OF LANGUAGES IN AMHERST COLLEGE',
    'gap_CSEUAAAAYAAJ':'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. BY EDWARD GIBBON',
    'gap_VPENAAAAQAAJ':'THE HISTORY OF THE ROMAN EMPIRE 1821',
    'gap_udEIAAAAQAAJ':'NATURAL HISTORY - THE WHOLE WORK, WITH NOTES, By JOHN BOSTOCK'
}

g_doc_auther = {
    'gap_aLcWAAAAQAAJ':'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE By EDWARD GIBBON',
    'gap_dIkBAAAAQAAJ':'THE HISTORY OF ROME BY THE REV. WILLIAM P. DICKSON',
    'gap_MEoWAAAAYAAJ':'THE ANNALS OF TACITUS BY ARTHUR MURPHY',
    'gap_Bdw_AAAAYAAJ':'THE HISTORY OF ROME, BY TITUS LIVIUS 1797',
    'gap_TgpMAAAAYAAJ':'ANTIQUITIES OF THE JEWS, THE LIFE OF JOSEPHUS, WRITTEN BY HIMSELF',
    'gap_m_6B1DkImIoC':'TITUS LIVIUS ROMAN HISTORY - TRANSLATED INTO ENGLISH BY WILLIAM GORDON',
    'gap_2X5KAAAAYAAJ':'THE WORKS OF CORNELIUS TACITUS BY ARTHUR MURPHY',
    'gap_DhULAAAAYAAJ':'THE DESCRIPTION OF GREECE, BY PAUS ANIAS',
    'gap_fnAMAAAAYAAJ':'THE HISTORY OF THE PELOPONNESIAN WAR, BY THUCYDIDES',
    'gap_-C0BAAAAQAAJ':'DICTIONARY GREEK AND ROMAN GEOGRAPHY BY WILLIAM SMITH',
    'gap_XmqHlMECi6kC':'THE HISTORY DECLINE AND FALL ROMAN EMPIRE. BY EDWARD GIBBON, ESQ',
    'gap_y-AvAAAAYAAJ':'THE PARKMAN COLLECTION. BEQUEATHED BY Fran',
    'gap_CnnUAAAAMAAJ':'THE WHOLE GENUINE WORKS OF FLAVIUS JOSEPHUS',
    'gap_pX5KAAAAYAAJ':'THE WORKS OF CORNELIUS TACITUS 1805',
    'gap_9ksIAAAAQAAJ':'THE HISTORY OF THE PELOPONNESIAN WAR, TRANSLATED FROM THE GREEK',
    'gap_IlUMAQAAMAAJ':'GIBBONS HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE, By THOMAS BOWDLER - sensored',
    'gap_GIt0HMhqjRgC':'GIBBONS HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE By THOMAS BOWDLER 1826',
    'gap_ogsNAAAAIAAJ':'THE WORKS Of JOSEPH US, WITH A LIFE WRITTEN BY HIMSELF',
    'gap_DqQNAAAAYAAJ':'HISTORY OF ROME. TRANSLATED BY GEORGE BAKER',
    'gap_RqMNAAAAYAAJ':'HISTORY OF ROME, TRANSLATED BY GEORGE BAKER - THE FIFTH VOLUME',
    'gap_WORMAAAAYAAJ':'THE HISTORIES CAIUS COBNELIUS TACITUS: PROFESSOR OF LANGUAGES IN AMHERST COLLEGE',
    'gap_CSEUAAAAYAAJ':'THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE. BY EDWARD GIBBON',
    'gap_VPENAAAAQAAJ':'THE HISTORY OF THE ROMAN EMPIRE 1821',
    'gap_udEIAAAAQAAJ':'NATURAL HISTORY - THE WHOLE WORK, WITH NOTES, By JOHN BOSTOCK'
}

g_doc_topic = {
    'gap_aLcWAAAAQAAJ':'FALL OF ROMAN',
    'gap_dIkBAAAAQAAJ':'HISTORY OF ROME',
    'gap_MEoWAAAAYAAJ':'ANNALS OF TACITUS',
    'gap_Bdw_AAAAYAAJ':'HISTORY OF ROME',
    'gap_TgpMAAAAYAAJ':'THE LIFE OF JOSEPHUS',
    'gap_m_6B1DkImIoC':'HISTORY OF ROMAN',
    'gap_2X5KAAAAYAAJ':'ANNALS OF TACITUS',
    'gap_DhULAAAAYAAJ':'GREECE',
    'gap_fnAMAAAAYAAJ':'HISTORY OF PELOPONNESIAN WAR',
    'gap_-C0BAAAAQAAJ':'DICTIONARY GREEK AND ROMAN GEOGRAPHY',
    'gap_XmqHlMECi6kC':'FALL OF ROMAN',
    'gap_y-AvAAAAYAAJ':'THE PARKMAN COLLECTION',
    'gap_CnnUAAAAMAAJ':'WORKS OF JOSEPHUS',
    'gap_pX5KAAAAYAAJ':'WORKS OF TACITUS',
    'gap_9ksIAAAAQAAJ':'HISTORY OF PELOPONNESIAN WAR',
    'gap_IlUMAQAAMAAJ':'FALL OF ROMAN',
    'gap_GIt0HMhqjRgC':'FALL OF ROMAN',
    'gap_ogsNAAAAIAAJ':'WORKS Of JOSEPH',
    'gap_DqQNAAAAYAAJ':'HISTORY OF ROME',
    'gap_RqMNAAAAYAAJ':'HISTORY OF ROME',
    'gap_WORMAAAAYAAJ':'HISTORY OF TACITUS',
    'gap_CSEUAAAAYAAJ':'FALL OF ROMAN',
    'gap_VPENAAAAQAAJ':'HISTORY OF ROMAN',
    'gap_udEIAAAAQAAJ':'NATURAL HISTORY'
}

g_tokenizer = RegexpTokenizer(r'\w+')
g_stopword_set = set(stopwords.words('english'))

def print_over(str):
    sys.stdout.write(str + '\r')
    sys.stdout.flush()

def clean_str(text):
    new_str = text.lower()
    ordered_list = g_tokenizer.tokenize(new_str)
    clipped_list = list(set(ordered_list).difference(g_stopword_set)) # side effect of changing word order
    clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
    return clipped_ordered_list

def read_page(page_path):
    with io.open(page_path, 'r', encoding='utf-8') as html_file:
        return g_html_converter.handle(html_file.read())

def read_text_doc(doc_path):
    pages = []
    with io.open(doc_path, 'r', encoding='utf-8') as text_file:
        page = ''
        for line in text_file:
            if g_page_break_marker in line:
                pages.append(page)
                page = ''
            else:
                page = page + line + '\n'
    return pages

def save_text_doc(pages, doc_path):
    tmp_path = doc_path + '.tmp'
    with io.open(tmp_path, 'w', encoding='utf-8') as text_file:
        for page in pages:
            text_file.write(page)
            text_file.write(g_page_break_marker + '\n')
    os.rename(tmp_path, doc_path)

def read_doc(doc_path):
    pages = []
    for filename in sorted(os.listdir(doc_path)):
        pages.append(read_page(doc_path + '/' + filename))
    return pages

def read_docs(docs_path, gen_path):
    os.makedirs(gen_path, exist_ok=True)
    docs = {}
    i = 0
    for doc_path, d1, d2 in os.walk(docs_path):
        if doc_path == docs_path:
            continue
        i = i + 1
        doc_folder = doc_path.split('/')[2]
        gen_doc_path = gen_path + doc_folder
        if os.path.exists(gen_doc_path):
            docs[doc_folder] = read_text_doc(gen_doc_path)
            print_over('Read '+ str(i) + ' ' + doc_folder)
        else:
            # return docs
            docs[doc_folder] = read_doc(doc_path)
            save_text_doc(docs[doc_folder], gen_doc_path)
            print_over('Converted ' + str(i) + ' ' + doc_path)
    print('Docs Loaded              ')
    return docs

def simple_doc_screen(docs):
    for key in docs:
        doc = docs[key]
        for page in doc:
            print(key + ' ' + page.replace('\n', '_'))
            command = input('s to skip')
            if 's' in command:
                break

def create_tagged_docs(docs):
    print_over('tagging doc : ')
    tagged_docs = []
    for doc_tag in docs:
        print_over('tagging doc : ' + doc_tag)
        i = 0
        doc = docs[doc_tag]
        for page in doc:
            i = i + 1
            page_tag = doc_tag + '_PAGE_' + str(i)
            cleaned_page = clean_str(page)
            tagged_docs.append(gensim.models.doc2vec.TaggedDocument(cleaned_page, [doc_tag, page_tag]))
    print('tagging doc : Done            ')
    return tagged_docs
    
def load_or_create_d2vm():
    d2vm = {}
    vector_size = 100
    vec_space_model_path = g_gen_data_path + 'doc2vec.model'
    if os.path.exists(vec_space_model_path):
        print('Loading doc2vec model')
        d2vm = gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        # create from scratch
        docs = read_docs(g_source_path, g_gen_data_path)
        tagged_docs = create_tagged_docs(docs)
        print('Creating doc2vec model')
        d2vm = gensim.models.Doc2Vec(vector_size=vector_size, min_count=0, alpha=0.025, min_alpha=0.025)
        d2vm.build_vocab(tagged_docs)
        print('    created vocab')
        d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=100)
        print('    trained model')
        # d2vm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        d2vm.save(vec_space_model_path)
    return d2vm

def find_most_similar(d2vm):
    for doc_tag in g_doc_desription:
        desc = g_doc_desription[doc_tag]
        print(doc_tag + ' : ' + desc)
        doc_vec = d2vm.docvecs[doc_tag]
        sim_docs = d2vm.docvecs.most_similar([doc_vec], topn=24)
        for sim_tag, likelyhood in sim_docs:
            sim_doc_tag = sim_tag[:16]
            print('   ' + '{:.2f}'.format(likelyhood) + ' ' + sim_tag + ' : ' + g_doc_desription[sim_doc_tag])

def get_matix(d2vm):
    tag_order = []
    matrix = np.empty((24, 100))
    i = 0;
    for doc_tag in g_doc_desription:
        tag_order.append(doc_tag)
        matrix[i, :] = d2vm.docvecs[doc_tag]
        i = i +1
    return tag_order, matrix

def find_hierachical(tag_order, matrix): 
    Z = linkage(matrix)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig(g_gen_data_path + 'Hierarchical')

def show_clusters(tag_order, matrix):
    similarities = euclidean_distances(matrix)
    
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=g_seed,
        dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    '''
    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=g_seed, n_jobs=1,
                    n_init=1)
    pos = nmds.fit_transform(similarities, init=pos)
    '''
    clf = PCA(n_components=2)
    pos = clf.fit_transform(pos)
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], color='turquoise')
    for i in range(len(tag_order)):
        ax.annotate(g_doc_topic[tag_order[i]], (pos[i, 0], pos[i, 1]))
    plt.show()
    

def main():
    d2vm = load_or_create_d2vm()
    #find_most_similar(d2vm)
    tag_order, matrix = get_matix(d2vm)
    find_hierachical(tag_order, matrix)
    show_clusters(tag_order, matrix)

main()