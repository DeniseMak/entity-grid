#!/bin/python3

print("Starting Corpus Loader")

import sys
import re
import os
import string
from lxml import etree
from lxml import html
from lxml.etree import tostring
import gzip
import nltk
nltk.download('punkt')
import nltk.data
from nltk.corpus import stopwords
import glob
import pickle
import spacy
import numpy as np
import multiprocessing_on_dill as mp
import cleaner
print("Starting compressor import")
from sentence_compressor import *
from command_args import *
import datetime





print("Loaded Corpus Loader")
currentDT = datetime.datetime.now()
print (str(currentDT))
sys.stdout.flush()

project_root = os.path.join(os.path.dirname(__file__), '..')

class Sentence:

    def __init__(self, text, document, pos, comp=False):
        self.pos = pos
        self.document = document
        self.clean_text = cleaner.cleaner(text)
        self.text = text
        self.full_tokens = [w.lower().strip(string.punctuation) for w in self.clean_text.split()]
        self.tokens = [w for w in self.full_tokens if w and not w in stopwords.words('english')]
        self.vocab = self.get_vocab()
        self.ent_parse = ner_model(self.text)
        self.dep_parse = dep_model(self.clean_text)
        self.is_compressed = comp

    def get_vocab(self):
        return set(self.tokens)

    def get_tf_array(self):
        return np.array([self.tokens.count(w) for w in self.document.topic.vocab])

    def length(self):
        return len(self.tokens)

    def named_entity_count(self):
        return len(self.ent_parse.ents)

    def doc_pos(self):
        return float(self.pos) / len(self.document.sentences)

    def feat_vec(self):
        feat_methods = [
                self.named_entity_count,
                self.doc_pos
                ]
        return [x() for x in feat_methods]

class Model:

    def __init__(self, docset_id, text, topic):
        self.docset_id = docset_id
        self.text = text
        self.topic = topic
        self.sentences = [Sentence(s, self, i) for i, s in enumerate(sent_tokenize.tokenize(text))]

    def vocab(self):
        return set().union(*[s.vocab for s in self.sentences])

    def length(self):
        return sum(s.length() for s in self.sentences)

class Document:

    def __init__(self, doc_id, headline, topic, text):
        self.doc_id = doc_id
        self.headline = headline
        self.text = text
        self.topic = topic

        if args.excludeempty:
            # TODO: sentence len threshold of 2 here?
            atleastonechar = re.compile(r'\w+')
            self.sentences = [Sentence(s, self, i) for i, s in enumerate(sent_tokenize.tokenize(text)) if len(s) >= 2 and atleastonechar.search(s)]
        else:
            self.sentences = [Sentence(s, self, i) for i, s in enumerate(sent_tokenize.tokenize(text))]
        if args.compress:
            sentence_tuples =([(i,compress(sentence.dep_parse)) for i,sentence in enumerate(self.sentences) if len(sentence.tokens) >= 25])
            self.sentences.extend([Sentence(tup[1],self,tup[0],comp=True) for tup in sentence_tuples if tup[1]])
        self.vocab = self.get_vocab()

    def get_vocab(self):
        return set().union(*[s.vocab for s in self.sentences])

    def length(self):
        return sum(s.length() for s in self.sentences)

class Topic:

    def __init__(self, title, year, docset_id, doc_ids, corpus, model_path=None):
        self.model_path = model_path
        self.corpus = corpus
        self.title = title
        self.year = year
        self.docset_id = docset_id
        if model_path:
            self.models = self.load_models()
        # TODO: Make duc2007 a parameter
        self.docs = self.load_documents(doc_ids, docset_id, Duc2007=True)
        self.vocab = self.get_vocab()
        self.selected_sentences = []
        self.sentence_dict = self.get_sentence_dict()

    # Funky way to allow parallel processing of matrix calculations
    def set_matrices(self, data):
        self.tfidf_matrix = data['tfidf_matrix']
        self.tfidf_centroid = data['tfidf_centroid']
        self.bert_matrix = data['bert_matrix']
        self.bert_centroid = data['bert_centroid']
        self.topic_bert = data['topic_bert']

    def calculate_matrices(self):
        tfidf_matrix = self.get_tfidf_matrix()
        bert_matrix = self.get_bert_matrix()
        return (self.docset_id, {
            'tfidf_matrix': tfidf_matrix,
            'tfidf_centroid': self.get_centroid(tfidf_matrix),
            'bert_matrix': bert_matrix,
            'bert_centroid': self.get_centroid(bert_matrix),
            'topic_bert': self.get_topic_bert()
        })

    def get_vocab(self):
        return set().union(*[d.vocab for d in self.docs])

    def length(self):
        return sum(d.length() for d in self.docs)

    def load_document(self, doc_id, docset_id, Duc2007 = False):
        # We don't know how to parse the id until we know what year it's
        # from, so grab the year first by taking the first 4 digits.
        # Use this to decide if we'll be drawing from AQUAINT or AQUAINT-2
        year = re.sub('\D', '', doc_id)[:4]
        if Duc2007:
            doc_corpus = 'DUC2007'
            path = self.corpus.get_duc2007_path(docset_id, doc_id)
            print(path)
        elif self.year == '2011':
            doc_corpus = 'gigaword'
            path = self.corpus.get_gigaword_path(year, doc_id)
        elif int(year) > 2000:
            doc_corpus = 'aquaint2'
            path = self.corpus.get_aquaint2_path(year, doc_id)
        else:
            doc_corpus = 'aquaint'
            path = self.corpus.get_aquaint_path(year, doc_id)
        # Load the tree from memory if the path has already been parsed.
        if self.corpus.xml_memo.get(path) is not None:
            tree = self.corpus.xml_memo.get(path)
        else:
            if doc_corpus == 'DUC2007':
                tree = self.corpus.read_sgml(path)
                # todo: what tree is this?
                # print(html.tostring(tree))
            elif doc_corpus == 'aquaint2':
                tree = self.corpus.read_xml(path)
            elif doc_corpus == 'aquaint':
                tree = self.corpus.read_sgml(path)
            else:
                tree = self.corpus.read_sgmlgz(path)
            self.corpus.xml_memo[path] = tree
        if doc_corpus == 'DUC2007':
            # doc_elem = tree.getroot()
            # root = html.parse("example.html").getroot()
            # element = root.get_element_by_id("hello")
            # doc_elem1 = tree.get_element_by_id("DOC")
            # print(doc_elem1 + "---------------")
            # doc_elem1 = html.parse(path) # .getroot()
            doc_elem = tree.xpath('doc') # tree.xpath(".//DOCNO[text()[normalize-space(.)='{}']]".format(doc_id))#.getparent()
            # TODO: do separate investigation into htmltree vs etree
            # print(etree.tostring(doc_elem1))
            # print(etree.tostring(doc_elem[0]))

            return self.corpus.parse_duc2007_article(doc_id, doc_elem[0], self)
        if  doc_corpus == 'aquaint2':
            doc_elem = tree.getroot().find("./DOC[@id='{}']".format(doc_id))
            return self.corpus.parse_aquaint2_article(doc_id, doc_elem, self)
        elif doc_corpus == 'gigaword':
            doc_elem = tree.find(".//doc[@id='{}']".format(doc_id))
            return self.corpus.parse_gigaword_article(doc_id, doc_elem, self)
        else:
            doc_elem = tree.xpath(".//docno[text()[normalize-space(.)='{}']]".format(doc_id))[0].getparent()
            return self.corpus.parse_aquaint_article(doc_id, doc_elem, self)

    def get_sentence_dict(self):
        return dict(enumerate([s for doc in self.docs for s in doc.sentences]))

    def load_documents(self, doc_ids, docset_id, Duc2007=False):
        return [self.load_document(doc_id, docset_id, Duc2007) for doc_id in doc_ids]

    def get_model_paths(self):
        filename = self.docset_id[:5] + '-A.M.100.A.*'
        path = os.path.join(self.model_path, self.year, filename)
        return glob.glob(path)

    def get_duc_model_paths(self):
        self.docset_id = self.docset_id.strip()
        filename = self.docset_id[:5] + '.M.250.*.*'
        path = os.path.join(self.model_path, filename)        
        gl = glob.glob(path)
        print('model_path={}, filename={}, glob={}'.format(path, filename, gl))
        sys.stdout.flush()
        return gl   # glob.glob(path)

    def load_models(self):
        models = []
        for path in self.get_duc_model_paths():  # todo: re-add option for self.get_model_paths():
            with open(path) as f:
                models.append(Model(self.docset_id, f.read(), self))
                print('Model: {}'.format(path))
                sys.stdout.flush()
        return models

    def get_idf_vector(self):
        minimum = max(self.corpus.gigaword_idf_dict.values())
        return np.array([self.corpus.gigaword_idf_dict.get(w) or minimum for w in self.vocab])

    def get_tfidf_matrix(self):
        return np.multiply(np.array([s.get_tf_array() for doc in self.docs for s in doc.sentences]), self.get_idf_vector())

    def get_topic_bert(self):
        # I have no idea why there are two np.sums here, but it works
        return np.sum(np.sum(np.array([emb[1] for emb in bert_embedding([self.title]) if emb[1]]), axis=0), axis=0)

    def get_centroid(self, m):
        return np.sum(m,axis=0)

    def get_bert_matrix(self):
        return np.array([np.sum(np.array(arrs), axis=0) for arrs in [emb[1] for emb in bert_embedding([s.text for doc in self.docs for s in doc.sentences])]])

class Corpus:

    @classmethod
    def load_pickle(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # for DUC2007, docset_list_path = //corpora/DUC/DUC/duc07.results.data/testdata/duc2007_topics.sgml ...
    '''
    <topic>
    <num> D0701A </num>
    <title> Southern Poverty Law Center  </title>
    <narr>
    Describe the activities of Morris Dees and the Southern Poverty Law Center. 
    </narr>
    <docs>
    XIE19980304.0061
    NYT19980715.0137
    ...
    '''
    def __init__(self, docset_list_path, corpus_root, model_path=None):
        self.gigaword_idf_dict = self.load_gigaword_idf()
        self.docset_list_path = docset_list_path
        self.corpus_root = corpus_root
        self.model_path = model_path
        self.topics = self.load_topics(True)  # duc2007=True
        self.length = sum(t.length() for t in self.topics)

    def read_xml(self, path):
        parser = etree.XMLParser(remove_blank_text=True)
        try:
            tree = etree.parse(path, parser)
        except:
            print("Error: lxml was not able to read a path")
            raise
        return tree

    def load_gigaword_idf(self):
        with open(os.path.join(project_root, "./resources/freq_out")) as f:
            return dict([(splat[0], float(splat[1])) for splat in [line.split() for line in f.readlines()]])

    def read_sgml(self, path):
        try:
            with open(path) as f:
                parser = etree.HTMLParser(
                        encoding='utf-8',
                        remove_blank_text=True)
                tree = html.fragment_fromstring(
                        f.read(),
                        create_parent='body',
                        parser=parser)
        except Exception as e:
            print("Error: lxml was not able to read a path")
            raise
        return tree

    def read_sgmlgz(self, path):
        try:
            with gzip.open(path) as f:
                parser = etree.HTMLParser(
                        encoding='utf-8',
                        remove_blank_text=True)
                tree = html.fragment_fromstring(
                        f.read(),
                        create_parent='body',
                        parser=parser)
        except Exception as e:
            print("Error: lxml was not able to read a path")
            raise
        return tree

    # doc_id in example of doc_filename: XIE19980304.0061
    def get_duc2007_path(self, docset_id, doc_id):
        subdir = 'duc2007_testdocs'
        return os.path.join(self.corpus_root,
                            subdir,
                            'main',
                            docset_id.strip(),
                            doc_id)

    def get_aquaint2_path(self, year, doc_id):
        doc_id_fields = re.split('[_\.]', doc_id)
        pub = doc_id_fields[0]
        month = doc_id_fields[2][4:6]
        lang = doc_id_fields[1]
        subdir = 'LDC08T25'
        filename = '_'.join([pub.lower(),
            lang.lower(),
            year + month + '.xml'])
        return os.path.join(self.corpus_root,
                subdir,
                'data',
                pub.lower() + '_' + lang.lower(),
                filename)

    def get_aquaint_path(self, year, doc_id):
        pub = doc_id[:3]
        month = doc_id[7:9]
        subdir = 'LDC02T31'
        # There's not really a set pattern for these suffixes, so each one
        # is hard-coded here.
        if pub == 'APW':
            suffix = 'APW_ENG'
        elif pub == 'XIE':
            suffix = 'XIN_ENG'
        else:
            suffix = 'NYT'
        return os.path.join(self.corpus_root,
                subdir,
                pub.lower(),
                year,
                doc_id.split('.')[0][3:] + '_' + suffix)

    def get_gigaword_path(self, year, doc_id):
        pub = doc_id[:3]
        month = doc_id[7:9]
        #/corpora/LDC/LDC11T07/data/nyt_eng/nyt_eng_199407.gz
        #NYT_ENG_19940701.0001
        filename = doc_id.lower()[:14] + ".gz"
        return os.path.join(self.corpus_root,
                'LDC11T07',
                'data',
                doc_id.lower()[:7],
                filename)

    def parse_aquaint_article(self, doc_id, doc_elem, topic):
        headline_elem = doc_elem.find("./headline")
        if headline_elem is not None:
            headline = headline_elem.text
        else:
            headline = ''
        text = ''.join(doc_elem.find("./text").itertext())
        return Document(doc_id, headline, topic, text)

    def parse_aquaint2_article(self, doc_id, doc_elem, topic):
        headline_elem = doc_elem.find("./HEADLINE")
        if headline_elem is not None:
            headline = headline_elem.text
        else:
            headline = ''
        text = ''.join(doc_elem.find("./TEXT").itertext())
        return Document(doc_id, headline, topic, text)

    def parse_duc2007_article(self, doc_id, doc_elem, topic):
        headline_elem = doc_elem.find("./headline")
        if headline_elem is not None:
            headline = headline_elem.text
            if headline:
                headline_to_print = headline.strip()
            else:
                headline_to_print = ''
            print('*****headline={}*******'.format(headline_to_print))
            sys.stdout.flush()
        else:
            headline = ''
        text = ''.join(doc_elem.find("./text").itertext())
        # print('*****text={}*******'.format(text))
        return Document(doc_id, headline, topic, text)

    def parse_gigaword_article(self, doc_id, doc_elem, topic):
        headline_elem = doc_elem.find("./headline")
        if headline_elem is not None:
            headline = headline_elem.text
        else:
            headline = ''
        text = ''.join(doc_elem.find("./text").itertext())
        return Document(doc_id, headline, topic, text)

    def calculate_topic_matrices(self, topic):
        return topic.calculate_matrices()

    def load_topics(self, duc2007 = False):
        self.xml_memo = {}
        topics = []
        if duc2007:
            tree = self.read_sgml(self.docset_list_path)
        else:
            tree = self.read_xml(self.docset_list_path)
        # TODO: FIX for non-DUC2007
        # year = tree.getroot().get('year')
        topic_elems = tree.findall('./topic')
        topic_id_dict = {}

        for topic_elem in topic_elems:
            title = ''.join(topic_elem.find('./title').itertext())
            print("Loading: " + title)
            sys.stdout.flush()
            if duc2007:
                docset = topic_elem.find('./docs')
                docset_id = ''.join(topic_elem.find('./num').itertext())
                docset_id = docset_id.strip()
                doc_elems = str.split(docset.text)
                year = '2007'
                topic = Topic(title, year, docset_id, doc_elems, self, self.model_path)
            else:
                docset = topic_elem.find('./docsetA')
                docset_id = docset.attrib['id']
                doc_elems = docset.findall('.//doc')
                topic = Topic(title, year, docset_id, [doc_elem.attrib['id'] for doc_elem in doc_elems], self, self.model_path)
            topic_id_dict[docset_id] = topic
            topics.append(topic)
        self.xml_memo = {}
        print("beginning matrix calculations")
        sys.stdout.flush()
        for i, topic in enumerate(topics):
            print("calculating matrix for: " + topics[i].title)
            sys.stdout.flush()
            topic_id, result = self.calculate_topic_matrices(topic)
            topic_id_dict[topic_id].set_matrices(result)

        print("completed matrix calculations")
        matrixDT = datetime.datetime.now()
        print(str(matrixDT))
        elapsedDT = matrixDT - currentDT
        print(divmod(elapsedDT.total_seconds(), 60))
        sys.stdout.flush()
        return topics

    def pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

if __name__ == "__main__":
    print("Beginning corpus loader")
    from bert_embedding import BertEmbedding
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
    ner_model = spacy.load('/home/khenner/shared_resources/ner_model')
    dep_model = spacy.load('/home/khenner/shared_resources/en_core_web_md/en_core_web_md-2.1.0')
    # ----------------------------------------------------------
    # Use command_args.py to define new arguments
    args = parseArgs(sys.argv[1:])  # Omit argv[0] because argv[0] is just the path to this file
    if args.corpuspath:
        pickle_path = args.corpuspath
    else:
        pickle_path = sys.argv[2]
    if args.xmlpath:
        topic_xml_path = args.xmlpath
    else:
        topic_xml_path = sys.argv[1]
    if args.modelpath:
        model_path = args.modelpath
        print('MODEL_PATH={}'.format(model_path))
    elif len(sys.argv) >= 4:
        model_path = sys.argv[3]
        print('MODEL_PATH={}'.format(model_path))
    else:
        model_path = None
        print('NO MODEL PATH!')
    # TODO: corpus_root = '/corpora/DUC/duc07.results.data/testdata',
    # todo: or 'corpora/DUC/duc07.results.data/testdata/duc2007_testdocs/main/DxxxxA/APWyyyyNNNN.NNNN'
    # corpus_root = '/corpora/LDC'
    corpus_root = '/corpora/DUC/duc07.results.data/testdata'
    print("Creating corpus")

    corpus = Corpus(topic_xml_path, corpus_root, model_path)
    with open(pickle_path, "wb+") as f:
        pickle.dump(corpus, f)
