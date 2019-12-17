#!/bin/python3

import sys
import re
import os
import string
from lxml import etree
from lxml import html
from lxml.etree import tostring
import glob
import pickle
import random
import spacy
from spacy.util import minibatch, compounding
from pathlib import Path

class TrainingData:

    def __init__(self, paths):
        self.paths = paths
        self.examples = []
        for path in self.paths:
            tree = self.read_sgml(path)
            self.examples += self.get_examples_from_tree(tree)

    def get_examples_from_tree(self, tree):
        doc_elems = tree.findall('.//doc')
        train_data = []
        for doc_elem in doc_elems:
            train_data.append(self.get_examples_from_elem(doc_elem))
        return train_data

    def get_examples_from_elem(self, elem, pos=0):
        entities = []
        text = elem.text or ""
        entity_type = elem.get('type') or None
        if entity_type:
            entities += [(pos, pos+len(text), entity_type)]
        for e in elem:
            res = self.get_examples_from_elem(e, pos + len(text))
            text += res[0]
            entities += res[1]
            if e.tail:
                text += e.tail

        return (text, entities)

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

# Code adapted from https://spacy.io/usage/training
def train(train_data, model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations:
            ner.add_label(ent[2])

    # reformat with entities dict
    train_data = [(x[0], {'entities': x[1]}) for x in train_data]

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

def test(model, train_data):
    # test the trained model
    for text, _ in train_data:
        doc = model(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), '..')
    paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk('/corpora/LDC/LDC05T33/data') for f in filenames if os.path.splitext(f)[1] == '.qa']
    corpus_root = '/corpora/LDC'
    corpus = TrainingData(paths)
    nlp = spacy.blank("en")  # create blank Language class
    train(corpus.examples, output_dir = os.path.join(project_root, './resources/ner_model')
