#!/bin/python3

import pickle
from corpus_loader import *

import sklearn.svm
import random
import itertools
import math
import sys

# for create directory
import os.path
# for argument parsing
import argparse
from command_args import *
# helpers for creating the results files
from create_results import *


# "./resources/entitygrid_devtest.model"
DEFAULT_ENTITY_GRID_PATH = "./resources/entitygrid_devtest_nsent_fix.model"
project_root = os.path.join(os.path.dirname(__file__), '..')
TRAINING_DOC_INDEX = 0
class Trainer():
    '''
    out_dir: The directory for saving files, relative to the constant output_dir
    If not provided, it defaults to the global constant output_subdir
    '''
    def __init__(self, topic, n_sentences=None):

        if n_sentences is None:
            n_sentences = len(topic.docs[TRAINING_DOC_INDEX].sentences)
            # For class project, we were using all previously selected sentences
            # n_sentences = len(topic.selected_sentences)

            # the maximimum number of sentences to output. Different from number of already selected sentences
        self.n_sentences = n_sentences
        self.topic = topic

        # entity grids
        self.naive_grid_cols = {}
        self.naive_grid_row_ind = {}



    ###############################################################
    #  Helpers for 'naive' entity grid that simple has a 1 or a 0
    #  for presence or absence of an entity
    def update_naive_entity_grid(self, entity, sentence_num):
    # for a given sentence, update the grid
        ent_inf = (sentence_num, entity[1], entity[2], entity[3]) # sent, begin, end, type
        self.naive_grid_cols.setdefault(entity[0], []).append(ent_inf)
        self.naive_grid_row_ind.setdefault(sentence_num, []).append(entity[0])
        return

    def create_rows_cols(self, sents):
        cur_grid_cols = {}
        cur_grid_row_ind = {}
        for k, sentence in enumerate(sents):
            ent_count = sentence.named_entity_count()
            ent_dbg_info = ''
            if ent_count > 0:
                ent_dbg_info += '\nSentence {}: #Entities={}\n'.format(k, ent_count)
                ents = [(e.text, e.start_char, e.end_char, e.label_) for e in sentence.ent_parse.ents]
                for en in ents:
                    ent_inf = (k, en[1], en[2], en[3])  # sent, begin, end, type
                    cur_grid_cols.setdefault(en[0], []).append(ent_inf)
                    cur_grid_row_ind.setdefault(k, []).append(en[0])
                    ent_dbg_info += '({},{},{},{})'.format(en[0], en[1], en[2], en[3])
        return (cur_grid_cols, cur_grid_row_ind)

    def create_naive_grid(self, sentence_num, cols=None, rows=None):
        if cols:
            num_ents = len(cols.keys())
        else:
            cols = self.naive_grid_cols
            num_ents = len(cols) # len(self.naive_grid_cols.keys())
        if rows == None:
            rows = self.naive_grid_row_ind

        Grid = [[0 for w in range(num_ents)] for h in range(sentence_num)]
        for x in range(sentence_num):
            for i, ent in enumerate(cols):  # enumerate(self.naive_grid_cols):
                try:
                    # if by-row list contain the entity
                    if ent in rows.get(x, []):  # self.naive_grid_row_ind.get(x, []):
                        Grid[x][i] = 1
                    else:
                        Grid[x][i] = 0
                except KeyError:
                    print('KeyError: sentence x={}'.format(x))
                except:
                    print('an exception occurred')
        # debug print
        return Grid

    #---------------------------
    #  Creates the entity grid for a permuted set of sentences
    def vectors_permuted_originals(self, num_perm, d=0, permute=None, dbg=None):
        entity_info = ''
        total_num_ents = 0
        letter = self.topic.docset_id[5]
        filename = topic.docset_id[:5] + '-A.M.100.' + letter + '.1'

        vectorlist = []
        sent_perms = []
        # TODO: make TRAINING_DOC_INDEX configurable. it's just arbitrarily hardcoded
        sent_perms.append(self.topic.docs[TRAINING_DOC_INDEX].sentences)
        # 1. get all possible permutations of self.topic.docs[0]
        all_indices = []
        if permute:
            all_indices = [index for index in
                itertools.permutations(range(len(self.topic.docs[0].sentences)), len(self.topic.docs[0].sentences))]
            all_indices.pop(0)  # get rid of the 0 1 2 3..n element since that's a dup of the original
            random.shuffle(all_indices)
        else:
            # default to number of sentences in topic (or --maxsentences)
            # TODO: not using this
            all_indices = [ind for ind in range(self.n_sentences)]
        # 2. Choose num_perm of those (perhaps shuffled) and add them to sent_perms
        for i in range(num_perm + 1):
            if i > len(all_indices) and permute:
                print('Note: training sentence too short to provide requested num_perm')
                break
            # entity_file = open(os.path.join(
            #     self.subdirectory, 'PermutationEntities' + '{}-{}'.format(filename, i)), "w")
            cur_naive_grid_cols = {}
            cur_naive_grid_row_ind = {}
            if permute:
                # TODO: make this more memory efficient
                try:
                    sent_perms.append(
                        [self.topic.docs[0].sentences[j] for j in all_indices[i]]  # add permutation of sentences
                )
                except:
                    print('error')
            else:  # sample instead of permuting
                # todo: make more memory efficient
                sent_perms.append(random.sample(self.topic.docs[0].sentences,
                                                len(self.topic.docs[0].sentences)))
            for k, sentence in enumerate(sent_perms[i]):
                ent_count = sentence.named_entity_count()
                total_num_ents += ent_count
                if ent_count > 0:
                    entity_info += '\nSentence {}: #Entities={}\n'.format(k, ent_count)
                    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in sentence.ent_parse.ents]
                    for en in ents:
                        ent_inf = (k, en[1], en[2], en[3])  # sent, begin, end, type
                        cur_naive_grid_cols.setdefault(en[0], []).append(ent_inf)
                        cur_naive_grid_row_ind.setdefault(k, []).append(en[0])
                        entity_info += '({},{},{},{})'.format(en[0], en[1], en[2], en[3])
            # create a vector from a grid
            grid = self.create_naive_grid(self.n_sentences, cur_naive_grid_cols, cur_naive_grid_row_ind)
            cur_vector = self.get_naive_entity_vector(grid)
            vectorlist.append(cur_vector)
            # if (dbg):
            entity_file = open("entity_info.txt", "w")
            self.print_entity_grid(grid, entity_file, entity_info, cur_naive_grid_cols)
        # return the list of vectors for the permutations
        return vectorlist

    def print_entity_grid(self, grid, entity_file, entity_info, cur_naive_grid_cols):
        entity_grid_values = ''
        for en in cur_naive_grid_cols.keys():
            entity_grid_values += '\t{}:{}\n'.format(en, cur_naive_grid_cols[en])
        entity_file.write(entity_info)
        entity_file.write(entity_grid_values)
        entity_grid_print = '\n'
        for i in range(self.n_sentences):
            for j in range(len(cur_naive_grid_cols)):
                entity_grid_print += str(grid[i][j]) + ' '
            entity_grid_print += '\n'
        entity_file.write(entity_grid_print)
        return
    #---------------------------
    #  Creates the entity grid
    def get_naive_entity_grid(self, dbg=None):
        entity_info = ''
        total_num_ents = 0

        for i, sentence in enumerate(self.topic.selected_sentences):
            # TODO: Use total_num_ents
            ent_count = sentence.named_entity_count()
            total_num_ents += ent_count
            if ent_count > 0 and i < self.n_sentences:
                entity_info += '\nSentence {}: #Entities={}\n'.format(i, ent_count)
                ents = [(e.text, e.start_char, e.end_char, e.label_) for e in sentence.ent_parse.ents]
                for en in ents:
                    # add to naive_grid_cols hash
                    self.update_naive_entity_grid(en, i)
                    # add entity info to debug output string
                    entity_info += '({},{},{},{})'.format(en[0], en[1], en[2], en[3])

        grid = self.create_naive_grid(self.n_sentences)
        if (dbg):
            letter = self.topic.docset_id[5]
            filename = topic.docset_id[:5] + '-A.M.100.' + letter + '.1'
            entity_file = open(os.path.join(self.subdirectory, 'Entities-' + filename), "w")
            self.print_entity_grid(grid, entity_file, entity_info, self.naive_grid_cols)
        return grid

    # ---------------------------------------------------------
    #  Creates the entity grid vector
    #  using transitions in a column to the following row
    # ---------------------------------------------------------
    def get_naive_entity_vector(self, grid):
        # vector has a feature for each transition state between a cell and the next row (0,0)(0,1)(1,0)(1,1)
        # Total transitions = num_columns * num_sentences-1 transitions
        counts = {'11':0, '10':0, '01':0, '00':0}
        for i in range(len(grid) - 1):
            for j in range(len(grid[i])):
                if grid[i][j] == 1 and grid[i+1][j] == 1:
                    counts['11'] += 1
                if grid[i][j] == 1 and grid[i+1][j] == 0:
                    counts['10'] += 1
                if grid[i][j] == 0 and grid[i+1][j] == 1:
                    counts['01'] += 1
                if grid[i][j] == 0 and grid[i+1][j] == 0:
                    counts['00'] += 1
        vector = [0, 0, 0, 0]
        # get probability by dividing count by num_sentences * num_entities-1
        for i, c in enumerate(counts.values()):
            if self.n_sentences == 0 or len(grid) == 0 or (len(grid[0]) - 1) == 0:
                try:
                    print('Zero in denominator in topic {}. Empty grid in {}'.format(self.topic.docset_id[:5]))
                except:
                    print('error: empty grid')
                vector[i] = 0
            else:
                vector[i] = c / (self.n_sentences * (len(grid[0]) - 1))
        return vector


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def rank_svm(X_pairs, y_pairs):
    rank_model = sklearn.svm.SVC(kernel='linear', C=0.1)
    rank_model.fit(X_pairs, y_pairs)

    return rank_model
# ----------------------------------------------------------
# Use command_args.py to define new arguments
args = parseArgs(sys.argv[1:])  # Omit argv[0] because argv[0] is just the path to this file

if args.corpuspath:
    corpus_pickle_path = args.corpuspath
else:
    corpus_pickle_path = os.path.join(project_root, "./resources/corpus.obj")
corpus = Corpus.load_pickle(corpus_pickle_path)

if args.orderingmodel:
    training_filename = args.orderingmodel
else:
    training_filename = os.path.join(project_root, DEFAULT_ENTITY_GRID_PATH)

if args.permutations:
    NUM_PERMUTATIONS = args.permutations
else:
    NUM_PERMUTATIONS = 20


diff_vectors = []
arr = []  # [ -1 or 1 for x in range(NUM_PERMUTATIONS)]
for t, topic in enumerate(corpus.topics):
    trainer = Trainer(topic)
    tpc_vect_list = trainer.vectors_permuted_originals(NUM_PERMUTATIONS)
    # todo Check: tpc_vect_list[0] must be first element if permuting
    orig_vector = tpc_vect_list[0]
    # TODO: Try randomizing the list instead of doing first half
    # TODO: Try START_VECTOR_INDEX =1. (always predicting 1 if we start at 1 instead of at tpc_vect_list[0])
    START_VECTOR_INDEX = 0
    for i in range(START_VECTOR_INDEX, int(len(tpc_vect_list) / 2)):
        diff_vectors.append([tpc_vect_list[i][j] - orig_vector[j] for j in range(len(tpc_vect_list[i]))])
        arr.append(-1)  # negative because it's permutation-orig and the original should have the highest value
    for i in range(int(len(tpc_vect_list) / 2), len(tpc_vect_list) - 1):
        # difference of orig and one following it
        diff_vectors.append([orig_vector[j] - tpc_vect_list[i][j] for j in range(len(tpc_vect_list[i]))])
        arr.append(1)  # positive because it's orig-permutation and the original should have the highest value
# print('diff vector:{}'.format(t, str(diff_vectors)))

model = rank_svm(diff_vectors, arr)
with open(training_filename, "wb") as f:
    pickle.dump(model, f)

correct = 0
for i, k in enumerate(model.predict(diff_vectors)):
    # print('Vec#{} predicted score = {},'.format(i, k))
    # print('Vec#{} actual value  = {},'.format(i, arr[i]))
    if k == arr[i]:
        correct += 1
print('training: {} correct out of {}'.format(correct, i+1))





