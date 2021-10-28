#!/bin/python3

# TODO: Run reordering to check performance on new social media dataset
import pickle
from corpus_loader import *
import permutations as Permutations
import sklearn.svm
import numpy as np
import random
import itertools
import time
import math
import sys
import re

# for create directory
import os.path
# for argument parsing
import argparse
from command_args import *
# helpers for creating the results files
from create_results import *

TRAINING_SET_INDEX = 0
TAC_2010 = False  # todo: make this an argument
TEST_DOC = 0     # used only for initialization and output to arbitrarily sample first doc in a topic.
MAX_ORDERINGS = 24  # m = 5040 if  40K > m > 5040 because = 7!
MAX_SENTENCES_PER_DOC = 500  # not using this yet
e_orig = "./resources/entitygrid.model"
e_basic = "./src/entitygrid_basic_devtest0.model"
e_oneoff = "./src/entitygrid_fix_one_off.model"
# "./src/entitygrid_devtest_nsent_fix.model"
ENTITY_GRID_MODEL_PATH = e_oneoff
PROGRESS_DETAIL = 1  # 1 to show topic progress, 2 to show ordering progress
TIME_DETAIL = 1
DOC_DETAIL = 1  # 2 to show doc-level progress
DEBUG_POSITION = False
SHOW_REORDERED_DOC = False
UNIT_TEST = False
project_root = os.path.join(os.path.dirname(__file__), '..')
output_dir = './outputs/'

# output_subdir defaults to 'D3' for deliverable.
# For testing, change output_subdir to save test runs in separate subdirectories
output_subdir = 'D3'

if UNIT_TEST:
    '''sanity test of Permutations import'''
    Permutations.unit_test()

def print_elapsed_time(begin_t, end_t, message=".", show_seconds=True):
    '''
    Prints a message about how many minutes and seconds a process took.
    :param begin_t: start time, calculated by time.time()
    :param end_t: start time, calculated by time.time()
    :param message: optional message like "to process corpus"
    :return:
    '''
    if show_seconds == True:
        print('Took {} minutes, {} seconds {}'.format((end_t - begin_t) // 60, (end_t - begin_t) % 60, message))
    elif show_seconds == False:
        print('Took {} minutes {}'.format(
            (end_t - begin_t) / 60,
            (end_t - begin_t) % 60,
            message)
        )

def print_sentence_array(sentlist, showindices=True):
    for i, s in enumerate(sentlist):
        if showindices:
            print('{}: {}'.format(i, str(s.text).rstrip()))
        else:
            print('{}'.format(str(s.text).rstrip()))
    return

class OutputGenerator():
    '''
    num_sentences: Will be set to number of sentences that were selected
         by the output selector. Not normally provided unless testing.
    out_dir: The directory for saving files, relative to the constant output_dir
    If not provided, it defaults to the global constant output_subdir
    '''
    def __init__(self, topic, n_sentences=None, out_dir=output_subdir):
        # TODO: Need to scrub
        if n_sentences is None:
            # n_sentences = len(topic.selected_sentences)
            # TODO: initialize it to length of first doc, but we actually change this field for each doc
            # See the main loop through the corpus
            n_sentences = len(topic.docs[TEST_DOC].sentences)
        if out_dir is None:
            out_dir = output_subdir

        base_output_dir = os.path.join(project_root, output_dir)
        self.subdirectory = os.path.join(base_output_dir, out_dir)
        self.create_output_dir()
        # the maximimum number of sentences to output. Different from number of already selected sentences
        self.n_sentences = n_sentences
        self.topic = topic

        # entity grids
        self.naive_grid_cols = {}
        self.naive_grid_row_ind = {}

    def create_output_dir(self):
        # create directory
        os.makedirs(self.subdirectory, exist_ok=True)


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
    # TODO: Do we need an option to base on n_sentences?
    def vectors_permuted_originals(self, num_perm, d=0, permute=None, dbg=None):
        entity_info = ''
        total_num_ents = 0
        letter = self.topic.docset_id[5]
        filename = topic.docset_id[:5] + '-A.M.100.' + letter + '.1'

        vectorlist = []
        sent_perms = []
        # TODO: make docs[0] configurable. it's just arbitrarily hardcoded
        sent_perms.append(self.topic.docs[0].sentences)
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
            entity_file = open(os.path.join(
                self.subdirectory, 'PermutationEntities' + '{}-{}'.format(filename, i)), "w")
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
            if (dbg):
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
    def get_naive_entity_grid(self, d_sentences, dbg=None):
        entity_info = ''
        total_num_ents = 0

        # CURRENT_DOC = ?
        sents = d_sentences   # self.topic.docs[CURRENT_DOC].sentences
        for i, sentence in enumerate(sents):
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
        # TODO: CHECK grid not empty
        # get probability by dividing count by num_sentences * num_entities-1
        for i, c in enumerate(counts.values()):
            if self.n_sentences == 0 or (len(grid[0]) - 1) == 0:
                print('Zero in denominator in topic {}. Length of grid[0]={}'.format(self.topic.docset_id[:5], len(grid[0])))
                vector[i] = 0
            else:
                vector[i] = c / (self.n_sentences * (len(grid[0]) - 1))
        return vector

    # output the summary
    # TODO: for factorial(n_sentences), multiply them all their feature vectors by the weights from the model,
    # then pick the optimal ordering to print here.
    def output_summary(self, reordered_sents=None, exclude=None, verbose=None):
    # PARAMETERS:
    #     exclude: Specifies if entire sentences that push the output over 100 words should be excluded.
    #              Has the effect of truncating the summary to under 100 words, which reduces ROUGE score
    #     verbose: Print out a message indicating the subdirectory and filename being written to.
        # TODO: This currently just outputs one of the docs to save space.
        sents = self.topic.docs[TEST_DOC].sentences
        if reordered_sents:
            sents = reordered_sents

        letter = self.topic.docset_id[5]
        if verbose:
            print('writing to ' + generator.subdirectory + '/' + str(topic.docset_id[:5] + '-A.M.100.' + letter + '.1'))
        filename = topic.docset_id[:5] + '-A.M.100.' + letter + '.1'
        with open(os.path.join(self.subdirectory, filename), "w") as output_file:
            output = ''
            for i, sentence in enumerate(sents):
                if i < self.n_sentences:
                    if exclude:
                        if len(output.split()) + len(sentence.clean_text.split()) < 100:
                            text = sentence.clean_text
                            newtext = re.sub(r'\s+', ' ', sentence.clean_text)
                            output += newtext.strip() + '\n'
                        else:
                            break
                    else:
                        text = sentence.clean_text
                        newtext = re.sub(r'\s+', ' ', sentence.clean_text)
                        output += newtext.strip() + '\n'

            output_file.write(output)

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
print("About to load corpus from {}.".format(corpus_pickle_path))
t_corpus_start = time.time()
corpus = Corpus.load_pickle(corpus_pickle_path)
t_corpus_end = time.time()
print_elapsed_time(t_corpus_start, t_corpus_end, message="to load corpus.")

#  '../resources/entitygrid.model'
training_filename = os.path.join(project_root, ENTITY_GRID_MODEL_PATH)
print("About to load entity grid model from {}.".format(training_filename))
model = load_model(training_filename)

# Variables used for recap
num_reordered = 0
total_misclassified_perms = 0
total_perms_checked = 0
reordered_doc_list = []
num_sents_in_doc = {}
t_begin_process_topics = time.time()
total_doc_count = 0
total_sent_count = 0
skip_count = 0
topic_accuracies = {}

'''
main loop through topics
'''
for topic in corpus.topics:
    # TODO: Go through pairs of candidates and run the classifier to get -1 or 1
    # randomly shuffling is less memory-intensive
    # need a vector for each of num_perms iterations of s = random.sample(topic.selected_sentences, n_sent)
    # in this set, find the highest scoring vector by sorting them.
        # to compare two vectors, get the difference between them and run the model.
        # -1 means the second is greater, 1 means the first is greater.

    generator = OutputGenerator(topic, out_dir=args.directory)
    print("processing {}".format(topic.docset_id))
    time_begin_topic = time.time()
    doc_num = 0
    topic_misclassified_perms = 0
    topic_perms_checked = 0

    for d in topic.docs:
        if doc_num == TRAINING_SET_INDEX and TAC_2010:
            # skip the docs we used to train the model
            doc_num += 1
            continue
        # try setting n_sentences here
        generator.n_sentences = len(d.sentences)
        s_to_reorder = d.sentences  # topic.docs[TEST_DOC].sentences

        # Create grid
        grid = generator.get_naive_entity_grid(d_sentences=s_to_reorder)
        orig_order_vector = generator.get_naive_entity_vector(grid)
        best_order_vector = orig_order_vector
        label = topic.docset_id

        # TODO: try only considering docs with < MAX_SENTENCES_PER_DOC
        if len(s_to_reorder) > MAX_SENTENCES_PER_DOC:
            print("Skipping a doc in {}, len={} too long.".format(topic.docset_id, len(s_to_reorder)))
            skip_count += 1
            continue
        else:
            doc_num += 1
            if DOC_DETAIL > 1:
                print("#{}, # len={}.".format(doc_num, len(s_to_reorder)))
            total_doc_count += 1
            total_sent_count += len(d.sentences)
        num_sents_in_doc[label] = len(s_to_reorder)
        # the "best" order starts off being the originally selected one
        best_order = s_to_reorder

        # todo: put code to get vector in a separate function
        dif_vects = []
        dif_vects_wrt_orig = []
        pred_vects = []
        misclassified_count = 0

        perms = Permutations.get_perms_arr(s_to_reorder, MAX_ORDERINGS, True)
        total_perms_checked += len(perms)
        topic_perms_checked += len(perms)
        # if len=n_sents: shorter_perms = [np.random.choice(s_to_reorder, size=len(s_to_reorder), replace=False)]

        orig_order = s_to_reorder  # Was: perms[0]
        best_order_ind = -1

        for i in range(MAX_ORDERINGS):
            if i % 500 == 0 and PROGRESS_DETAIL > 1:
                print("Processed {} orderings".format(i))
            sample_size = generator.n_sentences  # n_sentences = len(topic.doc[TEST_DOC].sentences)
            if sample_size > len(s_to_reorder):
                sample_size = len(s_to_reorder)
            # curr_sample = random.sample(s_to_reorder, sample_size)
            if i < len(perms):
                curr_sample = perms[i]
            else:
                break
            # get a vector for the current sample
            (cols, rows) = generator.create_rows_cols(curr_sample)
            grid = generator.create_naive_grid(generator.n_sentences, cols, rows)
            pvect = generator.get_naive_entity_vector(grid)
            # --------------------------------
            # todo: put comparison in separate function
            # compare this vector to the generated summary vector (first the original, then best found)
            dif_vect_wrt_orig = [orig_order_vector[ind] - pvect[ind] for ind in range(len(pvect))]
            dif_vects_wrt_orig.append(dif_vect_wrt_orig)
            wrt_orig_pred = model.predict([dif_vect_wrt_orig])
            if wrt_orig_pred < 0:
                misclassified_count += 1

            dif_vect = [best_order_vector[ind] - pvect[ind] for ind in range(len(pvect))]
            dif_vects.append(dif_vect)
            cur_pred = model.predict([dif_vect]) # is just this current vector the best?

            if (cur_pred < 0):
                # if pred is better than the original then make this one the best
                best_order = curr_sample
                best_order_vector = pvect
                best_order_ind = i
        pred_vects = model.predict(dif_vects)  # get a whole list of comparisons

        # We found a reorder
        if best_order_ind != -1:
            if DOC_DETAIL > 1:
                print('***REORDERED Doc#{}, {}/{} misclassified'.format(
                    str(topic.docset_id[:5]),
                    misclassified_count, len(perms)  # if limit n_sent, and max=n_sent!, use math.factorial(len(s_to_reorder))
                ))
            total_misclassified_perms += misclassified_count
            topic_misclassified_perms += misclassified_count
            reordered_doc_list.append(str(topic.docset_id[:5]))
            if DEBUG_POSITION:
                print('Best i={}, Predictions of permutations: {}'.format(best_order_ind, pred_vects))
            if SHOW_REORDERED_DOC:
                print('Original:')
                print_sentence_array(orig_order)
                print('Reordered:')
                print_sentence_array(perms[best_order_ind])
            num_reordered += 1
    # end for d in topics.docs

    topic_accuracies[topic.docset_id] = (1 - (topic_misclassified_perms/topic_perms_checked))
    print("Topic {}: {} docs, {} mistakes, {} permutations, acc={}".format(
        topic.docset_id, doc_num,
        topic_misclassified_perms, topic_perms_checked,
        1 - (topic_misclassified_perms/topic_perms_checked)))
    time_end_topic = time.time()
    if TIME_DETAIL > 1:
        print_elapsed_time(time_begin_topic, time_end_topic,
                       message="#sentences={}".format(generator.n_sentences))
    if args.reorder:
        generator.output_summary(reordered_sents=best_order, exclude=args.excludepartialsentence, verbose=args.verbose)
    else:
        generator.output_summary(exclude=args.excludepartialsentence, verbose=args.verbose)

t_end_process_topics = time.time()
print_elapsed_time(t_begin_process_topics, t_end_process_topics, message="to process corpus.")

# print summary
print('\nSUMMARY: {} docs, {} sentences, {} avg sentences per doc'.format(total_doc_count, total_sent_count, total_sent_count / total_doc_count))
s = ', '
print('Number reordered={}, ({})'.format(num_reordered, s.join(reordered_doc_list)))
print('{}/{}={} error'.format(
    total_misclassified_perms, total_perms_checked, total_misclassified_perms/total_perms_checked))
print('{}/{}={} acc'.format(
    total_perms_checked - total_misclassified_perms, total_perms_checked,
    1 - (total_misclassified_perms/total_perms_checked) ))
lg_perm_pairs = str(num_sents_in_doc)
print('topic lengths={}\n'.format(lg_perm_pairs))

# print accuracies per topic
for k, v in topic_accuracies.items():
    print('{}\t{}'.format(k, v))



# Generate ROUGE score files
if args.directory:
    createXmlConfigCopy(args.directory)
    if args.runrouge:
        runRouge(args.directory)
else:
    # Don't need to create file, we just use rouge_run_output.xml that's already in repo
    if args.runrouge:
        runRouge()
