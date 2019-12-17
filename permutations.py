#!/bin/python3

import pickle
# from corpus_loader import *

import sklearn.svm
import numpy as np
import random
import itertools
import math
import string
import sys
import re

# for create directory
import os.path
# for argument parsing
import argparse
from command_args import *
# helpers for creating the results files
from create_results import *

project_root = os.path.join(os.path.dirname(__file__), '..')
output_dir = './outputs/'

class TestDoc:
    def __init__(self, sentences):
        self.sentences = sentences

class TestSentence:
    def __init__(self, text):
        self.text = text

def print_sentence_array(sentlist, showindices=True):
    print('------')
    for i, s in enumerate(sentlist):
        if showindices:
            print('{}: {}'.format(i, str(s.text).rstrip()))
        else:
            print('{}'.format(str(s.text).rstrip()))
    return


def get_perms_arr(sentences, max_num_p, include_first=True):
    list_perms = []
    if len(sentences) < 1 or max_num_p < 1: # shouldn't get here
        print('-WARNING: tried to get_perms_arr on empty list or specifying 0 permutations')
        return list_perms
    # base case 1 -- len(sentences) == 1, doesn't matter what max_num_p is
    if len(sentences) == 1:
        list_perms.append(sentences)
        return list_perms
    if len(sentences) <= max_num_p:
        for f in range(len(sentences)):
            s_minus_f = list(sentences)
            s_minus_f.remove(sentences[f])
            sublist = get_perms_arr(s_minus_f,
                                    max_num_p=math.floor(max_num_p/len(sentences)))
            for l in sublist:
                f_perm = []
                f_perm.append(sentences[f])
                f_perm.extend(l)
                list_perms.append(f_perm)
    # base case 2 --
    elif len(sentences) > max_num_p :
        # todo: there should be an option to always include a permutation with first element first
        # firsts = []
        if include_first:
            first_sentence = sentences[0]
            s_minus_first = list(sentences)
            s_minus_first.remove(first_sentence)
            firsts_include_1st = []
            firsts_include_1st.append(first_sentence)
            firsts_include_1st.extend(np.random.choice(s_minus_first, max_num_p - 1, replace=False))
            firsts = firsts_include_1st
        else:
            # randomly pick max_num_p elements to be first
            firsts_random = np.random.choice(sentences, max_num_p, replace=False)
            firsts = firsts_random
        for f in firsts:
            s_minus_f = list(sentences)
            s_minus_f.remove(f)
            remainder = np.random.choice(s_minus_f, len(s_minus_f), replace=False)
            # join f to remainder
            f_perm = []
            f_perm.append(f)
            f_perm.extend(remainder)
            list_perms.append(f_perm)
    return list_perms


def unit_test(num_sents=5, num_perms=2, repeats=1, include_first=True):
    NUM_PERMUTATIONS = num_perms
    print("NUM_PERMUTATIONS={}, include_first={}".format(num_perms, include_first))
    large_permutations = []

    # make 26 sentences, which are "a", "b", ... "z".
    alphabet = list(string.ascii_lowercase)
    alphabet_n = alphabet[:num_sents]
    test_sentences = [TestSentence(text) for text in alphabet_n]

    s_to_reorder = test_sentences

    # the "best" order starts off being the originally selected one
    best_order = s_to_reorder

    num_orderings = math.factorial(len(s_to_reorder))
    if num_orderings > NUM_PERMUTATIONS:
        large_permutations.append((s_to_reorder, len(s_to_reorder)))
        num_orderings = NUM_PERMUTATIONS

    for i in range(0, repeats):
        print("**** i={} ****".format(i))
        perms = get_perms_arr(s_to_reorder, NUM_PERMUTATIONS, include_first)
        for j in range(len(perms)):
            print_sentence_array(perms[j])

    first_order = perms[0]
    best_order_ind = 0


def run_tests():
    unit_test(num_perms=2, repeats=2, include_first=True)
    unit_test(num_perms=2, repeats=2, include_first=False)




