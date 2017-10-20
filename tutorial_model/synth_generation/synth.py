import nltk
from nltk.corpus import wordnet as wn
from nltk.metrics import distance
from nltk.wsd import lesk
from nltk import download
import pandas as pd
import random
import itertools
import json
from collections import defaultdict
from random import shuffle
import csv

download('wordnet', './nltk_data')
download('averaged_perceptron_tagger')
nltk.data.path.append('./nltk_data')

DATASET = pd.read_csv('../data_files/processed_dataset.csv')

SWAPS = {
    "google ": "amazon",
    "games ": "marketplace",
    "players ": "shoppers",
    "playerid ": "shopperid",
    "plus ": "prime",
    "achievements ": "shoppinglist",
    "achievementid ": "itemid",
    "twitter ": "instagram",
    "friendships ": "followers",
    "favorites ": "liked",
    "ads ": "adverts",
    "facebook ": "myspace",
    "graph ": "web",
    "page ": "space",
    "feeds ": "lists",
    "contacts ": "customers",
    "linkid ": "linkdata",
    "pageid ": "spaceid",
    "statuses ": "pictures",
    "messageid ": "postid",
}

NO_MUTATE = ['twitter', 'facebook', 'google', 'amazon', 'aura', 'i', 'was', 'am', 'are']

IGNORE_POS_TAGS = [ 'PRP$', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'MD', 'LS', 'PDT', 'PRP', 'PDT', 'RP', 'SYM', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB' ]

PTB_TO_WN_MAP = {
    'JJ' : 'a',
    'JJR' : 'a',
    'JJS' : 'a',
    'NN' : 'n',
    'NNS' : 'n',
    'NNP' : 'n',
    'NNPS' : 'n',
    'RB' : 'r',
    'RBR' : 'r',
    'RBS' : 'r',
    'VB' : 'v',
    'VBD' : 'v',
    'VBG' : 'v',
    'VBN' : 'v',
    'VBP' : 'v',
    'VBZ' : 'v',
}

corpus_root = './'


def split_sentence(data):
    return data.split()

def handle_missing_synonyms():
    pass
def choose_synonyms():
    pass
pairs = list(zip(DATASET['from'], DATASET['to']))
frequencies = {}
print('...Loading bigram frequencies...')
with open('freqs.json', 'r') as f:
    for line in f:
        word_pair = json.loads(line)
        frequencies[tuple(word_pair['key'])] = word_pair['value']
generated_sentences = {}
print('Generating data...')
for _ in range(0, 1000):
    for pair in pairs:
        generated_sentences[pair[0]] = { 'before': pair[0], 'query': pair[1] }
        word_list = split_sentence(pair[0])
        tagged_words = nltk.pos_tag(word_list)
        new_word_list = word_list[:]
        max_index = len(word_list)-2
        start = random.randint(0, max_index)
        max_mutations = 1 if max_index-start == 0 else max_index-start
        mutations = random.randint(1, max_mutations)
        for i in range(0, len(word_list[start:])-1, 2):
            words = word_list[start+i:start+i+2]
            parts_of_speech = [PTB_TO_WN_MAP.get(x[1]) for x in tagged_words[start+i:start+i+2]]
            synonym_set = set()
            word1_synset = lesk(word_list, words[0], parts_of_speech[0])
            word2_synset = lesk(word_list, words[1], parts_of_speech[1])
            if (words[0] in NO_MUTATE or parts_of_speech[0] is None) and (words[1] in NO_MUTATE or parts_of_speech[1] is None):
                continue
            if words[0] in NO_MUTATE or parts_of_speech[0] is None:
                lemma_names = []
                if word2_synset is None:
                    potential_swap = SWAPS.get(words[1])
                    if potential_swap is None:
                        continue
                    lemma_names = [ potential_swap ]
                else:
                    lemma_names = word2_synset.lemma_names()
                lemma_tup = itertools.product([words[0]], lemma_names)
            elif words[1] in NO_MUTATE or parts_of_speech[1] is None:
                lemma_names = []
                if word1_synset is None:
                    potential_swap = SWAPS.get(words[0])
                    if potential_swap is None:
                        continue
                    lemma_names = [ potential_swap ]
                else:
                    lemma_names = word1_synset.lemma_names()
                lemma_tup = itertools.product( lemma_names, [words[1]] )
            else:
                if word1_synset is None and word2_synset is None:
                    continue
                if word1_synset is None:
                    lemma1_names = [words[0]]
                else:
                    lemma1_names = word1_synset.lemma_names()
                if word2_synset is None:
                    lemma2_names = [words[1]]
                else:
                    lemma2_names = word2_synset.lemma_names()
                lemma_tup = itertools.product(
                    lemma1_names, lemma2_names
                )
            synonym_set |= set(lemma_tup)
            if not synonym_set:
                continue
            chosen_syn_pair = None
            synonym_set = list(synonym_set)
            shuffle(synonym_set)
            for syns in synonym_set:
                try:
                    tmp_syn_pair = {'k': syns, 'v': frequencies[syns]}
                except:
                    tmp_syn_pair = {'k': ('UNK', 'UNK'), 'v': -1}
                if distance.edit_distance(tmp_syn_pair['k'][0], words[0]) < 3 and distance.edit_distance(tmp_syn_pair['k'][1], words[1]) < 3:
                    continue
                if tmp_syn_pair['k'] == tuple(word_list):
                    continue
                if chosen_syn_pair is None or chosen_syn_pair['k'] == ('UNK', 'UNK'):
                    chosen_syn_pair = tmp_syn_pair
                    if chosen_syn_pair['k'] == ('UNK', 'UNK'):
                        continue
                    break
            if chosen_syn_pair is None or chosen_syn_pair['k'] == ('UNK', 'UNK'):
                continue
            new_word_list[start+i] = chosen_syn_pair['k'][0]
            new_word_list[start+i+1] = chosen_syn_pair['k'][1]
            before = ' '.join(word_list)
            after = ' '.join(new_word_list)
            if before == after:
                continue
            generated_sentences[after] = { 'before': before, 'query': pair[1] }

print(f'..{len(generated_sentences.items())} items.')

with open('synthdata.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['input', 'output'])
    for k,v in generated_sentences.items():
        writer.writerow([k, v['query']])

print('....Done')
