import nltk
from nltk.corpus import wordnet as wn
from nltk import download
import pandas as pd
import random

download('wordnet', '/Users/panda/Desktop/aura_ml/tutorial_model/nltk_data')
nltk.data.path.append('/Users/panda/Desktop/aura_ml/tutorial_model/nltk_data')

DATASET = pd.read_csv('./processed_dataset.csv')

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

NO_MUTATE = ['twitter', 'facebook', 'google', 'amazon']

corpus_root = './'


def split_sentence(data):
    return data.split(' ')


pairs = list(zip(DATASET['from'], DATASET['to']))

for pair in pairs:
    word_list = split_sentence(pair[0])
    new_word_list = word_list[:]
    max_index = len(word_list)-1
    start = random.randint(0, max_index)
    max_mutations = 1 if max_index-start == 0 else max_index-start
    mutations = random.randint(1, max_mutations)
    for i, word in enumerate(word_list[start:]):
        if word in NO_MUTATE:
            continue
        synonymSet = set()
        wordNetSynset = wn.synsets(word)
        for synSet in wordNetSynset:
            synonymSet |= set(synSet.lemma_names())
        if not synonymSet:
            continue
        chosen_syn = random.choice(list(synonymSet))
        new_word_list[start+i] = chosen_syn
    print({'before': ' '.join(word_list), 'after': ' '.join(new_word_list)})
