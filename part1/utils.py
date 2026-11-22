import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

KEYBOARD_NEIGHBORS = {
    "q": "wa",     "w": "qeas",    "e": "wrsd",    "r": "etdf",    "t": "ryfg",
    "y": "tugh",   "u": "yihj",    "i": "uojk",    "o": "ipkl",    "p": "ol",
    "a": "qwsz",   "s": "qweadzx", "d": "werfsxc", "f": "ertdgcv", "g": "rtyfhvb",
    "h": "tyugjbn","j": "yuihknm", "k": "uiojlm",  "l": "iopk",
    "z": "asx",    "x": "zsdc",    "c": "xdfv",    "v": "cfgb",    "b": "vghn",
    "n": "bhjm",   "m": "njk",
}

def custom_transform(example):
    words = example['text'].split()
    new_words = []
    for word in words:
        if random.random() < 0.5:
            if len(word) < 3:
                new_words.append(word)
                continue
            action = random.choice(['typo', 'swap', 'delete', 'insert'])
            letter_i = random.randrange(len(word))
            if action == 'typo':
                if word[letter_i].lower() in KEYBOARD_NEIGHBORS:
                    replacement = random.choice(KEYBOARD_NEIGHBORS[word[letter_i].lower()])
                    new_words.append(word[:letter_i] + replacement + word[letter_i + 1:])
                else:
                    new_words.append(word)
            elif action == 'swap':
                if letter_i < len(word)-1:
                    new_words.append(word[:letter_i] + word[letter_i+1] + word[letter_i] + word[letter_i+2:])
                else:
                    new_words.append(word[:letter_i-1] + word[letter_i] + word[letter_i-1])
            elif action == 'delete':
                new_words.append(word[:letter_i] + word[letter_i + 1:])
            else:
                new_words.append(word[:letter_i] + word[letter_i] + word[letter_i] + word[letter_i + 1:])
        else:
            new_words.append(word)
    example['text'] = ' '.join(new_words)
    return example
