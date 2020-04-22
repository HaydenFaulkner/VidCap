import nltk
from nltk.corpus import wordnet

# can comment out the 2 downloads once done...
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def id_to_syn(wn_id):
    return wordnet.synset_from_pos_and_offset('n', int(wn_id[1:]))


def word_to_syn(word):
    return wordnet.synsets(word)[0]


def syn_to_id(synset):
    return 'n0' + str(synset.offset())


def syn_to_word(synset):
    return synset.name().split('.')[0]


def get_synonyms(synset):
    return list(set([lm.name() for lm in synset.lemmas()]))


def extract_nouns_verbs(sentence, unique=True):
    """
    Extracts the noun and verb words from a sentence
    :param sentence: The sentence
    :param unique: Should the return lists have only unique entries, even when same noun/verb appears twice in sentence
    :return: words list, nouns list, verbs list
    """
    nouns = []
    verbs = []
    words = []

    for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        words.append(word)
        if pos.startswith('NN'):
            nouns.append(word)
        elif pos.startswith('VB'):
            if word not in ['is', 'are', 'has', 'be']:
                verbs.append(word)

    if unique:
        nouns = list(set(nouns))
        verbs = list(set(verbs))
        words = list(set(words))

    return words, nouns, verbs


def parse(sentence):
    """
    Removes everything but alpha characters and spaces, transforms to lowercase
    :param sentence: the sentence to parse
    :return: the parsed sentence
    """
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return ''.join(filter(whitelist.__contains__, sentence)).lower()
