import nltk

# can comment out the 2 downloads once done...
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


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
