import string, re, nltk
from nltk.corpus import stopwords

## GENERAL CONSTANTS
LANG_EN = "en"
LANG_FR = "fr"

word_split_regex = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
stopwords_en = set(stopwords.words('english'))
stopwords_fr = set(stopwords.words('french'))


def correct_spelling(word, lang = LANG_EN):
    """
    :param word: the word requiring correction
    :param lang: language of the word
    :return: the most probable correct version of the input word
    """
    import enchant
    d = enchant.request_dict(lang)
    if not d.check(word):
        return d.suggest(word)[0]

    return word


def generate_ngrams(text, size):
    """
    :param text: Unicode string or list of words
    :param size: Maximum size of n-gram window
    :return: Generate n-grams of length <= `size' starting from the input text.
        The function returns a list of word list, where each word list represents a n-gram instance.
    """
    pass

def remove_stopwords(text, lang = LANG_EN):
    """
    :param text: Unicode string or list of words
    :param lang: Language for which to filter stopwords
    :return: Return list of words (in original sequence) from which stopwords are removed or None if the input was not a string or list of strings
    """
    words = []
    if isinstance(text, basestring):
        # split the text into sequence of words
        words = word_split_regex.split(text)
    elif isinstance(text, (list, tuple)):
        words = list(text)

    if words:
        if lang == LANG_EN:
            return [w for w in words if w and w not in stopwords_en]
        elif lang == LANG_FR:
            return [w for w in words if w and w not in stopwords_fr]
        else:
            return None
    else:
        return None

def apply_pos_tag(text, lang = LANG_EN):
    """
    :param text: Single word or sentence
    :param lang: language for which to apply POS-tagging
    :return: A list of tuples of the (word,tag) pairs
    """
    if lang == LANG_FR:
        from pattern.text.fr import tag
    else:
        from pattern.text.en import tag

    return tag(text)
