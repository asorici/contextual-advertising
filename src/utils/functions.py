
def correct_spelling(word, lang = "en"):
    """
    :param word: the word requiring correction
    :param lang: language of the word
    :return: the most probable correct version of the input word
    """
    pass


def generate_ngrams(text, size):
    """
    :param text: Unicode string or list of words
    :param size: Maximum size of n-gram window
    :return: Generate n-grams of length <= `size' starting from the input text.
        The function returns a list of word list, where each word list represents a n-gram instance.
    """
    pass

def remove_stopwords(text, lang = "en"):
    """
    :param text: Unicode string or list of words
    :param lang: Language for which to filter stopwords
    :return: Return list of words (in original sequence) from which stopwords are removed
    """
    pass

def apply_pos_tag(text, lang = "en"):
    """
    :param text: Single word or sentence
    :param lang: language for which to apply POS-tagging
    :return: The original sentence where each word tagged (using /<tag>) with its corresponding part of speech
    """