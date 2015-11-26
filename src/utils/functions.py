LANG_EN = "en"
LANG_FR = "fr"

def correct_spelling(word, lang = LANG_EN):
    import enchant
    """
    :param word: the word requiring correction
    :param lang: language of the word
    :return: the most probable correct version of the input word
    """
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
    :return: Return list of words (in original sequence) from which stopwords are removed
    """
    pass

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
