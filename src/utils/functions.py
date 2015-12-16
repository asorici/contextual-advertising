import string, re, nltk
from nltk.corpus import stopwords

## GENERAL CONSTANTS
LANG_EN = "en"
LANG_FR = "fr"
import numpy as np
import pandas as pd


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


def remove_stopwords(text, lang=LANG_EN):
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


def apply_pos_tag(text, lang=LANG_EN):
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


def split_into_words(text):
    """
    :param text: Sentence
    :return: list of words split by space
    """
    assert isinstance(text, unicode)
    return text.split()


def clean_string(text):
    import re
    text = u''.join(map(lambda x: x if unicode.isalnum(x) or x == " " else " ", text))
    multiSpacePattern = re.compile(r'\s+')
    text = re.sub(multiSpacePattern, " ", text)
    text.strip()
    return text

class TermFeatures:
    def __init__(self, df, urlTokenColum="linkTokens", titleColumn="title", descriptionColumn="resume", textzoneColumn="textZone", anchorColumn="anchors", imgDescColumn="alternateTxt"):
        self.df = df

        if (type(df) != pd.DataFrame):
            raise ValueError("Df is not a pandas.Dataframe")
        else:
            if (urlTokenColum not in df.columns) or (titleColumn not in df.columns) or (descriptionColumn not in df.columns) or (textzoneColumn not in df.columns) or (anchorColumn not in df.columns):
                raise ValueError("Dataframe doesn't contain necessary columns!")

        if (imgDescColumn not in df.columns):
            if ("alternateTxtDesc" in df.columns) and ("alternateTxtZone" in df.columns):
                df["alternateTxt"] = df.alternateTxtZone + df.alternateTxtDesc.apply(lambda x: [x])
            else:
                raise ValueError("Dataframe doesn't contain necessary columns!")

        self.urlTokenColum = urlTokenColum
        self.titleColumn = titleColumn
        self.descriptionColumn = descriptionColumn
        self.textzoneColumn = textzoneColumn
        self.anchorColumn = anchorColumn
        self.imgDescColumn = imgDescColumn

    @staticmethod
    def split_term_grams(term):
        #split only by space (nothing else! like , . ! ? ...)
        grams = []
        sentence = term.split()
        for n in xrange(1, len(sentence)):
            grams = grams + [sentence[i : i+n] for i in xrange(len(sentence)- n+1)]
        return np.ravel(grams)

    #not sensitive to order
    def isURL(self, term, idx):
        if idx not in self.df.index:
            return 0

        line = self.df.loc[idx, self.urlTokenColum]
        if (type(term) is not list) and (type(term) is not np.ndarray):
            ans = line.count(term)
        else:
            ans = reduce(lambda x, y: line.count(x) + line.count(y), term)
        return ans


    def isTitle(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return self.df.loc[idx, self.titleColumn].count(term)

    def isDescription(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return self.df.loc[idx, self.descriptionColumn].count(term)

    def isAnchor(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.anchorColumn]))

    def isImgDesc(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.imgDescColumn]))

    def isFirstParagraph(self, term, idx):
        if idx not in self.df.index:
            return 0

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.textzoneColumn][0]))

    def isLastParagraph(self, term, idx):
        if idx not in self.df.index:
            return 0

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), self.df.loc[idx, self.textzoneColumn][-1]))

    def posInDoc(self, term, idx):
        if idx not in self.df.index:
            return None

        if len(self.df.loc[idx, self.textzoneColumn]) < 1:
            return None

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        y = ' '.join(TermFeatures.flatten_list(self.df.loc[idx, self.textzoneColumn]))

        if term in y:
            return float(y.index(term))/float(len(y))
        else:
            return None

    def isTextZone(self, term, idx):
        if idx not in self.df.index:
            return 0

        if (type(term) is list) or (type(term) is np.ndarray):
            term = " ".join(term)

        return sum(map(lambda x: x.count(term), TermFeatures.flatten_list(self.df.loc[idx, self.textzoneColumn])))

    @staticmethod
    def flatten_list(x):
        if isinstance(x, (list, np.ndarray)):
            return [a for i in x for a in TermFeatures.flatten_list(i)]
        else:
            return [x]
























"""
:param text:
:return:

util for:

regex match if containing anything else than :: pattern = r'[^a-zA-Z0-9-_]'
regex matches if contains letters a-zA-Z :: pattern = r'[a-zA-Z]'
unicode.strip() -> strip whitespacest
pd.Series.str.split() -> split in words
evaluate list from string -> ast.literal_eval()

#remove from list elem that do not contain letters
df.keywords.apply(lambda ls: [x for x in ls if (re.search('[a-zA-Z]', unidecode(x)))])

#keep unique
df.keywords = df.keywords.apply(np.unique)

df.keywords.apply(lambda ls: [x for x in ls if len(x) > 1])

#match link ending -12341234-0.html
[-[0-9]+]*.html

#----url link----
df.linkTokens = df.linkTokens.str.replace("http://www.generation-nt.com/", "")
df.linkTokens = df.linkTokens.str.replace(r"[-[0-9]+]*.html", "")
df.linkTokens = df.linkTokens.str.split("-")
df.linkTokens.apply(lambda x: x.pop())

#----title-----
df.title = df.title.apply(clean_string)

#----alternateTxtDesc----
df.alternateTxtDesc = df.alternateTxtDesc.apply(clean_string)
df.alternateTxtDesc = df.alternateTxtDesc.apply(unicode.strip)

#----alternateTxtZone----
df.alternateTxtZone = df.alternateTxtZone.apply(lambda x: map(clean_string, x))
df.alternateTxtZone = df.alternateTxtZone.apply(lambda x: map(unicode.strip, x))

#---anchors----
anc = df.anchors
anc = anc.apply(lambda x: map(lambda y: y.values(),x))
anc = anc.apply(lambda x: reduce(lambda o,p: o + p, x) if x else [])
anc = anc.apply(lambda x: filter(lambda y: True if y.startswith("http://www.generation-nt.com/") else False, x))
anc = anc.apply(lambda x: map(lambda y: y.replace("http://www.generation-nt.com/", ""), x))
anc = anc.apply(lambda x: map(lambda y: re.sub(r"[-[0-9]*]*.html", "", y), x))
anc = anc.apply(lambda x: map(lambda y: re.sub(r"[-_+\/]", " ", y), x))
"""


