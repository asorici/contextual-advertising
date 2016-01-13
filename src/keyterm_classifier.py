import numpy as np
import ioData as io
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from patsy import dmatrices

class KeytermClassifier(object):
    TRAIN_DATASET_SIZE = 10000
    TEST_DATASET_SIZE = 5000

    def __init__(self, train_df_file, test_df_file):
        self.train_df_file = train_df_file
        self.test_df_file = test_df_file


    def prepare_training_set(self):
        self.train_df = io.readData(self.train_df_file)

        self.pos_train_df = self.train_df[self.train_df['relevant'] == True].sample(KeytermClassifier.TRAIN_DATASET_SIZE, random_state=1)
        self.neg_train_df = self.train_df[self.train_df['relevant'] == False].sample(KeytermClassifier.TRAIN_DATASET_SIZE, random_state=1)
        self.selected_train_df = pd.concat([self.pos_train_df, self.neg_train_df])

        #self.y, self.X = dmatrices('relevant ~ cvalue + tf + df + tfidf + doc_pos + is_title + is_url + \
        #                           is_anchor + is_description + is_first_par + is_last_par + is_img_desc',
        #                           self.selected_train_df, return_type = "dataframe")
        #self.y = np.ravel(self.y)
        # self.X = self.selected_train_df.drop(['relevant', 'doc_url', 'term'], axis = 1)
        self.X = self.selected_train_df.drop(['relevant', 'doc_url', 'term', 'df', 'tfidf'], axis = 1)
        self.y = self.selected_train_df['relevant']


    def prepare_test_set(self):
        self.test_df = io.readData(self.test_df_file)

        self.pos_test_df = self.test_df[self.test_df['relevant'] == True].sample(KeytermClassifier.TEST_DATASET_SIZE, random_state=1)
        self.neg_test_df = self.test_df[self.test_df['relevant'] == False].sample(KeytermClassifier.TEST_DATASET_SIZE, random_state=1)
        self.selected_test_df = pd.concat([self.pos_test_df, self.neg_test_df])

        # self.X_test = self.selected_test_df.drop(['relevant', 'doc_url', 'term'], axis = 1)
        self.X_test = self.selected_test_df.drop(['relevant', 'doc_url', 'term', 'df', 'tfidf'], axis = 1)
        self.y_test = self.selected_test_df['relevant']


    def fit(self):
        # self.model = linear_model.LogisticRegression(C=1e3)
        # self.model.fit(self.X, self.y)
        # self.model.score(self.X, self.y)

        X = self.X.copy()
        X['intercept'] = 1

        logit = sm.Logit(self.y, X)
        self.model = logit.fit()
        print self.model.summary()


    def test(self):
        X = self.X_test.copy()
        X['intercept'] = 1

        y_pred = self.model.predict(X)
        print classification_report(self.y_test, (y_pred > 0.5).astype(bool))

    def extract_test_keywords(self, top_k = 10):
        if self.model is None:
            raise ValueError("Untrained classifier! No model exists.")

        X = self.X_test.copy()
        X['intercept'] = 1

        test_df = self.selected_test_df.copy()
        test_df['relevant_pred'] = self.model.predict(X)

        extracted_keyterms = []

        test_doc_urls = test_df['doc_url'].unique()
        for url in test_doc_urls:
            doc_df = test_df[test_df['doc_url'] == url].copy()
            doc_df.sort_values(["relevant_pred", "cvalue"], ascending=[False,False], inplace=True)

            topk_keyterms = ",".join(doc_df[:top_k]['term'].values)
            extracted_keyterms.append((url, topk_keyterms))

        return pd.DataFrame(extracted_keyterms, columns=["url", "extracted_keyterms"])


    def predict_keyterms(self, doc):
        """
        :param doc: Document of type create_term_dataset.Document with the term features precomputed
        :return: the list of predicted keyterms
        """
        term_dataset = []
        term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos']

        for term in doc.relevant_terms:
            term_dataset.append([term.original, doc.url, term.cvalue, term.tf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position])

        term_df = pd.DataFrame(term_dataset, columns=term_df_headers)
        X = term_df.copy()
        X = X.drop(['doc_url', 'term'], axis = 1)
        X['intercept'] = 1

        term_df['relevant_pred'] = self.model.predict(X)
        term_df.sort_values('relevant_pred', inplace=True, ascending=False)

        return term_df[term_df['relevant_pred'] > 0.5], term_df[term_df['relevant_pred'] <= 0.5]


def extract_test_keywords():
    cl = KeytermClassifier("dataset/term-feature-train-dataset.json", "dataset/term-feature-test-dataset.json")

    print "Preparing training set ..."
    cl.prepare_training_set()

    print "Training model ..."
    cl.fit()

    print "Preparing test set ..."
    cl.prepare_test_set()

    print "Extracting keywords ..."
    return cl.extract_test_keywords()

def extracted_keyterms_overlap(row):
    gr_keywords = row['keywords'].lower().split(",")
    our_keyterms = row['extracted_keyterms'].split(",")

    overlap = 0
    for w in gr_keywords:
        s = sum(map(lambda x: x.count(w), our_keyterms))

        if s > 0:
            overlap += 1

    return float(overlap) / float(len(gr_keywords))

# df_raw.loc[map(lambda s: s, df_raw['textZone'].values)]

def has_jwplayer(paragraphs):
    for p in paragraphs:
        s = sum(map(lambda x : x.count("jwplayer("), p))
        if s > 0:
            return True

    return False

if __name__ == "__main__":
    extract_test_keywords()

