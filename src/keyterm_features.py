__author__ = 'alex'

import json, pandas as pd, numpy as np
import utils.functions as utils
from website_data_extractor import WebsiteDataExtractor

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 400)

def read_raw_data(filename):
    df = pd.read_json(filename)
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df


def is_meta_keyword(term, meta_keyword_list):
    import nltk
    stemmer = nltk.stem.snowball.FrenchStemmer()

    def mysplit(x):
        if " " in x:
            return x.split()
        return x

    split_keywords = map(mysplit, meta_keyword_list)
    if term.length == 1:
        for t in split_keywords:
            if isinstance(t, basestring) and t == term.split:
                return True
            elif isinstance(t, list) and term.split in t:
                return True

        return False
    else:
        for w in term.split:
            for t in split_keywords:
                if isinstance(t, basestring) and t == w:
                    return True
                elif isinstance(t, list) and w in t:
                    return True

        return False


class KeyTermFeatures(object):
    def __init__(self, url, website_data_dict, keyterm_candidate_dict, lang = utils.LANG_EN):
        self.website_data_dict = website_data_dict
        self.keyterm_candidate_dict = keyterm_candidate_dict

        self.url = url
        self.lang = lang


    def compute_features(self):
        ## create document
        doc_text = ''.join(self.website_data_dict[WebsiteDataExtractor.MAIN_TEXT])
        doc = utils.Document(self.url, doc_text, lang=self.lang)

        doc_terms = []
        gram_keys = [('t1gram', 1), ('t2gram', 2), ('t3gram', 3), ('t4gram', 4)]
        for key, length in gram_keys:
            raw_terms = self.keyterm_candidate_dict[key]["term"]
            cvalues = self.keyterm_candidate_dict[key]["cval"]

            for i in range(len(raw_terms)):
                term = utils.Term(raw_terms[i], doc, length, lang=self.lang)
                term.cvalue = cvalues[i]

                # add term to doc
                doc_terms.append(term)

        doc.load_relevant_terms(doc_terms)

        ## compute features of terms in document
        # create df from website_data_dict
        website_data_df = pd.DataFrame.from_dict({self.url : self.website_data_dict}, orient = 'index')
        feature_extractor = utils.TextualFeatureExtractor(website_data_df,
                                        urlTokenColum=WebsiteDataExtractor.URL_TOKENS,
                                        titleColumn=WebsiteDataExtractor.TITLE,
                                        descriptionColumn=WebsiteDataExtractor.SUMMARY,
                                        textzoneColumn=WebsiteDataExtractor.MAIN_TEXT,
                                        anchorColumn=WebsiteDataExtractor.HYPERLINKS,
                                        imgDescColumn=WebsiteDataExtractor.IMAGE_CAPTION)

        # 1) compute TF
        doc.compute_tf()

        term_dataset = []
        for term in doc.relevant_terms:
            # 2) compute linguistic features
            term.set_textual_feature_extractor(feature_extractor)
            term.extract_textual_features()

            term_dataset.append([term.original, doc.url, term.cvalue, term.tf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position])

        term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos']

        return pd.DataFrame(term_dataset, columns=term_df_headers)



def create_documents(json_file, raw_df):
    # load result of c-value experiment
    cvalRes = None
    with open(json_file) as fp:
        cvalRes = json.load(fp, encoding="utf-8")

    if cvalRes is None:
        raise ValueError("This shouldn't happen to a dog!")

    doc_list = []
    global_term_dict = {}
    empty_docs = 0

    # walk through urls in json dump
    nr_docs = len(cvalRes.keys())
    doc_items = cvalRes.items()

    #for url in cvalRes.keys():
    for idx in range(len(doc_items)):
        # get url
        url = doc_items[idx][0]

        # get Document for given URL
        textList = raw_df[raw_df['link'] == url]['textZone'].values
        doc_text = ''.join(utils.TextualFeatureExtractor.flatten_list(textList))
        doc_text = doc_text.lower()

        if doc_text:
            doc = utils.Document(url, doc_text, lang=utils.LANG_FR)

            # collect all terms for given document
            doc_terms = []
            term_dict = cvalRes[url]

            gram_keys = [('t1gram', 1), ('t2gram', 2), ('t3gram', 3), ('t4gram', 4)]
            for key, length in gram_keys:
                raw_terms = term_dict[key]["term"]
                cvalues = term_dict[key]["cval"]

                for i in range(len(raw_terms)):
                    term = utils.Term(raw_terms[i], doc, length, lang=utils.LANG_FR)
                    term.cvalue = cvalues[i]

                    # add term to doc
                    doc_terms.append(term)

                    # update its document frequency
                    if term in global_term_dict:
                        global_term_dict[term] = global_term_dict.get(term) + 1
                    else:
                       global_term_dict[term] = 1


            doc.load_relevant_terms(doc_terms)
            doc_list.append(doc)
        else:
            print "Empty doc: " + url
            empty_docs += 1


    print "Total empty docs = " + str(empty_docs)

    return doc_list, global_term_dict


def compute_term_features(doc_list, global_term_dict, df_raw, store_filename):
    import math, tabulate, ioData

    # reindex df_raw by link column
    df = df_raw.set_index('link')
    feature_extractor = utils.TextualFeatureExtractor(df)

    nr_docs = len(doc_list)
    nr_terms = reduce(lambda x, y: x + len(y.relevant_terms), doc_list, 0)

    term_dataset = []

    for doc in doc_list:
        print "======== " + doc.url + " ========"
        meta_keywords = df.loc[doc.url]['keywords']

        # 1) compute TF
        # print "Computing TF ... "
        doc.compute_tf()

        # print "Document parsed: "
        # print doc.transformed

        # print "Computing DF, TFIDF and Textual Features ... "
        for term in doc.relevant_terms:
            # 2) compute DF and TFIDF - use global_term_dict
            term.df = global_term_dict[term]
            term.tfidf = term.tf * math.log(1 + float(nr_docs) / float(term.df), 2)
            # print tabulate.tabulate([[term, term.cvalue, term.tf, term.df, term.tfidf]], headers=('term', 'cval', 'tf', 'df', 'tfidf'))
            # print ""

            # 3) compute linguistic features
            term.set_textual_feature_extractor(feature_extractor)
            term.extract_textual_features()

            # print tabulate.tabulate([[term, term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
            #                          term.is_img_caption, term.is_anchor, term.doc_position]],
            #                         headers=('term', 'is_title', 'is_url', 'is_first_par', 'is_last_par', 'is_description', 'is_img_desc', 'is_anchor', 'doc_pos'))

            # 4) check if term is in meta keywords (i.e is relevant)
            term.is_keyword = is_meta_keyword(term, meta_keywords)
            # print ":: IS RELEVANT = " + str(term.is_keyword)
            # print ("\n\n")

            term_dataset.append([term.original, doc.url, term.cvalue, term.tf, term.df, term.tfidf,
                                 term.is_title, term.is_url, term.is_first_par, term.is_last_par, term.is_description,
                                 term.is_img_caption, term.is_anchor, term.doc_position, term.is_keyword])

    term_df_headers = ['term', 'doc_url', 'cvalue', 'tf', 'df', 'tfidf', 'is_title', 'is_url',
                        'is_first_par', 'is_last_par', 'is_description',
                        'is_img_desc', 'is_anchor', 'doc_pos', 'relevant']

    term_df = pd.DataFrame(term_dataset, columns=term_df_headers)
    ioData.writeData(term_df, store_filename)



def create_term_train_test_dataset(global_term_feature_file, extracted_test_terms_file, train_feature_file, test_feature_file):
    import ioData as io
    df = io.readData(global_term_feature_file)

    cvalRes = None
    with open(extracted_test_terms_file) as fp:
        cvalRes = json.load(fp, encoding="utf-8")

    test_urls = cvalRes.keys()
    test_df = df.loc[df['doc_url'].isin(test_urls)]
    io.writeData(test_df, test_feature_file)

    train_df = df.loc[~df.index.isin(test_df.index)]
    io.writeData(train_df, train_feature_file)
    #return train_df, test_df


# for term in test_doc_list[0].relevant_terms:
#     print term.transformed

if __name__ == "__main__":
    df_raw = read_raw_data("dataset/preProc2_lower.json")
    doc_list, global_term_dict = create_documents("dataset/extracted_terms_all_v2.json", df_raw)

    term_feature_file = "dataset/term-feature-dataset-v2.json"
    compute_term_features(doc_list, global_term_dict, df_raw, term_feature_file)

    create_term_train_test_dataset(term_feature_file, "dataset/extracted_terms_grapeshot_common_v2.json", "dataset/term-feature-train-dataset-v2.json", "dataset/term-feature-test-dataset-v2.json")