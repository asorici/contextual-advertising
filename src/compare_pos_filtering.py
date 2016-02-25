__author__ = "alex"

import pprint
import utils.functions as utils
from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor, KeyTermExtractor2
from keyterm_features import KeyTermFeatures
from keyterm_classifier import RelevanceFilter


if __name__ == "__main__":
    url = 'http://www.generation-nt.com/blackview-a8-smartphone-petit-budget-pas-cher-mwc-2016-actualite-1925283.html'

    ## 1) Extract webpage data
    print "[INFO] ==== Extracting webpage data ===="
    data_extractor = WebsiteDataExtractor("dataset/WebsiteElementsPathDef.xml")
    data_dict = data_extractor.crawlPage(url)

    ## 2) Extract candidate keyterms
    print "[INFO] ==== Extracting candidate keyterms ===="
    keyterm_extractor = KeyTermExtractor(data_dict)
    keyterm_extractor.execute()

    keyterm_extractor2 = KeyTermExtractor2(data_dict, lang="french")
    keyterm_extractor2.execute()

    print "======== Results from Extractor 1 ========"
    pprint.pprint(keyterm_extractor.result_dict)
    # print "Nr t1grams: " + str(len(keyterm_extractor.result_dict['t1gram']['term']))
    # print "Nr t2grams: " + str(len(keyterm_extractor.result_dict['t2gram']['term']))
    # print "Nr t3grams: " + str(len(keyterm_extractor.result_dict['t3gram']['term']))
    # print "Nr t4grams: " + str(len(keyterm_extractor.result_dict['t4gram']['term']))

    print "======== Results from Extractor 2 ========"
    pprint.pprint(keyterm_extractor2.result_dict)
    # print "Nr t1grams: " + str(len(set(keyterm_extractor2.result_dict['t1gram'])))
    # print "Nr t2grams: " + str(len(set(keyterm_extractor2.result_dict['t2gram'])))
    # print "Nr t3grams: " + str(len(set(keyterm_extractor2.result_dict['t3gram'])))
    # print "Nr t4grams: " + str(len(set(keyterm_extractor2.result_dict['t4gram'])))

    print "## Result diffs t1gram "
    pprint.pprint(set(keyterm_extractor2.result_dict['t1gram']) - set(keyterm_extractor.result_dict['t1gram']['term']))

    print "## Result diffs t2gram "
    pprint.pprint(set(keyterm_extractor2.result_dict['t2gram']) - set(keyterm_extractor.result_dict['t2gram']['term']))

    print "## Result diffs t3gram "
    pprint.pprint(set(keyterm_extractor2.result_dict['t3gram']) - set(keyterm_extractor.result_dict['t3gram']['term']))

    print "## Result diffs t4gram "
    pprint.pprint(set(keyterm_extractor2.result_dict['t4gram']) - set(keyterm_extractor.result_dict['t4gram']['term']))
