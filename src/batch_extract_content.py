__author__ = 'andrei'


import sys
import utils.functions as utils
from website_data_extractor import WebsiteDataExtractor
from keyterm_extractor import KeyTermExtractor
from keyterm_features import KeyTermFeatures
from keyterm_classifier import RelevanceFilter
import pandas as pd

def batch_extract_content(websiteElementsPath, urlData):
    ## 1) Extract webpage data
    print "[INFO] ==== Extracting webpage data ===="
    data_extractor = WebsiteDataExtractor(websiteElementsPath)

    out = pd.DataFrame(urlData["URL"])

    keyterms = []
    for url in urlData["URL"]:
        print url

        data_dict = data_extractor.crawlPage(url)

        ## 2) Extract candidate keyterms
        print "[INFO] ==== Extracting candidate keyterms ===="
        keyterm_extractor = KeyTermExtractor(data_dict)
        keyterm_extractor.execute()

        #print keyterm_extractor.result_dict
        ## 3) Compute candidate keyterm features
        print "[INFO] ==== Computing candidate keyterm features ===="
        keyterm_feat = KeyTermFeatures(url, data_dict, keyterm_extractor.result_dict, lang=utils.LANG_FR)
        candidate_keyterm_df = keyterm_feat.compute_features()

        selected_keyterms = []
        if not candidate_keyterm_df.empty:
        ## 4) Filter for relevancy and output top 10 keyterms
            print "[INFO] ==== Selecting relevant keyterms ===="
            relevance_filter = RelevanceFilter(candidate_keyterm_df, "dataset/keyterm-classifier-model-v2.pickle", topk=10)
            selected_keyterms = relevance_filter.select_relevant()

        keyterms.append(",".join(selected_keyterms))

    out["keyterms"] = keyterms
    return out



if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage: python batch_extract_content.py <WebsiteElementsPathDef.xml> <urlData.json!!containing a column URL!!> <save_to_csv>"
    else:
        pathDef = sys.argv[1]
        data = pd.read_json(sys.argv[2])
        filename = sys.argv[3]
        if "URL" not in data.columns:
            print "urlData.json doesn't contain a column <URL>"
        else:
            df = batch_extract_content(pathDef, data)
            df.to_json(filename)

