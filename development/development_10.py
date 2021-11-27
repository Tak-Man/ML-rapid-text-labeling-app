import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime
import copy
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import csv
import re


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_train_data = newsgroups_train.data
    print("newsgroups_train_data[:2] :", newsgroups_train_data[:2])
    print("len(newsgroups_train_data) :", len(newsgroups_train_data))

    newsgroups_test = fetch_20newsgroups(subset='test')
    newsgroups_test_data = newsgroups_test.data
    print("len(newsgroups_test_data) :", len(newsgroups_test_data))

    newsgroups_data = copy.deepcopy(newsgroups_train_data)
    newsgroups_data.extend(newsgroups_test_data)
    print("len(newsgroups_data) :", len(newsgroups_data))



    def basic_extraction_text_20newsgroup(newsgroups_data):
        regex = r"Lines: [0-9]{1,2}\s{1,3}(.*)"
        newsgroups_data = [x.replace("\n", " ") for x in newsgroups_data]
        regex_results = [re.findall(regex, x) for x in newsgroups_data]
        print("len(regex_results) :", len(regex_results))
        regex_results = [x[0] for x in regex_results if len(x) == 1]
        print("len(regex_results) :", len(regex_results))
        regex_results = [x for x in regex_results if len(x) > 10]
        print("len(regex_results) :", len(regex_results))

        newsgroups_df = pd.DataFrame({"newsgroups_id": ["%0.4d" % x for x in range(0, len(regex_results))],
                                      "newsgroups_text": regex_results})
        newsgroups_df["newsgroups_id"] = (newsgroups_df["newsgroups_id"].astype(int) + 1000).astype(str)

        return newsgroups_df


    newsgroups_df = basic_extraction_text_20newsgroup(newsgroups_data=newsgroups_data)

    print("len(newsgroups_df) :", len(newsgroups_df))
    print("newsgroups_df :", newsgroups_df.head())

    newsgroups_df.to_csv("../../../experimental_data/20newsgroups.csv", index=False)
    # *****************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)

