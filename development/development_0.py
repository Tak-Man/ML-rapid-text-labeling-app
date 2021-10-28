import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv")
    print("consolidated_disaster_tweet_data_df :")
    print(consolidated_disaster_tweet_data_df.head())
    print()

    all_texts_json, adj_text_ids = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=100000)
    print("all_texts_json :")
    print(all_texts_json[:5])
    print()

    corpus_text_id = "798262465234542592"
    print('consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["tweet_id"] == corpus_text_id] :')
    print(consolidated_disaster_tweet_data_df[consolidated_disaster_tweet_data_df["tweet_id"] == corpus_text_id])
    print()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
    vectorized_corpus = \
        vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"])

    vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus, columns=vectorizer.get_feature_names_out())
    corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]

    # *****************************************************************************************************************
    # Enter the development code here.
    # Enter the development code here.
    # Enter the development code here.
    # *****************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)