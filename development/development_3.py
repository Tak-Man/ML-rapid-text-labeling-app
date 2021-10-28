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
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv",
                                           number_samples=100,
                                           filter_data_types=["dev"],
                                           random_state=2584
                                           )
    print("consolidated_disaster_tweet_data_df :")
    print(consolidated_disaster_tweet_data_df.head())
    print()

    all_texts_json, adj_text_ids = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                          limit=100000)

    texts_list_labeled, adj_text_ids_labeled = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=50, keep_labels=True)
    print("texts_list_labeled :")
    print(texts_list_labeled[:5])
    print()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
    vectorized_corpus = \
        vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"])

    vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus, columns=vectorizer.get_feature_names_out())
    corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]

    # ******************************************************************************************************************
    y_classes = ["earthquake", "fire", "flood", "hurricane", "other"]

    fitted_classifier = utils.fit_classifier(sparse_vectorized_corpus=vectorized_corpus,
                                             corpus_text_ids=corpus_text_ids,
                                             texts_list_labeled=texts_list_labeled,
                                             y_classes=y_classes,
                                             classifier_list=[])

    predictions_all = utils.get_all_predictions(fitted_classifier=fitted_classifier,
                                                sparse_vectorized_corpus=vectorized_corpus,
                                                corpus_text_ids=corpus_text_ids,
                                                texts_list=all_texts_json,
                                                top=5,
                                                y_classes=y_classes,
                                                verbose=False,
                                                similar_texts=[])
    print("predictions_all :")
    print(predictions_all[:5])
    print()
    # ******************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)
