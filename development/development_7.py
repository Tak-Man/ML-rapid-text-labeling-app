import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_dev_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv",
                                           filter_data_types=["dev"],
                                           random_state=2584)
    texts_list_labeled, adj_text_ids_labeled = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_dev_df, limit=10000, keep_labels=True)

    dev_df = pd.DataFrame.from_dict(texts_list_labeled)

    consolidated_disaster_tweet_data_test_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv",
                                           filter_data_types=["test"],
                                           random_state=2584)
    texts_list_unlabeled, adj_text_ids_unlabeled = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_test_df, limit=100000,
                                               keep_labels=False)

    test_df = pd.DataFrame.from_dict(texts_list_unlabeled)

    combined_df = \
        pd.concat([dev_df, test_df], ignore_index=True).reset_index(drop=True)

    print("combined_df :")
    print(combined_df.head())
    print(combined_df.tail())

    combined_df_adj = combined_df.rename(columns={"id": "tweet_id", "text": "tweet_text", "label": "assigned_label"})
    all_texts_json, adj_text_ids = utils.convert_demo_data_into_list_json(combined_df_adj, limit=100000,
                                                                          keep_labels=True,
                                                                          new_label_col_name="assigned_label",
                                                                          old_label_col_name="assigned_label"
                                                                          )

    print("all_texts_json :", all_texts_json[:5])

    print('combined_df["label"].value_counts() :')
    print(combined_df["label"].value_counts())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=200)
    vectorized_corpus = \
        vectorizer.fit_transform(combined_df["text"])

    # vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus,
    #                                                          columns=vectorizer.get_feature_names_out())
    corpus_text_ids = [str(x) for x in combined_df["id"].values]

    # ******************************************************************************************************************
    y_classes = ["earthquake", "fire", "flood", "hurricane", "other"]

    fitted_classifier = utils.fit_classifier(sparse_vectorized_corpus=vectorized_corpus,
                                             corpus_text_ids=corpus_text_ids,
                                             texts_list_labeled=texts_list_labeled,
                                             y_classes=y_classes,
                                             classifier_list=[])

    updated_texts_list, updated_texts_list_list = \
        utils.label_all(fitted_classifier=fitted_classifier,
                        sparse_vectorized_corpus=vectorized_corpus,
                        corpus_text_ids=corpus_text_ids,
                        texts_list=all_texts_json,
                        label_only_unlabeled=True,
                        sub_list_limit=50,
                        update_in_place=False,
                        texts_list_list=[])

    print("updated_texts_list :")
    print(updated_texts_list[:5])
    # ******************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)

