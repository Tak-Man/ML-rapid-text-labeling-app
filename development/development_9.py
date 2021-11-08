# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_test_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv",
                                           filter_data_types=["test"],

                                           random_state=2584)

    source_dir = "../output/"

    vectorizer = pickle.load(open(source_dir + "fitted-vectorizer.pkl", "rb"))
    vectorized_corpus = \
        vectorizer.transform(consolidated_disaster_tweet_data_test_df["tweet_text"])

    vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus, columns=vectorizer.get_feature_names_out())
    corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_test_df["tweet_id"].values]

    label_map = {"hurricane": "Hurricane", "flood": "Flood", "fire": "Fire", "earthquake": "Earthquake"}
    consolidated_disaster_tweet_data_test_df["label"] = \
        consolidated_disaster_tweet_data_test_df["event_type"].map(label_map)

    # print("consolidated_disaster_tweet_data_test_df :")
    # print(consolidated_disaster_tweet_data_test_df.head())
    # print()

    trained_classifier = pickle.load(open(source_dir + "trained-classifier.pkl", "rb"))
    target_names = trained_classifier.classes_
    print("target_names :", target_names)

    # *****************************************************************************************************************
    # print('consolidated_disaster_tweet_data_test_df["label"].value_counts() :')
    # print(consolidated_disaster_tweet_data_test_df["label"].value_counts())

    y_pred = trained_classifier.predict(vectorized_corpus_df)
    predictions_proba = trained_classifier.predict_proba(vectorized_corpus_df)

    y_test = consolidated_disaster_tweet_data_test_df["label"].values
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    accuracy = accuracy_score(y_test, y_pred)

    micro_precision = precision_score(y_test, y_pred, average="micro")
    micro_recall = recall_score(y_test, y_pred, average="micro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    classification_report = classification_report(y_test, y_pred, target_names=target_names)
    print("classification_report :")
    print(classification_report)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    y_test_binarized = label_binarize(y_test, classes=target_names)
    y_score = trained_classifier.decision_function(vectorized_corpus)
    # y_score = predictions_proba

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f">> '{target_names[i]}' > auc_score-{roc_auc[i]}")

    # *****************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)