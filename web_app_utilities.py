import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing as pp
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter
import itertools
import re
from scipy.stats import entropy
import uuid


pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 200
pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)


RND_SEED = 45822
random.seed(RND_SEED)
np.random.seed(RND_SEED)


def get_disaster_tweet_demo_data(number_samples=None,
                                 filter_data_types=None,
                                 random_state=1144,
                                 source_file="./data/consolidated_disaster_tweet_data.tsv"):
    data_df = pd.read_csv(source_file, sep="\t")

    if filter_data_types:
        data_df = data_df[data_df["data_type"].isin(filter_data_types)]

    if number_samples:
        data_df = data_df.sample(number_samples, random_state=random_state)

    if filter_data_types or number_samples:
        data_df = data_df.reset_index(drop=True)

    return data_df


# def convert_demo_data_into_list(consolidated_disaster_tweet_data_df, limit=50):
#     consolidated_disaster_tweet_data_df["assigned_label"] = "-"
#     consolidated_disaster_tweet_data_df["tweet_id"] = consolidated_disaster_tweet_data_df["tweet_id"].values.astype(str)
#     all_texts = consolidated_disaster_tweet_data_df[["tweet_id", "tweet_text", "assigned_label"]].values.tolist()
#
#     max_length = len(all_texts)
#     if limit < max_length:
#         all_texts_adj = random.sample(all_texts, limit)
#     else:
#         all_texts_adj = all_texts
#
#     return all_texts_adj


def convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=50, keep_labels=False,
                                     shuffle_list=[], random_shuffle=False, random_state=21524,
                                     new_label_col_name="assigned_label",
                                     old_label_col_name="event_type"):
    id_col = "tweet_id"
    text_col = "tweet_text"
    label_col = "assigned_label"

    if keep_labels:
        consolidated_disaster_tweet_data_df[new_label_col_name] = consolidated_disaster_tweet_data_df[old_label_col_name]
    else:
        consolidated_disaster_tweet_data_df[new_label_col_name] = "-"

    consolidated_disaster_tweet_data_df[id_col] = consolidated_disaster_tweet_data_df[id_col].values.astype(str)

    if len(shuffle_list) > 0:
        sort_index = consolidated_disaster_tweet_data_df.index.values
        # consolidated_disaster_tweet_data_df["shuffle"] = shuffle_list

        group_dict = {}
        for group in set(shuffle_list):
            group_dict[group] = []

        for (group, index) in zip(shuffle_list, sort_index):
            group_dict[group].append(index)

        dictionaries = list(group_dict.values())

        sort_indices = []
        for sort_indices_tuple in itertools.zip_longest(*dictionaries):
            # print("len(sort_indices_tuple) :", len(sort_indices_tuple))
            # print("sort_indices_tuple :", sort_indices_tuple)
            temp_list = [x for x in [*sort_indices_tuple] if x is not None]
            sort_indices.extend(temp_list)
            # sort_indices = list(filter(None, sort_indices))

        consolidated_disaster_tweet_data_df = consolidated_disaster_tweet_data_df.iloc[sort_indices, :]

    if len(shuffle_list) == 0 and random_shuffle:
        consolidated_disaster_tweet_data_df = consolidated_disaster_tweet_data_df\
                                                .sample(frac=1, random_state=random_state)\
                                                .reset_index(drop=True)

    all_texts = consolidated_disaster_tweet_data_df[[id_col, text_col, label_col]].values.tolist()

    max_length = len(all_texts)
    if limit < max_length:
        all_texts_adj = random.sample(all_texts, limit)
    else:
        all_texts_adj = all_texts

    all_texts_json = [{"id": text[0], "text": text[1], "label": text[2]} for text in all_texts_adj]
    adj_text_ids = [text[0] for text in all_texts_adj]
    return all_texts_json, adj_text_ids


def update_all_texts(all_texts, text_id, label):
    all_texts_df = pd.DataFrame(all_texts, columns=["tweet_id", "tweet_text", "assigned_label"])
    all_texts_df.loc[all_texts_df["tweet_id"] == str(text_id), "assigned_label"] = label

    all_texts_updated = all_texts_df.values

    return all_texts_updated


def filter_all_texts(all_text, filter_list, exclude_already_labeled=False):
    filtered_all_text = []

    # Slow - 10,000 records - duration 0:00:31.719903
    # for filter_id in filter_list:
    #     for text in all_text:
    #         if text["id"] == filter_id:
    #             filtered_all_text.append(text)

    # Faster - 10,000 records - duration 0:00:07.619622
    # [filtered_all_text.append(text) for text in all_text if text["id"] in filter_list]

    # Fastest - 10,000 records - duration 0:00:00.102955
    all_text_df = pd.DataFrame(all_text)
    filtered_all_text_df = all_text_df[all_text_df["id"].isin(filter_list)]
    # print(">> filter_all_texts > filtered_all_text_df :")
    # print(filtered_all_text_df.head())

    if exclude_already_labeled:
        filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["label"].isin(["-"])]

    filtered_all_text = filtered_all_text_df.to_dict("records")

    return filtered_all_text


def search_all_texts(all_text, include_search_term, exclude_search_term,
                     exclude_already_labeled=False,
                     include_behavior="conjunction",
                     exclude_behavior="disjunction",
                     all_upper=True):

    all_text_df = pd.DataFrame(all_text)

    if exclude_already_labeled:
        filtered_all_text_df = all_text_df[all_text_df["label"].isin(["-"])]
    else:
        filtered_all_text_df = all_text_df

    if include_search_term and len(include_search_term) > 0:
        include_search_terms = include_search_term.split()
        if all_upper:
            if include_behavior == "disjunction":
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).str.upper().apply(
                    lambda text: any(word.upper() in text for word in include_search_terms))]
            else:
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).str.upper().apply(
                    lambda text: all(word.upper() in text for word in include_search_terms))]
        else:
            if include_behavior == "disjunction":
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).apply(
                    lambda text: any(word in text for word in include_search_terms))]
            else:
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).apply(
                    lambda text: all(word in text for word in include_search_terms))]

    if exclude_search_term and len(exclude_search_term) > 0:
        exclude_search_terms = exclude_search_term.split()
        if all_upper:
            if exclude_behavior == "disjunction":
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).str.upper().apply(
                    lambda text: any(word.upper() not in text for word in exclude_search_terms))]
            else:
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).str.upper().apply(
                    lambda text: all(word.upper() not in text for word in exclude_search_terms))]
        else:
            if exclude_behavior == "disjunction":
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).apply(
                    lambda text: any(word not in text for word in exclude_search_terms))]
            else:
                filtered_all_text_df = filtered_all_text_df[filtered_all_text_df["text"].astype(str).apply(
                    lambda text: all(word not in text for word in exclude_search_terms))]

    filtered_all_text = filtered_all_text_df.to_dict("records")
    return filtered_all_text


def update_texts_list(texts_list, sub_list_limit, old_obj_lst=[], new_obj_lst=[], texts_list_list=[]):
    # print("len(texts_list) :", texts_list)
    updated_texts_list = texts_list # .copy()

    if len(old_obj_lst) > 0 or len(new_obj_lst) > 0:
        if len(old_obj_lst) > 0:
            for old_obj in old_obj_lst:
                # print(f"Trying to remove obj : {old_obj}")
                updated_texts_list.remove(old_obj)

        if len(new_obj_lst) > 0:
            for new_obj in new_obj_lst:
                updated_texts_list.append(new_obj)

    texts_list_list.clear()
    updated_texts_list_list = \
        [updated_texts_list[i:i + sub_list_limit] for i in range(0, len(updated_texts_list), sub_list_limit)]
    texts_list_list.extend(updated_texts_list_list)
    # print("len(texts_list_list) :", len(texts_list_list))
    return updated_texts_list, updated_texts_list_list


def update_texts_list_by_id(texts_list, sub_list_limit, updated_obj_lst=[], texts_list_list=[],
                            labels_got_overridden_flag=[],
                            update_in_place=True):
    updated_texts_list_df = pd.DataFrame.from_dict(texts_list) # .copy()

    updated_obj_df = pd.DataFrame.from_dict(updated_obj_lst)
    update_ids = updated_obj_df["id"].values

    labels_to_be_updated = \
        list(set(updated_texts_list_df[updated_texts_list_df["id"].isin(update_ids)]["label"].values))
    if len(labels_to_be_updated) > 1 or (len(labels_to_be_updated) == 1 and "-" not in labels_to_be_updated):
        # "-" indicates an unlabeled text. So, if the labels_to_be_updated have anything other than "-"
        # this indicates that already assigned labeled are going to be overridden.
        overridden = True
    else:
        overridden = False
    labels_got_overridden_flag.clear()
    labels_got_overridden_flag.extend([overridden])

    if update_in_place:
        update_labels = list(set(updated_obj_df["label"].values))
        if len(update_labels) == 1:
            updated_texts_list_df.loc[updated_texts_list_df["id"].isin(update_ids), "label"] = update_labels[0]
        else:
            updated_texts_list_df.loc[updated_texts_list_df["id"].isin(update_ids), "label"] = \
                updated_obj_df["label"].values
    else:
        updated_texts_list_df = updated_texts_list_df[~updated_texts_list_df["id"].isin(update_ids)]
        updated_texts_list_df = updated_texts_list_df.append(updated_obj_df)

    updated_texts_list = updated_texts_list_df.to_dict("records")
    texts_list.clear()
    texts_list.extend(updated_texts_list)

    updated_texts_list_list = \
        [updated_texts_list[i:i + sub_list_limit] for i in range(0, len(updated_texts_list), sub_list_limit)]
    texts_list_list.clear()
    texts_list_list.extend(updated_texts_list_list)
    return updated_texts_list, updated_texts_list_list, overridden


def cosine_similarities(mat):
    # https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    #
    col_normed_mat = pp.normalize(mat.tocsc(), axis=1)
    return col_normed_mat * col_normed_mat.T


def get_all_similarities(sparse_vectorized_corpus, corpus_text_ids):
    # Slow - vectorized_corpus.shape : (76484, 10) - Unable to allocate 43.6 GiB for an array with shape (76484, 76484) and data type float64
    # similarities = cosine_similarity(sparse_vectorized_corpus)
    # similarities_df = pd.DataFrame(similarities, columns=corpus_text_ids)

    # Faster - vectorized_corpus.shape : (76484, 10) - duration 0:01:43.129781
    # similarities = cosine_similarity(sparse_vectorized_corpus, dense_output=False)
    # similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # Faster - vectorized_corpus.shape : (76484, 10) - duration 0:02:03.657139
    # similarities = np.dot(sparse_vectorized_corpus, sparse_vectorized_corpus.T)
    # similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # Fastest - vectorized_corpus.shape : (76484, 10) - duration 0:01:59.331099
    similarities = cosine_similarities(sparse_vectorized_corpus)
    similarities_df = pd.DataFrame.sparse.from_spmatrix(similarities, columns=corpus_text_ids)

    # print("similarities :")
    # print(similarities)

    similarities_df["id"] = corpus_text_ids
    similarities_df = similarities_df.set_index(["id"])
    return similarities_df


def get_all_similarities_one_at_a_time(sparse_vectorized_corpus, corpus_text_ids, text_id, keep_original=False):
    text_id_index = corpus_text_ids.index(text_id)
    # Fastest - vectorized_corpus.shape : (76484, 10) - duration 0:01:59.331099
    single_vectorized_record = sparse_vectorized_corpus[text_id_index, :]
    similarities = np.dot(single_vectorized_record, sparse_vectorized_corpus.T).toarray().ravel()
    similarities_series = pd.Series(similarities, index=corpus_text_ids)
    corpus_text_ids_adj = corpus_text_ids.copy()

    corpus_text_ids_adj.remove(text_id)

    similarities_series = similarities_series.filter(corpus_text_ids_adj)
    similarities_series.index.name = "id"
    similarities_series = similarities_series.sort_values(ascending=False)

    # print(">> In 'get_all_similarities_one_at_a_time' similarities_series :", similarities_series)

    if keep_original:
        similarities_series = pd.concat([pd.Series(99.0, index=[text_id]), similarities_series])

    return similarities_series


def fit_classifier(sparse_vectorized_corpus, corpus_text_ids, texts_list, texts_list_labeled,
                   y_classes=["earthquake", "fire", "flood", "hurricane"],
                   verbose=False,
                   classifier_list=[],
                   random_state=2584,
                   n_jobs=-1,
                   labels_got_overridden_flag=False,
                   full_fit_if_labels_got_overridden=False):
    texts_list_labeled_df = pd.DataFrame.from_dict(texts_list_labeled)

    if verbose:
        print("texts_list_labeled_df :")
        print(texts_list_labeled_df.head())

    ids = texts_list_labeled_df["id"].values
    y_train = texts_list_labeled_df["label"].values
    indices = [corpus_text_ids.index(x) for x in ids]
    X_train = sparse_vectorized_corpus[indices, :]

    if len(classifier_list) == 0:
        # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=2584))
        clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=n_jobs)
    elif len(classifier_list) == 1:
        clf = classifier_list[0]

    if labels_got_overridden_flag:
        if full_fit_if_labels_got_overridden:
            all_texts_list_labeled_df = pd.DataFrame.from_dict(texts_list)
            all_texts_list_labeled_df = all_texts_list_labeled_df[~all_texts_list_labeled_df["label"].isin(["-"])]

            y_classes_labeled = list(set(all_texts_list_labeled_df["label"].values))
            all_classes_present = all(label in y_classes_labeled for label in y_classes)
            clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3, random_state=random_state,
                                n_jobs=n_jobs)

            ids_all = all_texts_list_labeled_df["id"].values
            y_train_all = all_texts_list_labeled_df["label"].values
            indices_all = [corpus_text_ids.index(x) for x in ids_all]
            X_train_all = sparse_vectorized_corpus[indices_all, :]
            print("all_classes_present :", all_classes_present)
            if all_classes_present:
                print("Classifier fitted on all labels.")
                clf.fit(X_train_all, y_train_all)
            else:
                clf.partial_fit(X_train_all, y_train_all, classes=y_classes)
        else:
            clf.partial_fit(X_train, y_train, classes=y_classes)
    else:
        clf.partial_fit(X_train, y_train, classes=y_classes)

    classifier_list.clear()
    classifier_list.append(clf)
    return clf


def get_top_predictions(selected_class, fitted_classifier, sparse_vectorized_corpus, corpus_text_ids,
                        texts_list,
                        top=5,
                        cutoff_proba=0.95,
                        y_classes=["earthquake", "fire", "flood", "hurricane"],
                        verbose=False,
                        exclude_already_labeled=True,
                        similar_texts=[]):

    predictions = fitted_classifier.predict_proba(sparse_vectorized_corpus)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = y_classes
    predictions_df["id"] = corpus_text_ids

    keep_cols = ["id"]
    keep_cols.extend([selected_class])
    predictions_df = predictions_df[keep_cols]

    predictions_df = predictions_df[predictions_df[selected_class] > cutoff_proba]
    predictions_df = predictions_df.sort_values([selected_class], ascending=False)

    if exclude_already_labeled:
        texts_list_df = pd.DataFrame.from_dict(texts_list)
        predictions_df = predictions_df.merge(texts_list_df, left_on="id", right_on="id", how="left")
        predictions_df = predictions_df[predictions_df["label"].isin(["-"])]

    if verbose:
        print(">> get_top_predictions > predictions_df :")
        print(predictions_df.head(top))

    filter_list = predictions_df.head(top)["id"].values
    top_texts = filter_all_texts(texts_list, filter_list, exclude_already_labeled=False)
    similar_texts.clear()
    similar_texts.extend(top_texts)

    return top_texts


def score_predictions(predictions_df, use_entropy=True, num_labels=5):
    prediction_values = predictions_df.values

    if use_entropy:
        all_score_combined = []
        for prediction in prediction_values:
            qk = [1/num_labels] * num_labels
            all_score_combined.append(entropy(prediction, base=num_labels, qk=qk))
    else:
        all_scores_num_non_zero = []
        all_scores_std = []
        for prediction in prediction_values:
            score_std = np.std(prediction)
            all_scores_std.append(score_std)
            score_num_non_zero = len([x for x in prediction if x > 0.0])
            all_scores_num_non_zero.append(score_num_non_zero)

        all_score_combined = np.array(all_scores_std) / np.array(all_scores_num_non_zero)

    return all_score_combined


def get_all_predictions(fitted_classifier, sparse_vectorized_corpus, corpus_text_ids, texts_list,
                        top=5,
                        y_classes=["earthquake", "fire", "flood", "hurricane"],
                        verbose=False,
                        round_to=2,
                        format_as_percentage=False,
                        similar_texts=[],
                        predictions_report=[],
                        overall_quality_score=[]):

    predictions = fitted_classifier.predict_proba(sparse_vectorized_corpus)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = y_classes

    predictions_summary_full = predictions_df.describe()

    predictions_summary_alt = predictions_df.mean(axis=0)
    predictions_summary = predictions_df.replace(0.0, np.NaN).mean(axis=0)

    predictions_df["id"] = corpus_text_ids

    texts_list_df = pd.DataFrame(texts_list)
    predictions_df = predictions_df.merge(texts_list_df, left_on="id", right_on="id")

    keep_cols = ["id", "text"]
    keep_cols.extend(y_classes)
    predictions_df = predictions_df[keep_cols]

    pred_scores = score_predictions(predictions_df[y_classes], use_entropy=True)
    overall_quality = np.mean(pred_scores)
    predictions_df["pred_scores"] = pred_scores

    if round_to and not format_as_percentage:
        predictions_df[y_classes] = predictions_df[y_classes].round(round_to)
        predictions_summary = predictions_summary.round(round_to)
        overall_quality = overall_quality.round(round_to)

    if format_as_percentage:
        if verbose:
            print(">> get_all_predictions > predictions_df.head() :")
            print(predictions_df.head(top))
        predictions_df[y_classes] = predictions_df[y_classes]\
            .astype(float)\
            .applymap(lambda x: "{0:.0%}".format(x))

        predictions_summary = (predictions_summary.astype(float) * 100).round(1).astype(str) + "%" # .applymap(lambda x: "{0:.0%}".format(x))

        overall_quality = (overall_quality.astype(float) * 100).round(1).astype(str) + "%"

    predictions_df = predictions_df.sort_values(["pred_scores"], ascending=[True])

    if verbose:
        print(">> get_all_predictions > predictions_df.head() :")
        print(predictions_df.head(top))
        print(">> get_all_predictions > predictions_df.tail() :")
        print(predictions_df.tail(top))

    keep_cols = ["id", "text"]
    keep_cols.extend(y_classes)
    top_texts = predictions_df.head(top)[keep_cols]\
        .to_dict("records")

    similar_texts.clear()
    similar_texts.extend(top_texts)

    predictions_summary_df = pd.DataFrame({"label": predictions_summary.index,
                                           "avg_confidence": predictions_summary.values})
    predictions_summary_list = predictions_summary_df.to_dict("records")
    predictions_report.clear()
    predictions_report.extend(predictions_summary_list)

    overall_quality_score.clear()
    overall_quality_score.append(overall_quality)

    return top_texts, predictions_summary_list, overall_quality


def label_all(fitted_classifier, sparse_vectorized_corpus, corpus_text_ids, texts_list,
              label_only_unlabeled=True, sub_list_limit=50, update_in_place=True, texts_list_list=[]):

    texts_list_df = pd.DataFrame(texts_list)

    if not label_only_unlabeled:
        predictions = fitted_classifier.predict(sparse_vectorized_corpus)
        predictions_df = pd.DataFrame(predictions)
        predictions_df["id"] = corpus_text_ids
        labeled_text_ids = corpus_text_ids
    else:
        label_only_these_ids = texts_list_df[texts_list_df["label"] == "-"]["id"].values
        keep_indices = [corpus_text_ids.index(x) for x in label_only_these_ids]
        sparse_vectorized_corpus_alt = sparse_vectorized_corpus[keep_indices, :]
        predictions = fitted_classifier.predict(sparse_vectorized_corpus_alt)
        predictions_df = pd.DataFrame(predictions)
        predictions_df["id"] = label_only_these_ids
        labeled_text_ids = label_only_these_ids

    predictions_df = predictions_df.rename(columns={0: "label"})
    predictions_df = predictions_df.merge(texts_list_df[["id", "text"]], left_on="id", right_on="id", how="left")
    predictions_df = predictions_df[["id", "text", "label"]]
    update_objects = predictions_df.to_dict("records")

    updated_texts_list, updated_texts_list_list, overridden =  \
        update_texts_list_by_id(texts_list=texts_list,
                                sub_list_limit=sub_list_limit,
                                updated_obj_lst=update_objects,
                                texts_list_list=texts_list_list,
                                update_in_place=update_in_place)

    return updated_texts_list, updated_texts_list_list, labeled_text_ids


def get_similarities_single_record(similarities_df, corpus_text_id):
    keep_indices = [x for x in similarities_df.index.values if x not in [corpus_text_id]]
    similarities = similarities_df[corpus_text_id]
    similarities = similarities.filter(keep_indices)
    similarities = similarities.sort_values(ascending=False)

    return similarities


def get_top_similar_texts(all_texts_json, similarities_series, top=5, exclude_already_labeled=False, verbose=True,
                          similar_texts=[]):

    if exclude_already_labeled:
        all_texts_df = pd.DataFrame.from_dict(all_texts_json)
        similarities_df = pd.DataFrame(similarities_series).reset_index().rename(columns={0: "proba", "index": "id"})

        if verbose:
            print(">> get_top_similar_texts > similarities_df :")
            print(similarities_df.head())

            print(">> all_texts_df > all_texts_df :")
            print(all_texts_df.head())

        similarities_df = similarities_df.merge(all_texts_df, left_on="id", right_on="id")
        similarities_df = similarities_df[similarities_df["label"].isin(["-"])]

        # print(">> get_top_similar_texts > similarities_df :")
        # print(similarities_df.head())

        filter_list = similarities_df.head(top)["id"].values
    else:
        filter_list = similarities_series.head(top).index.values

    top_texts = filter_all_texts(all_texts_json, filter_list, exclude_already_labeled=False)

    if verbose:
        print(">> all_texts_df > len(top_texts) :", len(top_texts))

    similar_texts.clear()
    similar_texts.extend(top_texts)

    return top_texts


def generate_summary(text_lists, first_labeling_flag, total_summary=[], label_summary=[], number_unlabeled_texts=[]):
    labels = [text_obj["label"] for text_obj in text_lists]
    label_counter = Counter(labels)
    total_texts = len(text_lists)
    number_unlabeled = label_counter["-"]
    number_labeled = total_texts - number_unlabeled

    if number_labeled > 0:
        first_labeling_flag = False

    total_texts_percentage = "100.00%"
    number_unlabeled_percentage = "{:.2%}".format(number_unlabeled / total_texts)
    number_labeled_percentage = "{:.2%}".format(number_labeled / total_texts)

    total_summary.clear()
    total_summary.append({"name": "Total Texts",
                          "number": "{:,}".format(total_texts),
                          "percentage": total_texts_percentage})
    total_summary.append({"name": "Total Unlabeled",
                          "number": "{:,}".format(number_unlabeled),
                          "percentage": number_unlabeled_percentage})
    total_summary.append({"name": "Total Labeled",
                          "number": "{:,}".format(number_labeled),
                          "percentage": number_labeled_percentage})

    label_summary.clear()
    for key, value in label_counter.items():
        label_summary.append({"name": key,
                              "number": "{:,}".format(value),
                              "percentage": "{:.2%}".format(value / total_texts)})

    number_unlabeled_texts.clear()
    number_unlabeled_texts.append(number_unlabeled)

    return total_summary, label_summary, first_labeling_flag, number_unlabeled_texts


def generate_click_record(click_location, click_type, click_object, guid=None):
    time_stamp = datetime.now()

    if not guid:
        guid = uuid.uuid4()

    click_record = [{"click_id": str(guid),
                     "click_location": click_location,
                     "click_type": click_type,
                     "click_object": click_object,
                     "click_date_time": time_stamp}]
    return click_record, guid


def generate_value_record(guid, value_type, value):
    value_record = [{"click_id": str(guid),
                     "value_type": value_type,
                     "value": value}]
    return value_record


def add_log_record(record, log=[]):
    log.extend(record)
    return None


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_df = get_disaster_tweet_demo_data()
    # print("consolidated_disaster_tweet_data_df :")
    # print(consolidated_disaster_tweet_data_df.head())
    #
    # all_texts_adj = convert_demo_data_into_list(consolidated_disaster_tweet_data_df, limit=50)
    # print("all_texts_adj :")
    # print(all_texts_adj)
    # print()

    all_texts_json, adj_text_ids = convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=100000)
    # print("all_texts_json :")
    # print(all_texts_json[:5])
    # print()

    include_search_term = "hurrican earthquake"
    exclude_search_term = None

    search_results = search_all_texts(all_text=all_texts_json,
                                      include_search_term=include_search_term,
                                      exclude_search_term=exclude_search_term,
                                      exclude_already_labeled=False,
                                      behavior="conjunction", # disjunction conjunction
                                      all_upper=False)
    print("len(search_results) :", len(search_results))
    print("search_results :")
    print(search_results[:5])
    print()

    # print("type(all_texts_adj) :")
    # print(type(all_texts_adj))
    # print()
    # text_id = '798262465234542592'
    # all_texts_updated = update_all_texts(all_texts=all_texts_adj, text_id=text_id, label="Earthquake")
    # print("all_texts_updated :")
    # print(all_texts_updated)
    adj_consolidated_disaster_tweet_data_df = consolidated_disaster_tweet_data_df#.head(50)
    # print("adj_consolidated_disaster_tweet_data_df.head(5)['tweet_text'] :")
    # print(adj_consolidated_disaster_tweet_data_df.head(5)["tweet_text"])

    corpus_text_id = "798262465234542592"

    # print("798262465234542592 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == corpus_text_id]["tweet_text"])
    # print()
    # print("721752219201232897 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798014650231046144"]["tweet_text"])
    # print()
    # print("800600023310286848 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "797905553376899072"]["tweet_text"])
    # print()
    # print("1176515611297533952 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798277386928328704"]["tweet_text"])
    # print()
    # print("798100160148545536 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "722068424692789248"]["tweet_text"])
    # print()
    # print("910579763537788928 :")
    # print(adj_consolidated_disaster_tweet_data_df[adj_consolidated_disaster_tweet_data_df["tweet_id"] == "798027556041560064"]["tweet_text"])
    # print()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
    vectorized_corpus = \
        vectorizer.fit_transform(adj_consolidated_disaster_tweet_data_df["tweet_text"])

    print("vectorized_corpus.shape :", vectorized_corpus.shape)
    print("type(vectorized_corpus) :", type(vectorized_corpus))
    vectorized_corpus_df = pd.DataFrame.sparse.from_spmatrix(vectorized_corpus, columns=vectorizer.get_feature_names_out())
    print("vectorized_corpus_df.shape :", vectorized_corpus_df.shape)
    # print("vectorized_corpus_df.head ")
    # print(vectorized_corpus_df.head)

    temp_start_time = datetime.now()
    corpus_text_ids = [str(x) for x in adj_consolidated_disaster_tweet_data_df["tweet_id"].values]
    print(f">> 'corpus_text_ids' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
          f"duration {datetime.now()-temp_start_time}")
    # print("corpus_text_ids :", corpus_text_ids)

    # ********************************************************************************************************
    temp_start_time = datetime.now()
    similarities_alt = get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus,
                                                          corpus_text_ids=corpus_text_ids,
                                                          text_id=corpus_text_id)
    print(f">> 'similarities_alt' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
          f"duration {datetime.now()-temp_start_time}")
    print("type(similarities_alt) :", type(similarities_alt))

    print("similarities_alt :")
    print(similarities_alt.head(5))

    top_5_texts = get_top_similar_texts(all_texts_json=all_texts_json, similarities_series=similarities_alt, top=5,
                                        exclude_already_labeled=True, similar_texts=[])
    print("top_5_texts :")
    print(top_5_texts)
    # ********************************************************************************************************

    # ********************************************************************************************************
    # temp_start_time = datetime.now()
    # similarities_df = get_all_similarities(sparse_vectorized_corpus=vectorized_corpus,
    #                                        corpus_text_ids=corpus_text_ids)
    # print(f">> 'similarities_df' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # print("type(similarities_df) :", type(similarities_df))
    #
    # # print("similarities_df :")
    # # print(similarities_df)
    #
    # temp_start_time = datetime.now()
    # similarities = get_similarities_single_record(similarities_df=similarities_df, corpus_text_id=corpus_text_id)
    # print(f">> 'similarities' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # print("type(similarities) :", type(similarities))
    #
    # print("similarities :")
    # print(similarities.head(5))
    #
    # filter_list = list(similarities.index)
    # # print("filter_list :", filter_list)
    #
    # temp_start_time = datetime.now()
    # filtered_all_text = filter_all_texts(all_text=all_texts_json, filter_list=filter_list)
    # print(f">> 'filtered_all_text' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # # print("filtered_all_text :")
    # # print(filtered_all_text)
    # ********************************************************************************************************

    # ********************************************************************************************************
    temp_start_time = datetime.now()
    ml_test_tweet_data_df = get_disaster_tweet_demo_data(number_samples=100,
                                                         filter_data_types=["dev"],
                                                         random_state=2584)
    # print("ml_test_tweet_data_df :")
    # print(ml_test_tweet_data_df.head())

    print('ml_test_tweet_data_df["event_type"].value_counts() :')
    print(ml_test_tweet_data_df["event_type"].value_counts())

    texts_list_labeled, adj_text_ids_labeled = \
        convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=50, keep_labels=True)
    # print("texts_list_labeled :")
    # print(texts_list_labeled[:5])

    y_classes = ["earthquake", "fire", "flood", "hurricane"]

    fitted_classifier = fit_classifier(sparse_vectorized_corpus=vectorized_corpus,
                                       corpus_text_ids=corpus_text_ids,
                                       texts_list_labeled=texts_list_labeled,
                                       y_classes=y_classes,
                                       classifier_list=[])

    predictions = get_top_predictions(selected_class="fire",
                                      fitted_classifier=fitted_classifier,
                                      sparse_vectorized_corpus=vectorized_corpus,
                                      corpus_text_ids=corpus_text_ids,
                                      texts_list=all_texts_json,
                                      top=5,
                                      cutoff_proba=0.8,

                                      # tweet_data_df=consolidated_disaster_tweet_data_df,
                                      # fat_df=True,

                                      y_classes=y_classes,
                                      verbose=True,
                                      exclude_already_labeled=True,
                                      similar_texts=[])
    print("predictions :")
    print(predictions[:5])
    print()

    predictions_all = get_all_predictions(fitted_classifier=fitted_classifier,
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

    # print(f">> 'similarities_alt' calculated @{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} "
    #       f"duration {datetime.now()-temp_start_time}")
    # ********************************************************************************************************
    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)



