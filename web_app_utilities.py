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
import pickle
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import sqlite3
from collections import Counter
import numpy as np
from sklearn.linear_model import SGDClassifier
import pathlib

pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 200
pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)


RND_SEED = 45822
random.seed(RND_SEED)
np.random.seed(RND_SEED)


connection = sqlite3.connect('database.db', timeout=30)
with open("schema.sql") as f:
    connection.executescript(f.read())


def get_db_connection():
    conn = sqlite3.connect('database.db', timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def populate_texts_table_sql(texts_list, table_name="texts"):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM " + table_name + ";")
    conn.commit()
    for text_record in texts_list:
        cur.execute("INSERT INTO " + table_name + " (id, text, label) VALUES (?, ?, ?)",
                    (text_record["id"], text_record["text"], "-"))
    conn.commit()
    conn.close()
    return None


def get_decimal_value(name):
    conn = get_db_connection()
    query = "SELECT name, value FROM decimalValues WHERE name = '%s' ;" % name
    sql_table = conn.execute(query).fetchall()
    decimal_value = [dict(row)["value"] for row in sql_table][0]
    conn.close()
    return decimal_value


def set_decimal_value(name, value):
    conn = get_db_connection()
    query = "UPDATE decimalValues SET value = %s WHERE name = '%s' ;" % (value, name)
    conn.execute(query)
    conn.commit()
    conn.close()
    return None


def update_overall_quality_scores(value):
    current_score = get_decimal_value(name="OVERALL_QUALITY_SCORE_DECIMAL")
    set_decimal_value(name="OVERALL_QUALITY_SCORE_DECIMAL_PREVIOUS", value=current_score)
    set_decimal_value(name="OVERALL_QUALITY_SCORE_DECIMAL", value=value)
    return None


def set_pkl(name, pkl_data, reset=False):
    if not reset:
        pathlib.Path("./output/pkl/").mkdir(parents=True, exist_ok=True)
        with open("./output/pkl/" + name + ".pkl", "wb") as f:
            pickle.dump(pkl_data, f)
    else:
        if os.path.exists("./output/pkl/" + name + ".pkl"):
            os.remove("./output/pkl/" + name + ".pkl")
    return None


def get_pkl(name):
    try:
        pkl_data = pickle.load(open(os.path.join("./output/pkl", name + ".pkl"), "rb"))
        return pkl_data
    except:
        return None

    # conn = get_db_connection()
    # query = "SELECT * FROM pkls WHERE name = " + name + ";"
    # pkl_table = conn.execute(query).fetchall()
    # pkl_data = [dict(row) ["data"] for row in pkl_table]
    # conn.close()
    # return pkl_data
    #
    # query = u'''insert into testtable VALUES(?)'''
    # b = sqlite3.Binary(binarydata)
    # cur.execute(query, (b,))
    # con.commit()
    #
    # return None


def get_text_list(table_name="texts"):
    conn = get_db_connection()
    text_list_sql = conn.execute("SELECT * FROM " + table_name).fetchall()
    text_list_sql = [dict(row) for row in text_list_sql]
    conn.close()
    return text_list_sql


def set_text_list(label, table_name="searchResults"):
    conn = get_db_connection()
    conn.execute("UPDATE " + table_name + " SET label = '" + label + "'")
    conn.commit()
    conn.close()
    return None


def clear_text_list(table_name="searchResults"):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM " + table_name + ";")
    conn.commit()
    conn.close()
    return None


def get_appropriate_text_list_list(text_list_full_sql, total_pages_sql, search_results_length_sql, table_limit_sql):
    if search_results_length_sql > 0:
        text_list_full_sql = get_text_list(table_name="searchResults")
        total_pages_sql = get_variable_value(name="SEARCH_TOTAL_PAGES")
    text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    return text_list_list_sql, text_list_full_sql, total_pages_sql


def get_y_classes():
    conn = get_db_connection()
    y_classes_sql = conn.execute('SELECT className FROM yClasses;').fetchall()
    y_classes_sql = [dict(row)["className"] for row in y_classes_sql]
    conn.close()
    return y_classes_sql


def clear_y_classes():
    conn = get_db_connection()
    conn.execute('DELETE FROM yClasses;')
    conn.commit()
    conn.close()
    return []


def add_y_classes(y_classses_list, begin_fresh=True):
    conn = get_db_connection()
    cur = conn.cursor()

    if begin_fresh:
        cur.execute("DELETE FROM yClasses;")

    for i, value in enumerate(y_classses_list):
        cur.execute("INSERT INTO yClasses (classId, className) VALUES (?, ?)", (i, value))

    conn.commit()
    conn.close()

    return 1


def get_click_log():
    conn = get_db_connection()
    sql_table = \
        conn.execute('SELECT click_id, click_location, click_type, click_object, click_date_time FROM clickRecord;').fetchall()

    click_log_sql = list()
    for row in sql_table:
        dict_row = {"click_id": dict(row)["click_id"], "click_location": dict(row)["click_location"],
                    "click_type": dict(row)["click_type"], "click_object": dict(row)["click_object"],
                    "click_date_time" : dict(row)["click_date_time"]}
        click_log_sql.append(dict_row)
    conn.close()
    return click_log_sql


def get_value_log():
    conn = get_db_connection()
    sql_table = \
        conn.execute('SELECT click_id, value_type, value FROM valueRecord;').fetchall()

    click_log_sql = list()
    for row in sql_table:
        dict_row = {"click_id": dict(row)["click_id"], "value_type": dict(row)["value_type"],
                    "value": dict(row)["value"]}
        click_log_sql.append(dict_row)
    conn.close()
    return click_log_sql


def reset_log_click_record_sql():
    conn = get_db_connection()
    conn.execute("DELETE FROM clickRecord")
    conn.commit()
    conn.close()
    return None


def reset_log_click_value_sql():
    conn = get_db_connection()
    conn.execute("DELETE FROM valueRecord")
    conn.commit()
    conn.close()
    return None


def add_log_click_record_sql(records):
    conn = get_db_connection()
    cur = conn.cursor()

    for record in records:
        cur.execute("""INSERT INTO clickRecord (click_id, click_location, click_type, click_object, click_date_time) 
                        VALUES (?, ?, ?, ?, ?)""", (record["click_id"], record["click_location"], record["click_type"],
                                                    record["click_object"], record["click_date_time"]))

    conn.commit()
    conn.close()

    return None


def add_log_click_value_sql(records):
    conn = get_db_connection()
    cur = conn.cursor()

    for record in records:
        cur.execute("""INSERT INTO valueRecord (click_id, value_type, value) 
                        VALUES (?, ?, ?)""", (record["click_id"], record["value_type"], record["value"]))

    conn.commit()
    conn.close()

    return None


def get_panel_flags():
    conn = get_db_connection()
    sql_table = conn.execute('SELECT name, value FROM initializeFlags;').fetchall()
    panel_flags = {dict(row)["name"]: dict(row)["value"] for row in sql_table}
    conn.close()
    return panel_flags


def update_panel_flags_sql(update_flag):
    conn = get_db_connection()
    cur = conn.cursor()
    update_query = "UPDATE initializeFlags SET value = ? WHERE name = ?;"

    for name, value in update_flag.items():
        cur.execute(update_query, (value, name))
    conn.commit()
    conn.close()
    return None


def get_texts_group_x(table_name="group1Texts"):
    conn = get_db_connection()
    sql_table = conn.execute("SELECT id, text, label FROM " + table_name + ";").fetchall()
    conn.close()
    if len(sql_table) > 0:
        texts_group_2 = [{"id": dict(row)["id"], "text": dict(row)["text"], "label": dict(row)["label"]} for row in sql_table]
    else:
        texts_group_2 = []
    return texts_group_2


def set_texts_group_x(top_texts, table_name="group1Texts"):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM " + table_name + ";")
    if top_texts:
        for record in top_texts:
            cur.execute("INSERT INTO " + table_name + " (id, text, label) VALUES (?, ?, ?)",
                        (record["id"], record["text"], record["label"]))
    conn.commit()
    conn.close()
    return None


def get_total_summary_sql():
    conn = get_db_connection()
    sql_table = conn.execute('SELECT name, number, percentage FROM totalSummary;').fetchall()
    total_summary = [{"name": dict(row)["name"],
                      "number": dict(row)["number"],
                      "percentage": dict(row)["percentage"]} for row in sql_table]
    conn.close()
    return total_summary


def set_total_summary(text_lists):
    labels = [text_obj["label"] for text_obj in text_lists]
    label_counter = Counter(labels)
    total_texts = len(text_lists)
    number_unlabeled = label_counter["-"]
    number_labeled = total_texts - number_unlabeled
    total_texts_percentage = "100.00%"

    number_unlabeled_percentage = "{:.2%}".format(number_unlabeled / total_texts)
    number_labeled_percentage = "{:.2%}".format(number_labeled / total_texts)

    total_summary = list()
    total_summary.append({"name": "Total Texts",
                          "number": "{:,}".format(total_texts),
                          "percentage": total_texts_percentage})
    total_summary.append({"name": "Total Unlabeled",
                          "number": "{:,}".format(number_unlabeled),
                          "percentage": number_unlabeled_percentage})
    total_summary.append({"name": "Total Labeled",
                          "number": "{:,}".format(number_labeled),
                          "percentage": number_labeled_percentage})

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM totalSummary;")
    for record in total_summary:
        cur.execute("INSERT INTO totalSummary (name, number, percentage) VALUES (?, ?, ?)",
                    (record["name"], record["number"], record["percentage"]))

    conn.commit()
    conn.close()
    return None


def get_label_summary_sql():
    conn = get_db_connection()
    sql_table = conn.execute('SELECT name, number, percentage FROM labelSummary;').fetchall()
    label_summary = [{"name": dict(row)["name"],
                      "number": dict(row)["number"],
                      "percentage": dict(row)["percentage"]} for row in sql_table]
    conn.close()
    return label_summary


def set_label_summary(text_lists):
    labels = [text_obj["label"] for text_obj in text_lists]
    label_counter = Counter(labels)
    total_texts = len(text_lists)

    label_summary = []
    for key, value in label_counter.items():
        label_summary.append({"name": key,
                              "number": "{:,}".format(value),
                              "percentage": "{:.2%}".format(value / total_texts)})

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM labelSummary;")
    for record in label_summary:
        cur.execute("INSERT INTO labelSummary (name, number, percentage) VALUES (?, ?, ?)",
                    (record["name"], record["number"], record["percentage"]))
    conn.commit()
    conn.close()
    return None


def get_selected_text(selected_text_id, text_list_full_sql):
    selected_text_test = [text["text"] for text in text_list_full_sql if text["id"] == selected_text_id]
    if selected_text_id:
        if len(selected_text_test) == 0:
            selected_text = ""
        else:
            selected_text = selected_text_test[0]
    else:
        selected_text = ""
    return selected_text


def create_text_list_list(text_list_full_sql, sub_list_limit):
    texts_list_list = \
        [text_list_full_sql[i:i + sub_list_limit] for i in range(0, len(text_list_full_sql), sub_list_limit)]
    return texts_list_list


def update_texts_list_by_id_sql(update_objs=None, selected_label=None, update_ids=None, sub_list_limit=10,
                                labels_got_overridden_flag=[],
                                update_in_place=True):
    text_list_full = get_text_list(table_name="texts")

    conn = get_db_connection()
    cur = conn.cursor()

    if selected_label and update_ids and not update_objs:
        if update_in_place:
            update_query = "UPDATE texts SET label = ? WHERE id IN (%s)" % ",".join("?"*len(update_ids))
            update_values = [selected_label]
            update_values.extend(update_ids)
            cur.execute(update_query, update_values)
            conn.commit()
            conn.close()

        else:
            cur.execute("DROP TABLE IF EXISTS temp_table;")
            cur.execute("""
                        CREATE TABLE temp_table (
                            id TEXT NOT NULL,
                            text TEXT NOT NULL,
                            label TEXT NOT NULL
                        );
                        """)
            query = "INSERT INTO temp_table SELECT * FROM texts WHERE id IN (%s)" % ",".join("?" * len(update_ids))
            cur.execute(query, update_ids)
            cur.execute("UPDATE temp_table SET label = ?", (selected_label, ))
            cur.execute("DELETE FROM texts WHERE id IN (%s)" % ",".join("?" * len(update_ids)), update_ids)
            cur.execute("INSERT INTO texts SELECT * FROM temp_table;")
            conn.commit()
            conn.close()

    elif update_objs and not selected_label and not update_ids:
        if update_in_place:
            labels = set([obj["label"] for obj in update_objs])
            for label in labels:
                update_ids = [obj["id"] for obj in update_objs if obj["label"] == label]
                update_ids_sql = ", ".join(update_ids)
                update_query = "UPDATE texts SET label = ? WHERE id IN (%s)" % update_ids_sql

                conn.execute(update_query, (label, ))
            conn.commit()
            conn.close()
        else:
            cur.execute("DROP TABLE IF EXISTS temp_table;")
            cur.execute("""
                           CREATE TABLE temp_table (
                               id TEXT NOT NULL,
                               text TEXT NOT NULL,
                               label TEXT NOT NULL
                           );
                           """)
            all_update_ids = [obj["id"] for obj in update_objs]
            query = "INSERT INTO temp_table SELECT * FROM texts WHERE id IN (%s)" % ",".join("?" * len(all_update_ids))
            cur.execute(query, all_update_ids)
            labels = set([obj["label"] for obj in update_objs])
            for label in labels:
                update_ids = [obj["id"] for obj in update_objs if obj["label"] == label]
                update_ids_sql = ", ".join(update_ids)
                update_query = "UPDATE temp_table SET label = ? WHERE id IN (%s)" % update_ids_sql
                conn.execute(update_query, (label,))
            delete_query = "DELETE FROM texts WHERE id IN (%s)" % ",".join("?" * len(all_update_ids))
            cur.execute(delete_query, all_update_ids)
            cur.execute("INSERT INTO texts SELECT * FROM temp_table;")

            conn.commit()
            conn.close()

    text_list_full = get_text_list(table_name="texts")
    texts_list_list = create_text_list_list(text_list_full_sql=text_list_full, sub_list_limit=sub_list_limit)

    return text_list_full, texts_list_list


def label_all_sql(fitted_classifier, sparse_vectorized_corpus, corpus_text_ids, texts_list,
                  label_only_unlabeled=True, sub_list_limit=50, update_in_place=True):

    texts_list_df = pd.DataFrame(texts_list)

    if not label_only_unlabeled:
        predictions = fitted_classifier.predict(sparse_vectorized_corpus)
        predictions_df = pd.DataFrame(predictions)
        predictions_df["id"] = corpus_text_ids
        labeled_text_ids = corpus_text_ids
        number_to_label = len(labeled_text_ids)
    else:
        label_only_these_ids = texts_list_df[texts_list_df["label"] == "-"]["id"].values
        keep_indices = [corpus_text_ids.index(x) for x in label_only_these_ids]
        number_to_label = len(keep_indices)

    if number_to_label > 0:
        if label_only_unlabeled:
            sparse_vectorized_corpus_alt = sparse_vectorized_corpus[keep_indices, :]
            predictions = fitted_classifier.predict(sparse_vectorized_corpus_alt)
            predictions_df = pd.DataFrame(predictions)
            predictions_df["id"] = label_only_these_ids
            labeled_text_ids = label_only_these_ids

        predictions_df = predictions_df.rename(columns={0: "label"})
        predictions_df = predictions_df.merge(texts_list_df[["id", "text"]], left_on="id", right_on="id",
                                              how="left")
        predictions_df = predictions_df[["id", "text", "label"]]
        update_objects = predictions_df.to_dict("records")

        text_list_full, texts_list_list = \
            update_texts_list_by_id_sql(update_objs=update_objects,
                                        selected_label=None,
                                        update_ids=None,
                                        sub_list_limit=sub_list_limit,
                                        labels_got_overridden_flag=[],
                                        update_in_place=update_in_place)
    else:
        text_list_full = get_text_list(table_name="texts")
        texts_list_list = create_text_list_list(text_list_full_sql=text_list_full, sub_list_limit=sub_list_limit)
        labeled_text_ids = []

    return text_list_full, texts_list_list, labeled_text_ids


def generate_summary_sql(text_lists):
    labels = [text_obj["label"] for text_obj in text_lists]
    label_counter = Counter(labels)
    total_texts = len(text_lists)
    number_unlabeled = label_counter["-"]
    number_labeled = total_texts - number_unlabeled

    set_total_summary(text_lists=text_lists)
    set_label_summary(text_lists=text_lists)
    set_variable(name="NUMBER_UNLABELED_TEXTS", value=number_unlabeled)

    summary_headline = \
        "Total Labeled : {:,} / {:,} {:.1%}".format(number_labeled, total_texts, number_labeled / total_texts)
    set_variable(name="LABEL_SUMMARY_STRING", value=summary_headline)

    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    number_unlabeled_texts_sql = number_unlabeled
    label_summary_string_sql = summary_headline
    return total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql


def set_variable(name, value):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """INSERT OR REPLACE INTO variables (name, value) VALUES (?, ?) 
                """
    cur.execute(query, (name, value))
    conn.commit()
    conn.close()
    return 1


def get_variable_value(name):
    conn = get_db_connection()
    cur = conn.cursor()
    query = cur.execute('SELECT value FROM variables WHERE name = ?', (name,)).fetchall()
    value = [dict(row)["value"] for row in query]
    value = value[0]

    if name in ["TOTAL_PAGES", "NUMBER_UNLABELED_TEXTS", "MAX_CONTENT_PATH", "TEXTS_LIMIT", "TABLE_LIMIT",
                "MAX_FEATURES", "RND_STATE", "PREDICTIONS_NUMBER", "SEARCH_RESULTS_LENGTH", "GROUP_1_KEEP_TOP",
                "GROUP_3_KEEP_TOP", "CONFIRM_LABEL_ALL_TEXTS_COUNTS", "SEARCH_TOTAL_PAGES", "LABEL_ALL_BATCH_NO",
                "LABEL_ALL_TOTAL_BATCHES", "NUMBER_AUTO_LABELED", "LABEL_ALL_BATCH_SIZE"]:
        value = int(value)

    if name in ["KEEP_ORIGINAL", "GROUP_1_EXCLUDE_ALREADY_LABELED", "GROUP_2_EXCLUDE_ALREADY_LABELED",
                "PREDICTIONS_VERBOSE", "SIMILAR_TEXT_VERBOSE", "FIT_CLASSIFIER_VERBOSE", "FIRST_LABELING_FLAG",
                "FULL_FIT_IF_LABELS_GOT_OVERRIDDEN", "FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS",
                "LABELS_GOT_OVERRIDDEN_FLAG", "UPDATE_TEXTS_IN_PLACE"]:
        if value == "True":
            value = True
        else:
            value = False

    if name in ["PREDICTIONS_PROBABILITY"]:
        value = float(value)

    conn.commit()
    conn.close()
    return value


def get_difficult_texts_sql():
    try:
        conn = get_db_connection()
        select_diff_texts_query = get_variable_value(name="SELECT_DIFF_TEXTS_QUERY")
        sql_cols_list_y_classes = get_pkl(name="SQL_COLS_LIST_Y_CLASSES")
        sql_table = conn.execute(select_diff_texts_query).fetchall()

        total_summary = list()
        for row in sql_table:
            temp_row = {col: dict(row)[col] for col in sql_cols_list_y_classes}
            total_summary.append(temp_row)
        # total_summary = [{"name": dict(row)["name"],
        #                   "number": dict(row)["number"],
        #                   "percentage": dict(row)["percentage"]} for row in sql_table]
        conn.close()
        return total_summary
    except:
        return []


def reset_difficult_texts_sql():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM difficultTexts')
        conn.commit()
        conn.close()
        return None
    except:
        return None


def get_available_datasets():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM availableDatasets;")
    cur.execute("INSERT INTO availableDatasets SELECT * FROM fixedDatasets;")
    conn.commit()
    conn.close()
    dataset_name, dataset_url, date_time, y_classes, total_summary = has_save_data(source_dir="./output/save")

    if dataset_name:
        date_at_end_check = re.findall(r"(.*)\-[0-9]{4}\-[0-9]{2}\-[0-9]{2}\-\-[0-9]{2}\-[0-9]{2}\-[0-9]{2}", dataset_name)
        if len(date_at_end_check) > 0:
            dataset_name_alt = date_at_end_check[0]
        else:
            dataset_name_alt = dataset_name

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO availableDatasets (name, description, url) VALUES (?, ?, ?)",
                    (dataset_name_alt + "-" + date_time,
                     "A partially labeled dataset having " + total_summary[2]["percentage"] +
                     " of " + total_summary[0]["number"] + " texts labeled.",
                     dataset_url))
        conn.commit()
        conn.close()

    conn = get_db_connection()
    available_datasets_sql = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()
    return available_datasets_sql


def has_save_data(source_dir="./output"):
    try:
        dataset_name = pickle.load(open(os.path.join(source_dir, "DATASET_NAME.pkl"), "rb"))
        dataset_url = pickle.load(open(os.path.join(source_dir, "DATASET_URL.pkl"), "rb"))
        date_time = pickle.load(open(os.path.join(source_dir, "DATE_TIME.pkl"), "rb"))
        y_classes = pickle.load(open(os.path.join(source_dir, "Y_CLASSES.pkl"), "rb"))
        total_summary = pickle.load(open(os.path.join(source_dir, "TOTAL_SUMMARY.pkl"), "rb"))
        return dataset_name, dataset_url, date_time, y_classes, total_summary
    except:
        return None, None, None, None, None


def get_all_predictions_sql(fitted_classifier, sparse_vectorized_corpus, corpus_text_ids, texts_list,
                            top=5,
                            y_classes=["earthquake", "fire", "flood", "hurricane"],
                            verbose=False,
                            round_to=2,
                            format_as_percentage=False):

    predictions = fitted_classifier.predict_proba(sparse_vectorized_corpus)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = y_classes

    predictions_summary = predictions_df.replace(0.0, np.NaN).mean(axis=0)

    predictions_df["id"] = corpus_text_ids

    texts_list_df = pd.DataFrame(texts_list)
    predictions_df = predictions_df.merge(texts_list_df, left_on="id", right_on="id")

    keep_cols = ["id", "text"]
    keep_cols.extend(y_classes)
    predictions_df = predictions_df[keep_cols]

    pred_scores = score_predictions(predictions_df[y_classes], use_entropy=True, num_labels=len(y_classes))
    overall_quality = np.mean(pred_scores)
    overall_quality_score_decimal_sql = overall_quality
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
    sql_cols_list = [x + " TEXT NOT NULL" for x in keep_cols]
    sql_cols = ", ".join(sql_cols_list)
    top_texts = predictions_df.head(top)[keep_cols].to_dict("records")

    sql_query_1 = """
                    DROP TABLE IF EXISTS difficultTexts;
                """
    sql_query_2 = """
                    CREATE TABLE difficultTexts (
                """ + sql_cols + """        
                    );
                """

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(sql_query_1)
    conn.commit()
    cur.execute(sql_query_2)
    conn.commit()

    parameters = ", ".join(["?"] * len(keep_cols))
    query = "INSERT INTO difficultTexts (" + ", ".join(keep_cols) + ") VALUES (%s)" % parameters

    for record in top_texts:
        insert_values = [value for key, value in record.items()]
        cur.execute(query, (insert_values))
    conn.commit()
    conn.close()

    conn = get_db_connection()
    select_diff_texts_query = "SELECT " + ", ".join(keep_cols) + " FROM difficultTexts;"
    set_variable(name="SELECT_DIFF_TEXTS_QUERY", value=select_diff_texts_query)
    set_pkl(name="SQL_COLS_LIST_Y_CLASSES", pkl_data=keep_cols, reset=None)
    sql_table = conn.execute(select_diff_texts_query).fetchall()
    texts_group_3_sql = []
    for row in sql_table:
        texts_group_3_sql.append({key: value for key, value in dict(row).items()})
    conn.close()

    update_overall_quality_scores(value=overall_quality_score_decimal_sql)
    set_variable(name="OVERALL_QUALITY_SCORE", value=overall_quality)
    overall_quality_score_sql = overall_quality
    overall_quality_score_decimal_previous_sql = get_decimal_value(name="OVERALL_QUALITY_SCORE_DECIMAL_PREVIOUS")
    return texts_group_3_sql, overall_quality_score_sql, \
           overall_quality_score_decimal_sql, overall_quality_score_decimal_previous_sql


def get_top_predictions_sql(selected_class, fitted_classifier, sparse_vectorized_corpus, corpus_text_ids,
                            texts_list,
                            top=5,
                            cutoff_proba=0.95,
                            y_classes=["earthquake", "fire", "flood", "hurricane"],
                            verbose=False,
                            exclude_already_labeled=True):

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

    set_texts_group_x(top_texts=top_texts, table_name="group2Texts")
    texts_group_3_sql = get_texts_group_x(table_name="group2Texts")

    return texts_group_3_sql


def fit_classifier_sql(sparse_vectorized_corpus, corpus_text_ids, texts_list, texts_list_labeled,
                       y_classes=["earthquake", "fire", "flood", "hurricane"],
                       verbose=False,
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

    classifier_sql = get_pkl(name="CLASSIFIER")
    if classifier_sql:
        clf = classifier_sql
    else:
        # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=2584))
        clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=n_jobs)


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

            if all_classes_present:
                clf.fit(X_train_all, y_train_all)
            else:
                clf.partial_fit(X_train_all, y_train_all, classes=y_classes)
        else:
            clf.partial_fit(X_train, y_train, classes=y_classes)
    else:
        clf.partial_fit(X_train, y_train, classes=y_classes)

    set_pkl(name="CLASSIFIER", pkl_data=clf, reset=False)
    return clf


def load_new_data_sql(source_file,
                      text_id_col,
                      text_value_col,
                      source_folder="./output/upload/",
                      shuffle_by="kmeans",
                      table_limit=50, texts_limit=1000, max_features=100,
                      y_classes=["Label 1", "Label 2", "Label 3", "Label 4"], rnd_state=258):

    data_df = get_new_data(source_file=source_file,
                           source_folder=source_folder,
                           number_samples=None,
                           random_state=rnd_state)
    print("data_df :")
    print(data_df.head())
    corpus_text_ids = [str(x) for x in data_df[text_id_col].values]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)
    vectorized_corpus = vectorizer.fit_transform(data_df[text_value_col].values)

    if shuffle_by == "kmeans":
        kmeans = KMeans(n_clusters=len(y_classes), random_state=rnd_state).fit(vectorized_corpus)
        kmeans_labels = kmeans.labels_
        texts_list, adj_text_ids = convert_new_data_into_list_json(data_df,
                                                                   limit=texts_limit,
                                                                   shuffle_list=kmeans_labels,
                                                                   random_shuffle=False,
                                                                   random_state=rnd_state,
                                                                   id_col=text_id_col,
                                                                   text_col=text_value_col,
                                                                   label_col="label")
    else:
        texts_list, adj_text_ids = convert_new_data_into_list_json(data_df,
                                                                   limit=texts_limit,
                                                                   shuffle_list=[],
                                                                   random_shuffle=True,
                                                                   random_state=rnd_state,
                                                                   id_col=text_id_col,
                                                                   text_col=text_value_col,
                                                                   label_col="label")
    populate_texts_table_sql(texts_list=texts_list, table_name="texts")

    set_pkl(name="CORPUS_TEXT_IDS", pkl_data=None, reset=True)
    set_pkl(name="CORPUS_TEXT_IDS", pkl_data=adj_text_ids, reset=False)

    texts_list_list = [texts_list[i:i + table_limit] for i in range(0, len(texts_list), table_limit)]
    total_pages = len(texts_list_list)
    set_variable(name="TOTAL_PAGES", value=total_pages)

    set_pkl(name="TEXTS_LIST_LIST", pkl_data=None, reset=True)
    set_pkl(name="TEXTS_LIST_LIST", pkl_data=texts_list_list, reset=False)

    set_pkl(name="VECTORIZED_CORPUS", pkl_data=None, reset=True)
    set_pkl(name="VECTORIZED_CORPUS", pkl_data=vectorized_corpus, reset=False)

    set_pkl(name="VECTORIZER", pkl_data=None, reset=True)
    set_pkl(name="VECTORIZER", pkl_data=vectorizer, reset=False)

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids


def load_demo_data_sql(dataset_name="Disaster Tweets Dataset", shuffle_by="kmeans",
                       table_limit=50, texts_limit=1000, max_features=100,
                       y_classes=["Earthquake", "Fire", "Flood", "Hurricane"], rnd_state=258):
    if dataset_name == "Disaster Tweets Dataset":
        consolidated_disaster_tweet_data_df = get_disaster_tweet_demo_data(number_samples=None,
                                                                                 filter_data_types=["train"],
                                                                                 random_state=rnd_state)
        corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]
        set_pkl(name="CORPUS_TEXT_IDS", pkl_data=corpus_text_ids, reset=False)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)

        # https://stackoverflow.com/questions/69326639/sklearn-warnings-in-version-1-0
        vectorized_corpus = \
            vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"].values)

        set_pkl(name="VECTORIZER", pkl_data=vectorizer, reset=False)
        set_pkl(name="VECTORIZED_CORPUS", pkl_data=vectorized_corpus, reset=False)

        if shuffle_by == "kmeans":
            kmeans = KMeans(n_clusters=len(y_classes), random_state=rnd_state).fit(vectorized_corpus)
            kmeans_labels = kmeans.labels_
            texts_list, adj_text_ids = convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                              limit=texts_limit,
                                                                              keep_labels=False,
                                                                              shuffle_list=kmeans_labels,
                                                                              random_shuffle=False,
                                                                              random_state=rnd_state)
        else:
            texts_list, adj_text_ids = convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                              limit=texts_limit,
                                                                              keep_labels=False,
                                                                              shuffle_list=[],
                                                                              random_shuffle=True,
                                                                              random_state=rnd_state)

    texts_list_list = [texts_list[i:i + table_limit] for i in range(0, len(texts_list), table_limit)]

    total_pages = len(texts_list_list)
    set_variable(name="TOTAL_PAGES", value=total_pages)

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids


def generate_all_predictions_if_appropriate(n_jobs=-1,
                                            labels_got_overridden_flag=True,
                                            full_fit_if_labels_got_overridden=True,
                                            round_to=1,
                                            format_as_percentage=True):

    classifier_sql = get_pkl(name="CLASSIFIER")

    # try:
    if classifier_sql:
        label_summary_sql = get_label_summary_sql()
        label_summary_df = pd.DataFrame.from_dict(label_summary_sql)
        y_classes_labeled = label_summary_df["name"].values
        y_classes_sql = get_y_classes()
        all_classes_present = all(label in y_classes_labeled for label in y_classes_sql)
        if all_classes_present:
            force_full_fit_for_difficult_texts_sql = get_variable_value(name="FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS")
            random_state_sql = get_variable_value(name="RND_STATE")
            corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
            vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
            predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
            fit_classifier_verbose_sql = get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
            text_list_full_sql = get_text_list(table_name="texts")

            if force_full_fit_for_difficult_texts_sql:
                fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
                                   corpus_text_ids=corpus_text_ids_sql,
                                   texts_list=text_list_full_sql,
                                   texts_list_labeled=text_list_full_sql,
                                   y_classes=y_classes_sql,
                                   verbose=fit_classifier_verbose_sql,
                                   random_state=random_state_sql,
                                   n_jobs=n_jobs,
                                   labels_got_overridden_flag=labels_got_overridden_flag,
                                   full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden)

            top_sql = get_variable_value(name="GROUP_3_KEEP_TOP")
            texts_group_3_sql, overall_quality_score_sql, \
                overall_quality_score_decimal_sql, overall_quality_score_decimal_previous_sql = \
                    get_all_predictions_sql(fitted_classifier=classifier_sql,
                                            sparse_vectorized_corpus=vectorized_corpus_sql,
                                            corpus_text_ids=corpus_text_ids_sql,
                                            texts_list=text_list_full_sql,
                                            top=top_sql,
                                            y_classes=y_classes_sql,
                                            verbose=predictions_verbose_sql,
                                            round_to=round_to,
                                            format_as_percentage=format_as_percentage)
            return 1, "The difficult texts list has been generated.", \
                    texts_group_3_sql, overall_quality_score_sql, \
                        overall_quality_score_decimal_sql, overall_quality_score_decimal_previous_sql
        else:
            return 0, """Examples of all labels are not present. 
                              Label more texts then try generating the difficult text list.""", [], "-", 0.0, 0.0
    else:
            return 0, "Label more texts then try generating the difficult text list.", [], "-", 0.0, 0.0

    # except:
    #     return -1, "An error occurred when trying to generate the difficult texts.", [], "-", 0.0, 0.0


def get_top_similar_texts_sql(all_texts_json, similarities_series, top=5, exclude_already_labeled=False, verbose=True):
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

    set_texts_group_x(top_texts=top_texts, table_name="group1Texts")

    return top_texts


def read_new_dataset(source_file_name, text_id_col, text_value_col, source_dir="./output/upload"):
    try:
        dataset_df = pd.read_csv(os.path.join(source_dir, source_file_name))
        dataset_df = dataset_df[[text_id_col, text_value_col]]
        dataset_df.rename(columns={text_id_col: "id", text_value_col: "text"}, inplace=True)
        dataset_df["label"] = "-"
        print("dataset_df :")
        print(dataset_df.head())

        return 1, dataset_df
    except:
        return 0, None


def load_save_state_sql(source_dir="./output/save"):
    # try:
    save_state_json = json.load(open(os.path.join(source_dir, "save_state.json"), "rb"))
    set_pkl(name="DATE_TIME", pkl_data=pickle.load(open(os.path.join(source_dir, "DATE_TIME.pkl"), "rb")), reset=False)
    set_pkl(name="CLICK_LOG", pkl_data=pickle.load(open(os.path.join(source_dir, "CLICK_LOG.pkl"), "rb")), reset=False)
    set_pkl(name="VALUE_LOG", pkl_data=pickle.load(open(os.path.join(source_dir, "VALUE_LOG.pkl"), "rb")), reset=False)
    set_pkl(name="VECTORIZED_CORPUS", pkl_data=pickle.load(open(os.path.join(source_dir, "VECTORIZED_CORPUS.pkl"), "rb")),
            reset=False)
    set_pkl(name="VECTORIZER", pkl_data=pickle.load(open(os.path.join(source_dir, "VECTORIZER.pkl"), "rb")), reset=False)
    set_pkl(name="CLASSIFIER", pkl_data=pickle.load(open(os.path.join(source_dir, "CLASSIFIER.pkl"), "rb")), reset=False)
    set_pkl(name="TEXTS_LIST", pkl_data=pickle.load(open(os.path.join(source_dir, "TEXTS_LIST.pkl"), "rb")), reset=False)
    set_pkl(name="TEXTS_LIST_LIST", pkl_data=pickle.load(open(os.path.join(source_dir, "TEXTS_LIST_LIST.pkl"), "rb")),
            reset=False)
    set_pkl(name="CLASSIFIER", pkl_data=pickle.load(open(os.path.join(source_dir, "CLASSIFIER.pkl"), "rb")), reset=False)
    set_pkl(name="CORPUS_TEXT_IDS", pkl_data=pickle.load(open(os.path.join(source_dir, "CORPUS_TEXT_IDS.pkl"), "rb")),
            reset=False)
    set_pkl(name="TEXTS_GROUP_1", pkl_data=pickle.load(open(os.path.join(source_dir, "TEXTS_GROUP_1.pkl"), "rb")), reset=False)
    set_pkl(name="TEXTS_GROUP_2", pkl_data=pickle.load(open(os.path.join(source_dir, "TEXTS_GROUP_2.pkl"), "rb")), reset=False)
    set_pkl(name="TEXTS_GROUP_3", pkl_data=pickle.load(open(os.path.join(source_dir, "TEXTS_GROUP_3.pkl"), "rb")), reset=False)

    set_variable(name="TOTAL_PAGES", value=save_state_json["TOTAL_PAGES"])
    add_y_classes(y_classses_list=save_state_json["Y_CLASSES"], begin_fresh=True)
    set_variable(name="SHUFFLE_BY", value=save_state_json["SHUFFLE_BY"])
    set_variable(name="HTML_CONFIG_TEMPLATE", value=save_state_json["HTML_CONFIG_TEMPLATE"])
    set_variable(name="DATASET_NAME", value=save_state_json["DATASET_NAME"])

    set_variable(name="LABEL_ALL_BATCH_NO", value=-99)
    set_variable(name="LABEL_ALL_TOTAL_BATCHES", value=0)
    set_variable(name="NUMBER_AUTO_LABELED", value=0)

    test_dataset_name_sql = get_variable_value(name="DATASET_NAME")
    print("test_dataset_name_sql :", test_dataset_name_sql)
    set_variable(name="SEARCH_MESSAGE", value=save_state_json["SEARCH_MESSAGE"])
    set_variable(name="NUMBER_UNLABELED_TEXTS", value=save_state_json["NUMBER_UNLABELED_TEXTS"])
    set_variable(name="LABEL_SUMMARY_STRING", value=save_state_json["LABEL_SUMMARY_STRING"])
    set_variable(name="OVERALL_QUALITY_SCORE", value=save_state_json["OVERALL_QUALITY_SCORE"])
    set_variable(name="OVERALL_QUALITY_SCORE_DECIMAL", value=save_state_json["OVERALL_QUALITY_SCORE_DECIMAL"])
    set_variable(name="OVERALL_QUALITY_SCORE_DECIMAL_PREVIOUS",
                 value=save_state_json["OVERALL_QUALITY_SCORE_DECIMAL_PREVIOUS"])
    set_variable(name="ALLOW_SEARCH_TO_OVERRIDE_EXISTING_LABELS",
                 value=save_state_json["ALLOW_SEARCH_TO_OVERRIDE_EXISTING_LABELS"])
    texts_list_sql = get_pkl(name="TEXTS_LIST")
    # texts_list_list_sql = get_pkl(name="TEXTS_LIST_LIST")
    generate_summary_sql(text_lists=texts_list_sql)
    return 1
    # except:
    #     return 0


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


def get_new_data(source_file, source_folder="./output/upload/", number_samples=None, random_state=1144):
    full_source_file_name = os.path.join(source_folder, source_file)
    data_df = pd.read_csv(full_source_file_name, sep=",")
    if number_samples:
        data_df = data_df.sample(number_samples, random_state=random_state)
        data_df = data_df.reset_index(drop=True)
    return data_df


def convert_demo_data_into_list_json(data_df, limit=50, keep_labels=False,
                                     shuffle_list=[], random_shuffle=False, random_state=21524,
                                     new_label_col_name="assigned_label",
                                     old_label_col_name="event_type",
                                     id_col="tweet_id",
                                     text_col="tweet_text",
                                     label_col="assigned_label"):

    if keep_labels:
        data_df[new_label_col_name] = data_df[old_label_col_name]
    else:
        data_df[new_label_col_name] = "-"

    data_df[id_col] = data_df[id_col].values.astype(str)

    if len(shuffle_list) > 0:
        sort_index = data_df.index.values

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

        data_df = data_df.iloc[sort_indices, :]

    if len(shuffle_list) == 0 and random_shuffle:
        data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    all_texts = data_df[[id_col, text_col, label_col]].values.tolist()

    max_length = len(all_texts)
    if limit < max_length:
        all_texts_adj = random.sample(all_texts, limit)
    else:
        all_texts_adj = all_texts

    all_texts_json = [{"id": text[0], "text": text[1], "label": text[2]} for text in all_texts_adj]
    adj_text_ids = [text[0] for text in all_texts_adj]
    return all_texts_json, adj_text_ids


def convert_new_data_into_list_json(data_df,
                                    limit=50,
                                    shuffle_list=[],
                                    random_shuffle=False,
                                    random_state=21524,
                                    id_col="id",
                                    text_col="text",
                                    label_col="label"):

    data_df[id_col] = data_df[id_col].values.astype(str)
    data_df[label_col] = "-"

    if len(shuffle_list) > 0:
        sort_index = data_df.index.values

        group_dict = {}
        for group in set(shuffle_list):
            group_dict[group] = []

        for (group, index) in zip(shuffle_list, sort_index):
            group_dict[group].append(index)

        dictionaries = list(group_dict.values())

        sort_indices = []
        for sort_indices_tuple in itertools.zip_longest(*dictionaries):
            temp_list = [x for x in [*sort_indices_tuple] if x is not None]
            sort_indices.extend(temp_list)

        data_df = data_df.iloc[sort_indices, :]

    if len(shuffle_list) == 0 and random_shuffle:
        data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    all_texts = data_df[[id_col, text_col, label_col]].values.tolist()

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


def search_all_texts_sql(all_text, include_search_term, exclude_search_term,
                         allow_search_to_override_existing_labels="No",
                         include_behavior="conjunction",
                         exclude_behavior="disjunction",
                         all_upper=True):

    all_text_df = pd.DataFrame(all_text)

    if allow_search_to_override_existing_labels == "No":
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

    search_results_sql = filtered_all_text
    search_results_length_sql = len(filtered_all_text)
    populate_texts_table_sql(filtered_all_text, table_name="searchResults")
    set_variable(name="SEARCH_RESULT_LENGTH", value=search_results_length_sql)
    return search_results_sql, search_results_length_sql


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


def get_similarities_single_record(similarities_df, corpus_text_id):
    keep_indices = [x for x in similarities_df.index.values if x not in [corpus_text_id]]
    similarities = similarities_df[corpus_text_id]
    similarities = similarities.filter(keep_indices)
    similarities = similarities.sort_values(ascending=False)

    return similarities


def generate_click_record(click_location, click_type, click_object, guid=None):
    time_stamp = datetime.now()

    if not guid:
        guid = uuid.uuid4()

    click_record = {"click_id": str(guid),
                     "click_location": click_location,
                     "click_type": click_type,
                     "click_object": click_object,
                     "click_date_time": time_stamp}
    return click_record, guid


def generate_value_record(guid, value_type, value):
    value_record = {"click_id": str(guid),
                     "value_type": value_type,
                     "value": value}
    return value_record


def add_log_record(record, log=[]):
    log.extend(record)
    return None


def get_alert_message(label_summary_sql, overall_quality_score_decimal_sql, overall_quality_score_decimal_previous_sql,
                      texts_group_3_sql):
    # **** Future development ************************************************************************
    # This section will use these variables for future development.
    # The purpose is to give the user more sophisticated feedback while the user labels texts.
    # print("texts_group_3_sql :")
    # print(texts_group_3_sql)
    #
    # print(">> get_alert_message >> label_summary_sql :")
    # print(label_summary_sql)
    # **** Future development ************************************************************************

    if not overall_quality_score_decimal_previous_sql and not overall_quality_score_decimal_sql:
        alert_message = ""
    elif not overall_quality_score_decimal_previous_sql and overall_quality_score_decimal_sql:
        if overall_quality_score_decimal_sql < 0.60:
            alert_message = "More labels are required to improve the overall quality score."
        elif overall_quality_score_decimal_sql < 0.80:
            alert_message = "This is a fairly good start. Keep labeling to try to get the quality score close to 100%"
        else:
            alert_message = "This is a reasonable quality score. Click on the score above to go to the difficult text section."
    elif overall_quality_score_decimal_previous_sql < overall_quality_score_decimal_sql:
        alert_message = "The quality score is improving. Keep labeling."
    elif overall_quality_score_decimal_previous_sql > overall_quality_score_decimal_sql:
        alert_message = "The quality score has dropped. Examine the texts more carefully before assigning a label."
    else:
        alert_message = ""
    return alert_message


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    # ***** Add code for testing the functions here ********************************************************************

    # ******************************************************************************************************************
    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)




