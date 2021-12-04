from flask import Flask, render_template, request, jsonify, flash, make_response, redirect, g, send_file, session
from flask_session import Session
import web_app_utilities as utils
import configuration as config
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import zipfile
import pickle
import json
from werkzeug.utils import secure_filename
import os
import sqlite3
from collections import Counter
import numpy as np
from sklearn.linear_model import SGDClassifier
import pathlib


start_time = datetime.now()
print("*"*20, "Process started @", start_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)

app = Flask(__name__)
# Session(app)
# app.secret_key = "super secret key"
app.config.from_object(__name__)
app.config["UPLOAD_FOLDER"] = "./output/upload/"
# app.config["MAX_CONTENT_PATH"] = 10000

app.jinja_env.add_extension('jinja2.ext.do')

connection = sqlite3.connect('database.db')
with open("schema.sql") as f:
    connection.executescript(f.read())


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def populate_texts_table_sql(texts_list, table_name="texts"):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM " + table_name + ";")
    for text_record in texts_list:
        cur.execute("INSERT INTO " + table_name + " (id, text, label) VALUES (?, ?, ?)",
                    (text_record["id"], text_record["text"], "-"))
    conn.commit()
    conn.close()
    return None


def set_pkl(name, pkl_data, reset=False):
    if not reset:
        pathlib.Path("./output/pkl/").mkdir(parents=True, exist_ok=True)
        pickle.dump(pkl_data, open("./output/pkl/" + name + ".pkl", "wb"))
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


def add_log_click_record_sql(record):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""INSERT INTO clickRecord (click_id, click_location, click_type, click_object, click_date_time) 
                    VALUES (?, ?, ?, ?, ?)""", (record[0]["click_id"], record[0]["click_location"], record[0]["click_type"],
                                                record[0]["click_object"], record[0]["click_date_time"]))

    conn.commit()
    conn.close()

    return None


def add_log_click_value_sql(record):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""INSERT INTO valueRecord (click_id, value_type, value) 
                    VALUES (?, ?, ?)""", (record[0]["click_id"], record[0]["value_type"], record[0]["value"]))

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
    texts_group_2 = [{"id": dict(row)["id"], "text": dict(row)["text"], "label": dict(row)["label"]} for row in sql_table]
    conn.close()
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
    conn = get_db_connection()
    cur = conn.cursor()

    if selected_label and update_ids and not update_objs:
        update_query = "UPDATE texts SET label = ? WHERE id IN (%s)" % ",".join("?"*len(update_ids))
        update_values = [selected_label]
        update_values.extend(update_ids)
        cur.execute(update_query, update_values)

        if not update_in_place:
            cur.execute("""
                        IF OBJECT_ID(N'tempdb..#tempTexts') IS NOT NULL
                        BEGIN
                        DROP TABLE #tempTexts
                        END
                        GO
                        """)
            query = "SELECT * FROM texts INTO #tempTexts WHERE id IN (%s)" % ",".join("?" * len(update_ids))
            cur.execute(query, update_ids)
            cur.executemany("DELETE FROM texts WHERE id IN (%s)" % ",".join("?" * len(update_ids)), update_ids)
            cur.execute("SELECT * INTO texts FROM #tempTexts;")

        conn.commit()
        conn.close()

    # updated_texts_list_df = pd.DataFrame.from_dict(texts_list) # .copy()
    #
    # updated_obj_df = pd.DataFrame.from_dict(updated_obj_lst)
    # update_ids = updated_obj_df["id"].values
    #
    # labels_to_be_updated = \
    #     list(set(updated_texts_list_df[updated_texts_list_df["id"].isin(update_ids)]["label"].values))
    # if len(labels_to_be_updated) > 1 or (len(labels_to_be_updated) == 1 and "-" not in labels_to_be_updated):
    #     # "-" indicates an unlabeled text. So, if the labels_to_be_updated have anything other than "-"
    #     # this indicates that already assigned labeled are going to be overridden.
    #     overridden = True
    # else:
    #     overridden = False
    # labels_got_overridden_flag.clear()
    # labels_got_overridden_flag.extend([overridden])
    #
    # if update_in_place:
    #     update_labels = list(set(updated_obj_df["label"].values))
    #     if len(update_labels) == 1:
    #         updated_texts_list_df.loc[updated_texts_list_df["id"].isin(update_ids), "label"] = update_labels[0]
    #     else:
    #         updated_texts_list_df.loc[updated_texts_list_df["id"].isin(update_ids), "label"] = \
    #             updated_obj_df["label"].values
    # else:
    #     updated_texts_list_df = updated_texts_list_df[~updated_texts_list_df["id"].isin(update_ids)]
    #     updated_texts_list_df = updated_texts_list_df.append(updated_obj_df)
    #
    # updated_texts_list = updated_texts_list_df.to_dict("records")
    # texts_list.clear()
    # texts_list.extend(updated_texts_list)

    elif update_objs and not selected_label and not update_ids:
        update_query = "UPDATE texts SET label = ? WHERE id = ?"
        print("update_query :", update_query)
        update_values = [(obj["label"], obj["id"]) for obj in update_objs]
        for update_value in update_values:
            cur.execute(update_query, update_value)
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

    text_list_full, texts_list_list = \
        update_texts_list_by_id_sql(update_objs=update_objects,
                                    selected_label=None,
                                    update_ids=None,
                                    sub_list_limit=sub_list_limit,
                                    labels_got_overridden_flag=[],
                                    update_in_place=update_in_place)

    # updated_texts_list, updated_texts_list_list, overridden =  \
    #     update_texts_list_by_id(texts_list=texts_list,
    #                             sub_list_limit=sub_list_limit,
    #                             updated_obj_lst=update_objects,
    #                             texts_list_list=texts_list_list,
    #                             update_in_place=update_in_place)

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
                "GROUP_3_KEEP_TOP", "CONFIRM_LABEL_ALL_TEXTS_COUNTS", "SEARCH_TOTAL_PAGES"]:
        value = int(value)

    if name in ["KEEP_ORIGINAL", "GROUP_1_EXCLUDE_ALREADY_LABELED", "GROUP_2_EXCLUDE_ALREADY_LABELED",
                "PREDICTIONS_VERBOSE", "SIMILAR_TEXT_VERBOSE", "FIT_CLASSIFIER_VERBOSE", "FIRST_LABELING_FLAG",
                "FULL_FIT_IF_LABELS_GOT_OVERRIDDEN", "FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS",
                "LABELS_GOT_OVERRIDDEN_FLAG"]:
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
    dataset_name, dataset_url, date_time, y_classes, total_summary = config.has_save_data(source_dir="./output/save")
    if dataset_name:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO availableDatasets (name, description, url) VALUES (?, ?, ?)",
                    (dataset_name[0] + "-" + date_time[0],
                     "A partially labeled dataset having " + total_summary[2]["percentage"] +
                     " of " + total_summary[0]["number"] + " texts labeled.",
                     dataset_url[0]))
        conn.commit()
        conn.close()

    conn = get_db_connection()
    available_datasets_sql = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()
    return available_datasets_sql


def get_all_predictions_sql(fitted_classifier, sparse_vectorized_corpus, corpus_text_ids, texts_list,
                            top=5,
                            y_classes=["earthquake", "fire", "flood", "hurricane"],
                            verbose=False,
                            round_to=2,
                            format_as_percentage=False):

    predictions = fitted_classifier.predict_proba(sparse_vectorized_corpus)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = y_classes

    # predictions_summary_full = predictions_df.describe()
    # predictions_summary_alt = predictions_df.mean(axis=0)

    predictions_summary = predictions_df.replace(0.0, np.NaN).mean(axis=0)

    predictions_df["id"] = corpus_text_ids

    texts_list_df = pd.DataFrame(texts_list)
    predictions_df = predictions_df.merge(texts_list_df, left_on="id", right_on="id")

    keep_cols = ["id", "text"]
    keep_cols.extend(y_classes)
    predictions_df = predictions_df[keep_cols]

    pred_scores = utils.score_predictions(predictions_df[y_classes], use_entropy=True, num_labels=len(y_classes))
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

    # predictions_summary_df = pd.DataFrame({"label": predictions_summary.index,
    #                                        "avg_confidence": predictions_summary.values})
    # predictions_summary_list = predictions_summary_df.to_dict("records")
    # predictions_report.clear()
    # predictions_report.extend(predictions_summary_list)

    # overall_quality_score.clear()
    # overall_quality_score.append(overall_quality)
    set_variable(name="OVERALL_QUALITY_SCORE", value=overall_quality)
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")

    return texts_group_3_sql, overall_quality_score_sql


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
    top_texts = utils.filter_all_texts(texts_list, filter_list, exclude_already_labeled=False)

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
    print("y_classes :", y_classes)
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

    set_pkl(name="CLASSIFIER", pkl_data=clf, reset=False)
    classifier_sql = get_pkl(name="CLASSIFIER")
    return classifier_sql


def load_new_data(source_file,
                  text_id_col,
                  text_value_col,
                  source_folder="./output/upload/",
                  shuffle_by="kmeans",
                  table_limit=50, texts_limit=1000, max_features=100,
                  y_classes=["Label 1", "Label 2", "Label 3", "Label 4"], rnd_state=258):

    data_df = utils.get_new_data(source_file=source_file,
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
        texts_list, adj_text_ids = utils.convert_new_data_into_list_json(data_df,
                                                                         limit=texts_limit,
                                                                         shuffle_list=kmeans_labels,
                                                                         random_shuffle=False,
                                                                         random_state=rnd_state,
                                                                         id_col=text_id_col,
                                                                         text_col=text_value_col,
                                                                         label_col="label")
    else:
        texts_list, adj_text_ids = utils.convert_new_data_into_list_json(data_df,
                                                                         limit=texts_limit,
                                                                         shuffle_list=[],
                                                                         random_shuffle=True,
                                                                         random_state=rnd_state,
                                                                         id_col=text_id_col,
                                                                         text_col=text_value_col,
                                                                         label_col="label")

    texts_list_list = [texts_list[i:i + table_limit] for i in range(0, len(texts_list), table_limit)]
    total_pages = len(texts_list_list)
    set_variable(name="TOTAL_PAGES", value=total_pages)

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids


def load_demo_data_sql(dataset_name="Disaster Tweets Dataset", shuffle_by="kmeans",
                       table_limit=50, texts_limit=1000, max_features=100,
                       y_classes=["Earthquake", "Fire", "Flood", "Hurricane"], rnd_state=258):
    if dataset_name == "Disaster Tweets Dataset":
        consolidated_disaster_tweet_data_df = utils.get_disaster_tweet_demo_data(number_samples=None,
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
            texts_list, adj_text_ids = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                              limit=texts_limit,
                                                                              keep_labels=False,
                                                                              shuffle_list=kmeans_labels,
                                                                              random_shuffle=False,
                                                                              random_state=rnd_state)
        else:
            texts_list, adj_text_ids = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
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
            texts_group_3_sql, overall_quality_score_sql = \
                get_all_predictions_sql(fitted_classifier=classifier_sql,
                                        sparse_vectorized_corpus=vectorized_corpus_sql,
                                        corpus_text_ids=corpus_text_ids_sql,
                                        texts_list=text_list_full_sql,
                                        top=top_sql,
                                        y_classes=y_classes_sql,
                                        verbose=predictions_verbose_sql,
                                        round_to=round_to,
                                        format_as_percentage=format_as_percentage)
            return 1, "The difficult texts list has been generated.", texts_group_3_sql, overall_quality_score_sql
        else:
            return 0, """Examples of all labels are not present. 
                              Label more texts then try generating the difficult text list.""", [], None
    else:
            return 0, "Label more texts then try generating the difficult text list.", [], None

    # except:
    #     return -1, "An error occurred when trying to generate the difficult texts.", [], None


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

    top_texts = utils.filter_all_texts(all_texts_json, filter_list, exclude_already_labeled=False)

    if verbose:
        print(">> all_texts_df > len(top_texts) :", len(top_texts))

    set_texts_group_x(top_texts=top_texts, table_name="group1Texts")

    return top_texts


end_time = datetime.now()
print("*"*20, "Process ended @", end_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)
duration = end_time - start_time
print("*"*20, "Process duration -", duration, "*"*20)


@app.route('/label_entered', methods=['POST'])
def label_entered():
    if request.form["action"] == "add":
        label_entered = request.form["labelEntered"]
        print("label_entered :", label_entered)

        add_y_classes([label_entered], begin_fresh=False)
        y_classes_sql = get_y_classes()
        entered_labels = ", ".join(y_classes_sql)
    else:
        add_y_classes([], begin_fresh=True)
        entered_labels = ""

    prep_data_message1_sql = get_variable_value(name="PREP_DATA_MESSAGE1")
    if len(prep_data_message1_sql) > 0:
        prep_data_message1 = prep_data_message1_sql
    else:
        prep_data_message1 = None

    records_limit = 100

    prep_data_message2_sql = get_variable_value(name="PREP_DATA_MESSAGE2")
    prepared_data_list_sql = get_text_list(table_name="prepData")
    return render_template("start.html",
                           dataset_list=config.DATASETS_AVAILABLE,
                           prep_data_message1=prep_data_message1,
                           prep_data_message2=prep_data_message2_sql,
                           prepared_data_list=prepared_data_list_sql[:records_limit],
                           entered_labels=entered_labels)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        text_id_col = request.form["textIdCol"]
        text_value_col = request.form["textValueCol"]

        set_variable(name="TEXT_ID_COL", value=text_id_col)
        set_variable(name="TEXT_VALUE_COL", value=text_value_col)

        datasets_available_sql = get_variable_value(name="DATASETS_AVAILABLE")

        clear_y_classes()

        if not text_id_col or not text_value_col:
            prep_data_message = "Select BOTH a text ID column and the column that contains the text"
            return render_template("start.html",
                                   dataset_list=datasets_available_sql,
                                   prep_data_message=prep_data_message)
        else:
            result, dataset_df = utils.read_new_dataset(source_file_name=filename,
                                                        text_id_col=text_id_col,
                                                        text_value_col=text_value_col,
                                                        source_dir="./output/upload")
            if result == 1:
                prep_data_message1 = "Data set prepared. Inspect the data set below, and if it appears correct continue."
                set_variable(name="PREP_DATA_MESSAGE1", value=prep_data_message1)


                prepared_data_list = dataset_df.to_dict("records")
                populate_texts_table_sql(texts_list=prepared_data_list, table_name="prepTexts")
                records_limit = 100
                prepared_texts_sql = get_text_list(table_name="prepTexts")
                total_records = len(prepared_texts_sql)

                date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

                prep_data_message2 = "Showing {} of {} records of the prepared data".format(records_limit, total_records)
                set_variable(name="PREP_DATA_MESSAGE2", value=prep_data_message2)
                set_variable(name="DATASET_NAME", value=filename)
                set_variable(name="DATASET_URL", value="-")
                set_variable(name="DATE_TIME", value=date_time)

                return render_template("start.html",
                                       dataset_list=config.DATASETS_AVAILABLE,
                                       prep_data_message1=prep_data_message1,
                                       prep_data_message2=prep_data_message2,
                                       prepared_data_list=prepared_texts_sql[:records_limit])
            else:
                prep_data_message = """Could not prepare data. Make sure to select a columnar dataset that is comma 
                                        separated, with texts which contain commas in quotation marks."""

                return render_template("start.html",
                                       dataset_list=datasets_available_sql,
                                       prep_data_message=prep_data_message)


@app.route("/home")
def home():
    available_datasets_sql = get_available_datasets()
    return render_template("start.html", dataset_list=available_datasets_sql)


@app.route("/")
def index():
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    print("search_exclude_already_labeled_sql :", search_exclude_already_labeled_sql)
    available_datasets_sql = get_available_datasets()
    return render_template("start.html", dataset_list=available_datasets_sql)


@app.route("/dataset_selected", methods=["GET", "POST"])
def dataset_selected():
    dataset_name = request.args.get("dataset_name", None)
    set_variable(name="DATASET_NAME", value=dataset_name)
    dataset_name_sql = get_variable_value(name="DATASET_NAME")

    dataset_url = request.args.get("dataset_url", None)
    set_variable(name="DATASET_URL", value=dataset_url)

    conn = get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    table_limit_sql = get_variable_value(name="TABLE_LIMIT")

    if dataset_name in ["Disaster Tweets Dataset", "Disaster Tweets Dataset with 'Other'"]:
        if dataset_name == "Disaster Tweets Dataset with 'Other'":
            add_y_classes(y_classses_list=["Earthquake", "Fire", "Flood", "Hurricane", "Other"], begin_fresh=True)
        else:
            add_y_classes(y_classses_list=["Earthquake", "Fire", "Flood", "Hurricane"], begin_fresh=True)

        y_classes_sql = get_y_classes()
        set_variable(name="SHUFFLE_BY", value="random")
        shuffle_by_sql = get_variable_value(name="SHUFFLE_BY")
        texts_limit_sql = get_variable_value(name="TEXTS_LIMIT")
        max_features_sql = get_variable_value(name="MAX_FEATURES")
        random_state_sql = get_variable_value(name="RND_STATE")

        texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
            load_demo_data_sql(dataset_name="Disaster Tweets Dataset", shuffle_by=shuffle_by_sql,
                               table_limit=table_limit_sql, texts_limit=texts_limit_sql,
                               max_features=max_features_sql,
                               y_classes=y_classes_sql, rnd_state=random_state_sql)

        populate_texts_table_sql(texts_list=texts_list)

        # for master, update in zip([config.TEXTS_LIST, config.TEXTS_LIST_LIST, config.ADJ_TEXT_IDS,
        #                            config.TOTAL_PAGES, config.VECTORIZED_CORPUS, config.VECTORIZER_LIST,
        #                            config.CORPUS_TEXT_IDS],
        #                           [texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus,
        #                            vectorizer, corpus_text_ids]):
        #     master.clear()
        #     master.append(update)

        text_list_full_sql = get_text_list(table_name="texts")
        text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)

        set_variable(name="TOTAL_PAGES", value=total_pages)
        set_variable(name="SEARCH_MESSAGE", value="")

        set_pkl(name="CLASSIFIER", pkl_data=None, reset=True)
        set_texts_group_x(None, table_name="group1Texts")
        set_texts_group_x(None, table_name="group2Texts")
        reset_difficult_texts_sql()

        set_variable(name="OVERALL_QUALITY_SCORE", value="-")

        return render_template("start.html",
                               dataset_list=available_datasets,
                               texts_list=text_list_list_sql[0],
                               dataset_name=dataset_name_sql,
                               dataset_labels=y_classes_sql)

    else:
        load_status = utils.load_save_state_sql(source_dir="./output/save/")
        if load_status == 1:
            y_classes_sql = get_y_classes()
            text_list_full_sql = get_text_list(table_name="texts")
            text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                       sub_list_limit=table_limit_sql)
            dataset_name_sql = get_variable_value(name="DATASET_NAME")
            return render_template("start.html",
                                   dataset_list=available_datasets,
                                   texts_list=text_list_list_sql[0],
                                   dataset_name=dataset_name_sql,
                                   dataset_labels=y_classes_sql)
        else:
            print("Error loading dataset.")


@app.route("/begin_labeling", methods=["POST"])
def begin_labeling():
    dataset_name = request.args.get("dataset_name", None)
    selected_config = request.form.get("selected_config", None)

    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    set_variable(name="DATE_TIME", value=date_time)

    conn = get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=available_datasets,
                               config1_message=config1_message)

    page_number = 0

    text_list_full_sql = get_text_list(table_name="texts")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
        generate_summary_sql(text_lists=text_list_full_sql)

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    set_variable(name="HTML_CONFIG_TEMPLATE", value=html_config_template)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")

    y_classes_sql = get_y_classes()
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    set_variable(name="SEARCH_RESULTS_LENGTH", value=0)
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")

    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=0,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=text_list_list_sql[page_number],
                           texts_group_1=[],
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=[],
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=[],
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/begin_labeling_new_dataset", methods=["POST"])
def begin_labeling_new_dataset():
    set_variable(name="SHUFFLE_BY", value="random")
    shuffle_by_sql = get_variable_value(name="SHUFFLE_BY")
    dataset_name_sql = get_variable_value(name="DATASET_NAME")
    y_classes_sql = get_y_classes()
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    texts_limit_sql = get_variable_value(name="TEXTS_LIMIT")
    max_features_sql = get_variable_value(name="MAX_FEATURES")
    random_state_sql = get_variable_value(name="RND_STATE")
    text_id_col_sql = get_pkl(name="TEXT_ID_COL")
    text_value_col_sql = get_variable_value(name="TEXT_VALUE_COL")

    conn = get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
        load_new_data(dataset_name_sql,
                      text_id_col=text_id_col_sql,
                      text_value_col=text_value_col_sql,
                      source_folder="./output/upload/",
                      shuffle_by=shuffle_by_sql,
                      table_limit=table_limit_sql,
                      texts_limit=texts_limit_sql,
                      max_features=max_features_sql,
                      y_classes=y_classes_sql,
                      rnd_state=random_state_sql)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM texts;")
    for text_record in texts_list:
        cur.execute("INSERT INTO texts (id, text, label) VALUES (?, ?, ?)",
                    (text_record["id"], text_record["text"], "-"))
    conn.commit()
    conn.close()

    # for master, update in zip([config.TEXTS_LIST, config.TEXTS_LIST_LIST, config.ADJ_TEXT_IDS,
    #                            config.TOTAL_PAGES, config.VECTORIZED_CORPUS, config.VECTORIZER_LIST,
    #                            config.CORPUS_TEXT_IDS],
    #                           [texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus,
    #                            vectorizer, corpus_text_ids]):
    #     master.clear()
    #     master.append(update)

    set_variable(name="TOTAL_PAGES", value=total_pages)
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")

    set_pkl(name="CLASSIFIER", pkl_data=None, reset=True)

    dataset_name = get_variable_value(name="DATASET_NAME")
    selected_config = request.form.get("selected_config_new_data", None)

    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    set_variable(name="DATE_TIME", value=date_time)

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=available_datasets,
                               config1_message=config1_message)

    page_number = 0

    text_list_full_sql = get_text_list(table_name="texts")
    text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
        generate_summary_sql(text_lists=text_list_full_sql)

    set_variable(name="SEARCH_MESSAGE", value="Displaying all texts")

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    set_variable(name="HTML_CONFIG_TEMPLATE", value=html_config_template)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=text_list_list_sql[page_number],
                           texts_group_1=[],
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=[],
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=[],
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/text_labeling", methods=["GET", "POST"])
def text_labeling():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    app_section = request.args.get("app_section", None)
    selected_text_id = request.args.get("selected_text_id", None)

    y_classes_sql = get_y_classes()

    text_list_full_sql = get_text_list(table_name="texts")
    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql, table_limit_sql=table_limit_sql)

    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    keep_original_sql = get_variable_value(name="KEEP_ORIGINAL")
    corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    group_1_exclude_already_labeled_sql = get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    similar_text_verbose_sql = get_variable_value(name="SIMILAR_TEXT_VERBOSE")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")

    label_selected = request.args.get("label_selected", None)
    page_number = int(request.args.get("page_number", None))

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type="text_id_selected",
                                                     click_object="link",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    # Group 1 ************************************************************************************
    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                   corpus_text_ids=corpus_text_ids_sql,
                                                                   text_id=selected_text_id,
                                                                   keep_original=keep_original_sql)

    text_list_full_sql = get_text_list(table_name="texts")

    texts_group_1_sql = get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                  similarities_series=similarities_series,
                                                  top=group_1_keep_top_sql,
                                                  exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                  verbose=similar_text_verbose_sql)

    # **********************************************************************************************

    # Group 2 ************************************************************************************
    classifier_sql = get_pkl(name="CLASSIFIER_LIST")
    if classifier_sql and (label_selected is not None and label_selected != "" and label_selected != "None"):
        texts_group_2_sql = \
            get_top_predictions_sql(selected_class=label_selected,
                                    fitted_classifier=classifier_sql,
                                    sparse_vectorized_corpus=vectorized_corpus_sql,
                                    corpus_text_ids=corpus_text_ids_sql,
                                    texts_list=text_list_full_sql,
                                    top=predictions_number_sql,
                                    cutoff_proba=predictions_probability_sql,
                                    y_classes=y_classes_sql,
                                    verbose=predictions_verbose_sql,
                                    exclude_already_labeled=group_2_exclude_already_labeled_sql)
    else:
        texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags = get_panel_flags()
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    texts_group_3_sql = get_difficult_texts_sql()

    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message="Text selected",
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=text_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/go_to_page", methods=["GET", "POST"])
def go_to_page():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    app_section = request.args.get("app_section", None)
    selection = request.args.get("selection", None)
    page_number = int(request.args.get("page_number", None))
    label_selected = request.args.get("label_selected", None)
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    y_classes_sql = get_y_classes()
    text_list_full_sql = get_text_list(table_name="texts")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type=selection,
                                                     click_object="link",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           label_selected=label_selected,
                           info_message="",
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=text_list_list_sql[page_number],
                           texts_group_1=[],
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/single_text", methods=["POST"])
def single_text():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    new_id = request.form["selected_text_id"]

    y_classes_sql = get_y_classes()
    text_list_full_sql = get_text_list(table_name="texts")
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    new_text = get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql,
                                       table_limit_sql=table_limit_sql)

    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    random_state_sql = get_variable_value(name="RND_STATE")
    labels_got_overridden_sql = get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
    full_fit_if_labels_got_overridden_sql = get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = get_panel_flags()

    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")

    page_number = int(request.form["page_number"])
    new_label = request.form["selected_label_single"]
    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="current_text",
                                                     click_type="single_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=new_label)
    add_log_click_value_sql(record=value_record)
    value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=new_id)
    add_log_click_value_sql(record=value_record)

    if new_label is None or new_label == "" or new_id == "None":
        if new_label == "":
            flash("Select a 'Label'.", "error")
            info_message += f"Select a 'Label'."

        if new_id == "None":
            info_message += "\n" + f"Select a 'Text ID'."

        text_list_list = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=text_list_list[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               initialize_flags=initialize_flags_sql,
                               search_exclude_already_labeled=exclude_already_labeled_sql)
    else:
        new_obj = {"id": new_id, "text": new_text, "label": new_label}
        corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
        vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
        fit_classifier_verbose_sql = get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
        text_list_full_sql, texts_list_list_sql = update_texts_list_by_id_sql(selected_label=new_label,
                                                                              update_ids=[new_id],
                                                                              sub_list_limit=table_limit_sql,
                                                                              labels_got_overridden_flag=[],
                                                                              update_in_place=True)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        classifier_sql = fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
                                            corpus_text_ids=corpus_text_ids_sql,
                                            texts_list=text_list_full_sql,
                                            texts_list_labeled=[new_obj],
                                            y_classes=y_classes_sql,
                                            verbose=fit_classifier_verbose_sql,
                                            random_state=random_state_sql,
                                            n_jobs=-1,
                                            labels_got_overridden_flag=labels_got_overridden_sql,
                                            full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden_sql)

        if classifier_sql and \
                (new_label is not None and new_label != "" and new_label != "None"):
            texts_group_2_sql = \
                get_top_predictions_sql(selected_class=new_label,
                                        fitted_classifier=classifier_sql,
                                        sparse_vectorized_corpus=vectorized_corpus_sql,
                                        corpus_text_ids=corpus_text_ids_sql,
                                        texts_list=text_list_full_sql,
                                        top=predictions_number_sql,
                                        cutoff_proba=predictions_probability_sql,
                                        y_classes=y_classes_sql,
                                        verbose=predictions_verbose_sql,
                                        exclude_already_labeled=exclude_already_labeled_sql)
        else:
            texts_group_2_sql = get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=new_label,
                               info_message="Label assigned",
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=texts_list_list_sql[page_number],
                               texts_group_1=[], # TEXTS_GROUP_1,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/grouped_1_texts", methods=["POST"])
def grouped_1_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    # new_text = request.form["selected_text"]
    selected_label_group1 = request.form["selected_label_group1"]
    info_message = ""

    y_classes_sql = get_y_classes()
    text_list_full_sql = get_text_list(table_name="texts")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql,
                                       table_limit_sql=table_limit_sql)

    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")

    new_text = get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    random_state_sql = get_variable_value(name="RND_STATE")
    fit_classifier_verbose_sql = get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
    full_fit_if_labels_got_overridden_sql = get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
    labels_got_overridden_flag_sql = get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
    initialize_flags_sql = get_panel_flags()

    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    corpus_text_ids = get_pkl(name="CORPUS_TEXT_IDS")

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group1)
    add_log_click_value_sql(record=value_record)

    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    if (selected_label_group1 == "None") or (selected_label_group1 is None) or (selected_label_group1 == "") or \
            (len(texts_group_1_sql) == 0):
        if selected_label_group1 == "":
            info_message += f"Select a 'Label'."

        if len(texts_group_1_sql) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        total_summary_sql = get_total_summary_sql()
        label_summary_sql = get_label_summary_sql()
        label_summary_string_sql = get_variable_value("LABEL_SUMMARY_STRING")
        overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
        texts_group_3_sql = get_difficult_texts_sql()
        texts_group_1_sql = get_texts_group_x(table_name="group1Texts")

        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=text_list_list_sql[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=[],
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)

    else:
        texts_group_1_updated = get_texts_group_x(table_name="group1Texts")
        update_ids = []
        for obj in texts_group_1_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_group1
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            add_log_click_value_sql(record=value_record)

        text_list_full_sql, texts_list_list_sql = update_texts_list_by_id_sql(selected_label=selected_label_group1,
                                                                              update_ids=update_ids,
                                                                              sub_list_limit=table_limit_sql,
                                                                              labels_got_overridden_flag=[],
                                                                              update_in_place=True)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        classifier_sql = \
            fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
                               corpus_text_ids=corpus_text_ids,
                               texts_list=text_list_full_sql,
                               texts_list_labeled=texts_group_1_updated,
                               y_classes=y_classes_sql,
                               verbose=fit_classifier_verbose_sql,
                               random_state=random_state_sql,
                               n_jobs=-1,
                               labels_got_overridden_flag=labels_got_overridden_flag_sql,
                               full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden_sql)

        if classifier_sql and \
                (selected_label_group1 is not None and
                 selected_label_group1 != "" and
                 selected_label_group1 != "None"):
            texts_group_2_sql = \
                get_top_predictions_sql(selected_class=selected_label_group1,
                                        fitted_classifier=classifier_sql,
                                        sparse_vectorized_corpus=vectorized_corpus_sql,
                                        corpus_text_ids=corpus_text_ids,
                                        texts_list=text_list_full_sql,
                                        top=predictions_number_sql,
                                        cutoff_proba=predictions_probability_sql,
                                        y_classes=y_classes_sql,
                                        verbose=predictions_verbose_sql,
                                        exclude_already_labeled=group_2_exclude_already_labeled_sql)
        else:
            texts_group_2_sql = get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, recommendations_summary_sql, overall_quality_score_sql = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
        initialize_flags_sql = get_panel_flags()
        texts_group_3_sql = get_difficult_texts_sql()

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group1,
                               info_message="Labels assigned to group",
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=texts_list_list_sql[page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=recommendations_summary_sql,
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/grouped_2_texts", methods=["POST"])
def grouped_2_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]

    y_classes_sql = get_y_classes()

    text_list_full_sql = get_text_list(table_name="texts")
    new_text = get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql,
                                       table_limit_sql=table_limit_sql)

    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    corpus_text_ids = get_pkl(name="CORPUS_TEXT_IDS")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")

    selected_label_group2 = request.form["selected_label_group2"]

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group2)
    add_log_click_value_sql(record=value_record)

    info_message = ""
    if (selected_label_group2 is None) or (selected_label_group2 == "None") or (selected_label_group2 == "") or \
            (len(selected_label_group2) == 0):
        if selected_label_group2 == "":
            info_message += f"Select a 'Label'."

        if len(selected_label_group2) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                   sub_list_limit=table_limit_sql)
        texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
        texts_group_3_sql = get_difficult_texts_sql()
        total_summary_sql = get_total_summary_sql()
        label_summary_sql = get_label_summary_sql()
        label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
        overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label_group2,
                               info_message=info_message,
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=text_list_list_sql[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=[],
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)

    else:
        texts_group_2_updated = copy.deepcopy(get_texts_group_x(table_name="group2Texts"))
        update_ids = []
        for obj in texts_group_2_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_group2
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            add_log_click_value_sql(record=value_record)

        text_list_full_sql, texts_list_list_sql = update_texts_list_by_id_sql(selected_label=selected_label_group2,
                                                                              update_ids=update_ids,
                                                                              sub_list_limit=table_limit_sql,
                                                                              labels_got_overridden_flag=[],
                                                                              update_in_place=True)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
        corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
        fit_classifier_verbose_sql = get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
        labels_got_overridden_flag_sql = get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
        random_state_sql = get_variable_value(name="RND_STATE")
        full_fit_if_labels_got_overridden_sql = get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
        classifier_sql = fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
                                            corpus_text_ids=corpus_text_ids_sql,
                                            texts_list=text_list_full_sql,
                                            texts_list_labeled=texts_group_2_updated,
                                            y_classes=y_classes_sql,
                                            verbose=fit_classifier_verbose_sql,
                                            random_state=random_state_sql,
                                            n_jobs=-1,
                                            labels_got_overridden_flag=labels_got_overridden_flag_sql,
                                            full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden_sql)

        if len(selected_label_group2) > 0 and \
                (selected_label_group2 is not None and
                 selected_label_group2 != "" and
                 selected_label_group2 != "None"):
            texts_group_2_sql = \
                get_top_predictions_sql(selected_class=selected_label_group2,
                                        fitted_classifier=classifier_sql,
                                        sparse_vectorized_corpus=vectorized_corpus_sql,
                                        corpus_text_ids=corpus_text_ids,
                                        texts_list=text_list_full_sql,
                                        top=predictions_number_sql,
                                        cutoff_proba=predictions_probability_sql,
                                        y_classes=y_classes_sql,
                                        verbose=predictions_verbose_sql,
                                        exclude_already_labeled=group_2_exclude_already_labeled_sql)
        else:
            texts_group_2_sql = get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group2,
                               info_message="Labels assigned to group",
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=texts_list_list_sql[page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/label_all", methods=["POST"])
def label_all():
    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    selected_label = request.form["selected_label"]

    y_classes_sql = get_y_classes()

    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = get_text_list(table_name="searchResults")
    new_text = get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql, table_limit_sql=table_limit_sql)

    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")

    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    initialize_flags_sql = get_panel_flags()

    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    label_summary_sql = get_label_summary_sql()
    total_summary_sql = get_total_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    texts_group_3_sql = get_difficult_texts_sql()
    number_unlabeled_texts_sql = get_variable_value(name="NUMBER_UNLABELED_TEXTS")
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    if html_config_template_sql in ["text_labeling_2.html"]:
        scroll_to_id = "labelAllButton"
    else:
        scroll_to_id = None

    text_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)

    if number_unlabeled_texts_sql == 0:
        info_message = "There are no more unlabeled texts. If you are unhappy with the quality of the " \
                       "auto-labeling, then work through the 'Difficult Texts' to improve the quality."
        label_summary_message = info_message
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label,
                               info_message=info_message,
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=text_list_list_sql[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=[],
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               scroll_to_id=scroll_to_id,
                               label_summary_message=label_summary_message,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)

    classifier_sql = get_pkl(name="CLASSIFIER")
    if classifier_sql:
        confirm_label_all_texts_counts_sql = get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
        if confirm_label_all_texts_counts_sql == 0:
            info_message = \
                f"Are you sure that you want to auto-label the " \
                f"remaining {number_unlabeled_texts_sql:,} texts?"
            label_summary_message = info_message

            temp_count = get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
            set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=(temp_count + 1))

            return render_template(html_config_template_sql,
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=search_message_sql,
                                   search_results_length=search_results_length_sql,
                                   page_number=page_number,
                                   y_classes=y_classes_sql,
                                   total_pages=total_pages_sql,
                                   texts_list=text_list_list_sql[page_number],
                                   texts_group_1=texts_group_1_sql,
                                   group1_table_limit_value=group_1_keep_top_sql,
                                   texts_group_2=[],
                                   group2_table_limit_value=predictions_number_sql,
                                   texts_group_3=texts_group_3_sql,
                                   total_summary=total_summary_sql,
                                   label_summary=label_summary_sql,
                                   recommendations_summary=[],
                                   overall_quality_score=overall_quality_score_sql,
                                   label_summary_string=label_summary_string_sql,
                                   initialize_flags=initialize_flags_sql,
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=search_exclude_already_labeled_sql)

        elif confirm_label_all_texts_counts_sql == 1:
            info_message = \
                f"Click 'Label All Texts' again to confirm your choice. The remaining unlabeled texts will be " \
                f"labeled using machine-learning. Inspect the 'Difficult Texts' to see how well the " \
                f"machine-learning model is trained and if you are satisfied then proceed."
            label_summary_message = info_message
            temp_count = get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
            set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=(temp_count + 1))

            return render_template(html_config_template_sql,
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=search_message_sql,
                                   search_results_length=search_results_length_sql,
                                   page_number=page_number,
                                   y_classes=y_classes_sql,
                                   total_pages=total_pages_sql,
                                   texts_list=text_list_list_sql[page_number],
                                   texts_group_1=texts_group_1_sql,
                                   group1_table_limit_value=group_1_keep_top_sql,
                                   texts_group_2=[],
                                   group2_table_limit_value=predictions_number_sql,
                                   texts_group_3=texts_group_3_sql,
                                   total_summary=total_summary_sql,
                                   label_summary=label_summary_sql,
                                   recommendations_summary=[],
                                   overall_quality_score=overall_quality_score_sql,
                                   label_summary_string=label_summary_string_sql,
                                   initialize_flags=initialize_flags_sql,
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=search_exclude_already_labeled_sql)

        elif confirm_label_all_texts_counts_sql > 1:
            info_message = \
                f"{number_unlabeled_texts_sql} texts were auto-labeled."

            classifier_sql = get_pkl(name="CLASSIFIER")
            vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
            text_list_full_sql, texts_list_list_sql, labeled_text_ids = \
                label_all_sql(fitted_classifier=classifier_sql,
                              sparse_vectorized_corpus=vectorized_corpus_sql,
                              corpus_text_ids=corpus_text_ids_sql,
                              texts_list=text_list_full_sql,
                              label_only_unlabeled=True,
                              sub_list_limit=table_limit_sql,
                              update_in_place=True)

            for labeled_text_id in labeled_text_ids:
                value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=labeled_text_id)
                add_log_click_value_sql(record=value_record)

            total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
                generate_summary_sql(text_lists=text_list_full_sql)

            label_summary_message = info_message
            texts_group_2_sql = get_texts_group_x(table_name="group2Texts")

            return render_template(html_config_template_sql,
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=search_message_sql,
                                   search_results_length=search_results_length_sql,
                                   page_number=page_number,
                                   y_classes=y_classes_sql,
                                   total_pages=total_pages_sql,
                                   texts_list=texts_list_list_sql[page_number],
                                   texts_group_1=[], # texts_group_1_updated,
                                   group1_table_limit_value=group_1_keep_top_sql,
                                   texts_group_2=texts_group_2_sql,
                                   group2_table_limit_value=predictions_number_sql,
                                   texts_group_3=texts_group_3_sql,
                                   total_summary=total_summary_sql,
                                   label_summary=label_summary_sql,
                                   recommendations_summary=[],
                                   overall_quality_score=overall_quality_score_sql,
                                   label_summary_string=label_summary_string_sql,
                                   initialize_flags=initialize_flags_sql,
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=search_exclude_already_labeled_sql)
    # **********************************************************************************************
    else:
        info_message = "Label more texts before trying again."
    return render_template(html_config_template_sql,
                           selected_text_id=new_id,
                           selected_text=new_text,
                           label_selected=selected_label,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=[],
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           scroll_to_id=scroll_to_id,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route('/export_records', methods=['GET', 'POST'])
def export_records():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="export_records",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    download_files = []

    print("After download_files")

    text_list_full_sql = get_text_list(table_name="texts")
    texts_df = pd.DataFrame.from_dict(text_list_full_sql)
    texts_df.to_csv("./output/labeled_texts.csv", index=False)
    download_files.append("./output/labeled_texts.csv")

    click_log_sql = get_click_log()
    click_log_df = pd.DataFrame.from_dict(click_log_sql)
    click_log_df.to_csv("./output/click_log.csv", index=False)
    download_files.append("./output/click_log.csv")

    value_log_sql = get_value_log()
    value_log_df = pd.DataFrame.from_dict(value_log_sql)
    value_log_df.to_csv("./output/value_log.csv", index=False)
    download_files.append("./output/value_log.csv")

    print("After value_log_sql")
    classifier_sql = get_pkl(name="CLASSIFIER")
    if classifier_sql:
        filename = "trained-classifier.pkl"
        pickle.dump(classifier_sql, open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

        vectorizer_sql = get_pkl(name="VECTORIZER")
        filename = "fitted-vectorizer.pkl"
        pickle.dump(vectorizer_sql, open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

    total_summary_sql = get_total_summary_sql()
    if len(total_summary_sql) > 0:
        filename = "total-summary.csv"
        total_summary_df = pd.DataFrame.from_dict(total_summary_sql)
        total_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    label_summary_sql = get_label_summary_sql()
    if len(label_summary_sql) > 0:
        filename = "label-summary.csv"
        label_summary_df = pd.DataFrame.from_dict(label_summary_sql)
        label_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    date_time_sql = get_variable_value(name="DATE_TIME")
    filename = "rapid-labeling-results-" + date_time_sql + ".zip"
    file_path = "./output/download/rapid-labeling-results-" + date_time_sql + ".zip"
    zip_file = zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED)
    for file in download_files:
        zip_file.write(file)
    zip_file.close()
    return send_file(file_path, mimetype='zip', download_name=filename, as_attachment=True)


@app.route('/save_state', methods=['GET', 'POST'])
def save_state():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)
    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="save_state",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    y_classes_sql = get_y_classes()
    text_list_full_sql = get_text_list(table_name="texts")
    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)

    initialize_flags_sql = get_panel_flags()

    classifier_sql = get_pkl(name="CLASSIFIER")
    vectorizer_sql = get_pkl(name="VECTORIZER")
    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    click_log_sql = get_click_log()
    value_log_sql = get_value_log()

    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()

    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")

    save_state = {}
    save_state["DATASET_NAME"] = get_variable_value(name="DATASET_NAME")
    save_state["DATASET_URL"] = get_variable_value(name="DATASET_URL")
    save_state["TOTAL_SUMMARY"] = get_total_summary_sql()
    save_state["LABEL_SUMMARY"] = get_label_summary_sql()
    save_state["RECOMMENDATIONS_SUMMARY"] = []
    # save_state["TEXTS_LIST_LABELED"] = config.TEXTS_LIST_LABELED
    save_state["TEXTS_GROUP_1"] = get_texts_group_x(table_name="group1Texts")
    save_state["TEXTS_GROUP_2"] = get_texts_group_x(table_name="group2Texts")
    save_state["TEXTS_GROUP_3"] = get_difficult_texts_sql()
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    save_state["SEARCH_MESSAGE"] = search_message_sql

    save_state["TEXTS_LIST"] = get_text_list(table_name="texts")
    save_state["TEXTS_LIST_FULL"] = [text_list_full_sql]

    save_state["CORPUS_TEXT_IDS"] = get_pkl(name="CORPUS_TEXT_IDS")
    save_state["TEXTS_LIST_LIST"] = [texts_list_list_sql]
    save_state["TEXTS_LIST_LIST_FULL"] = [texts_list_list_sql]

    # save_state["TOTAL_PAGES_FULL"] = config.TOTAL_PAGES_FULL
    # save_state["ADJ_TEXT_IDS"] = config.ADJ_TEXT_IDS
    save_state["TOTAL_PAGES"] = get_variable_value(name="TOTAL_PAGES")
    save_state["Y_CLASSES"] = get_variable_value(name="Y_CLASSES")

    save_state["SHUFFLE_BY"] = get_variable_value(name="SHUFFLE_BY")
    save_state["NUMBER_UNLABELED_TEXTS"] = get_variable_value(name="NUMBER_UNLABELED_TEXTS")
    save_state["HTML_CONFIG_TEMPLATE"] = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    save_state["LABEL_SUMMARY_STRING"] = get_variable_value(name="LABEL_SUMMARY_STRING")

    save_state["PREDICTIONS_NUMBER"] = get_variable_value(name="PREDICTIONS_NUMBER")
    with open("./output/save/save_state.json", "w") as outfile:
        json.dump(save_state, outfile)

    files = [classifier_sql, vectorizer_sql, vectorized_corpus_sql, click_log_sql, value_log_sql]
    filenames = ["CLASSIFIER", "VECTORIZER", "VECTORIZED_CORPUS", "CLICK_LOG", "VALUE_LOG"]

    for file, filename in zip(files, filenames):
        temp_filename = filename + ".pkl"
        pickle.dump(file, open("./output/save/" + temp_filename), "wb")

    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=0,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[0],
                           texts_group_1=[],
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=[],
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=[],
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route('/update_panels', methods=['GET', 'POST'])
def update_panels():
    panel_flags_sql = get_panel_flags()
    panel_flag = eval(request.args.get("panel_flag", None))
    update_panel_flags_sql(update_flag=panel_flag)
    return panel_flag


@app.route('/label_selected', methods=['GET', 'POST'])
def label_selected():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    label_selected = request.args.get("label_selected")
    selected_text_id = request.args.get("selected_text_id")

    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = get_text_list(table_name="texts")
    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql, table_limit_sql=table_limit_sql)

    y_classes_sql = get_y_classes()
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_exclude_already_labeled_sql = get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    corpus_text_ids = get_pkl(name="CORPUS_TEXT_IDS")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    keep_original_sql = get_variable_value(name="KEEP_ORIGINAL")
    similar_text_verbose_sql = get_variable_value(name="SIMILAR_TEXT_VERBOSE")

    page_number = int(request.args.get("page_number"))

    click_record, guid = utils.generate_click_record(click_location="label",
                                                     click_type="label_selected",
                                                     click_object="radio_button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=label_selected)
    add_log_click_value_sql(record=value_record)

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        set_texts_group_x(None, "group2Texts")
        texts_group_1_sql = []
    else:
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                       corpus_text_ids=corpus_text_ids,
                                                                       text_id=selected_text_id,
                                                                       keep_original=keep_original_sql)

        texts_group_1_sql = get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                      similarities_series=similarities_series,
                                                      top=group_1_keep_top_sql,
                                                      exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                      verbose=similar_text_verbose_sql)
    # **********************************************************************************************

    # Group 2 ************************************************************************************
    classifier_sql = get_pkl(name="CLASSIFIER")
    if classifier_sql and label_selected != "None":
        texts_group_2_sql = \
            get_top_predictions_sql(selected_class=label_selected,
                                    fitted_classifier=classifier_sql,
                                    sparse_vectorized_corpus=vectorized_corpus_sql,
                                    corpus_text_ids=corpus_text_ids,
                                    texts_list=text_list_full_sql,
                                    top=predictions_number_sql,
                                    cutoff_proba=predictions_probability_sql,
                                    y_classes=y_classes_sql,
                                    verbose=predictions_verbose_sql,
                                    exclude_already_labeled=group_2_exclude_already_labeled_sql)
    else:
        texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************

    info_message = f"'{label_selected}' selected"


    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    initialize_flags_sql = get_panel_flags()
    label_summary_sql = get_label_summary_sql()
    total_summary_sql = get_total_summary_sql()
    texts_group_3_sql = get_difficult_texts_sql()
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=text_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route('/generate_difficult_texts', methods=['POST'])
def generate_difficult_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    id = request.form["selected_text_id"]
    label_selected = request.form["selected_label"]

    y_classes_sql = get_y_classes()
    text_list_full_sql = get_text_list(table_name="texts")
    table_limit = int(request.form.get("group1_table_limit", None))
    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    text = get_selected_text(selected_text_id=id, text_list_full_sql=text_list_full_sql)

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="generate_list",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    # Group 3 ************************************************************************************

    result, message, texts_group_3_sql, overall_quality_score_sql = \
        generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                full_fit_if_labels_got_overridden=True, round_to=1,
                                                format_as_percentage=True)
    info_message = message

    # **********************************************************************************************

    if html_config_template_sql in ["text_labeling_2.html"]:
        scroll_to_id = "labelAllButton"
        difficult_texts_message = info_message
    else:
        scroll_to_id = None
        difficult_texts_message = None

    return render_template(html_config_template_sql,
                           selected_text_id=id,
                           selected_text=text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           scroll_to_id=scroll_to_id,
                           difficult_texts_message=difficult_texts_message,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/set_group_1_record_limit", methods=["POST"])
def set_group_1_record_limit():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    y_classes_sql = get_y_classes()
    table_limit = int(request.form.get("group1_table_limit", None))
    text_list_full_sql = get_text_list(table_name="texts")
    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_exclude_already_labeled_sql = get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    keep_original_sql = get_variable_value(name="KEEP_ORIGINAL")

    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    label_selected = request.form.get("label_selected", None)
    info_message = f"Set 'Similarity' display limit to {table_limit}"
    set_variable(name="GROUP_1_KEEP_TOP", value=table_limit)
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=table_limit)
    add_log_click_value_sql(record=value_record)

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        set_texts_group_x(None, table_name="group1Texts")
        texts_group_1_sql = []
    else:
        vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
        corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
        similar_text_verbose_sql = get_pkl(name="SIMILAR_TEXT_VERBOSE")
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                       corpus_text_ids=corpus_text_ids_sql,
                                                                       text_id=selected_text_id,
                                                                       keep_original=keep_original_sql)

        texts_group_1_sql = get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                      similarities_series=similarities_series,
                                                      top=group_1_keep_top_sql,
                                                      exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                      verbose=similar_text_verbose_sql)

    # **********************************************************************************************
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = get_panel_flags()

    texts_group_3_sql = get_difficult_texts_sql()
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    # texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    print("texts_group_1_sql :", texts_group_1_sql)
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/set_group_2_record_limit", methods=["POST"])
def set_group_2_record_limit():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)

    y_classes_sql = get_y_classes()
    group_2_table_limit = int(request.form.get("group2_table_limit", None))

    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = get_text_list(table_name="texts")
    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    initialize_flags_sql = get_panel_flags()
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    label_selected = request.form.get("label_selected", None)


    info_message = f"Set 'Recommendations' display limit to {group_2_table_limit}"
    set_variable(name="PREDICTIONS_NUMBER", value=group_2_table_limit)
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=group_2_table_limit)
    add_log_click_value_sql(record=value_record)

    # Group 2 ************************************************************************************
    classifier_sql = get_pkl(name="CLASSIFIER")
    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    if classifier_sql:
        texts_group_2_sql = \
            get_top_predictions_sql(selected_class=label_selected,
                                    fitted_classifier=classifier_sql,
                                    sparse_vectorized_corpus=vectorized_corpus_sql,
                                    corpus_text_ids=corpus_text_ids_sql,
                                    texts_list=text_list_full_sql,
                                    top=predictions_number_sql,
                                    cutoff_proba=predictions_probability_sql,
                                    y_classes=y_classes_sql,
                                    verbose=predictions_verbose_sql,
                                    exclude_already_labeled=group_2_exclude_already_labeled_sql)
    else:
        texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    total_summary_sql = get_total_summary_sql()
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=page_number,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[page_number],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/search_all_texts", methods=["POST"])
def search_all_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)

    text_list_full_sql = get_text_list(table_name="texts")
    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    total_summary_sql = get_total_summary_sql()
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    initialize_flags_sql = get_panel_flags()
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    label_selected = request.form.get("label_selected", None)
    include_search_term = request.form.get("include_search_term", None)
    exclude_search_term = request.form.get("exclude_search_term", None)
    allow_search_override_labels = request.form.get('searchAllTextRadio', None)

    if allow_search_override_labels == "Yes":
        set_variable(name="SEARCH_EXCLUDE_ALREADY_LABELED", value=False)
    else:
        set_variable(name="SEARCH_EXCLUDE_ALREADY_LABELED", value=True)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="search_texts",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)
    value_record_1 = utils.generate_value_record(guid=guid, value_type="include_search_term", value=include_search_term)
    add_log_click_value_sql(record=value_record_1)
    value_record_2 = utils.generate_value_record(guid=guid, value_type="exclude_search_term", value=exclude_search_term)
    add_log_click_value_sql(record=value_record_2)

    y_classes_sql = get_y_classes()

    if len(include_search_term) > 0 or len(exclude_search_term) > 0:
        search_results_sql, search_results_length_sql = \
            utils.search_all_texts_sql(all_text=text_list_full_sql,
                                       include_search_term=include_search_term,
                                       exclude_search_term=exclude_search_term,
                                       search_exclude_already_labeled=search_exclude_already_labeled_sql,
                                       include_behavior="conjunction",
                                       exclude_behavior="conjunction",
                                       all_upper=True)

        if search_results_length_sql > 0:
            search_results_list = \
                [search_results_sql[i:i + table_limit_sql] for i in range(0, len(search_results_sql), table_limit_sql)]
            search_results_total_pages = len(search_results_list)
            search_result_page_number = 0

            set_variable(name="SEARCH_TOTAL_PAGES", value=search_results_total_pages)

            info_message = f"Showing results Include-'{include_search_term}', Exclude-'{exclude_search_term}'"
            set_variable(name="SEARCH_MESSAGE", value=info_message)
            set_variable(name="SEARCH_RESULTS_LENGTH", value=search_results_length_sql)

            return render_template(html_config_template_sql,
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   search_message=info_message,
                                   search_results_length=search_results_length_sql,
                                   page_number=search_result_page_number,
                                   y_classes=y_classes_sql,
                                   total_pages=search_results_total_pages,
                                   texts_list=search_results_list[search_result_page_number],
                                   texts_group_1=texts_group_1_sql,
                                   group1_table_limit_value=group_1_keep_top_sql,
                                   texts_group_2=texts_group_2_sql,
                                   group2_table_limit_value=predictions_number_sql,
                                   texts_group_3=texts_group_3_sql,
                                   total_summary=total_summary_sql,
                                   label_summary=label_summary_sql,
                                   recommendations_summary=[],
                                   overall_quality_score=overall_quality_score_sql,
                                   label_summary_string=label_summary_string_sql,
                                   initialize_flags=initialize_flags_sql,
                                   search_exclude_already_labeled=search_exclude_already_labeled_sql)
        else:
            info_message = f"No search results for Include-'{include_search_term}', Exclude-'{exclude_search_term}'"
            set_variable(name="SEARCH_MESSAGE", value=info_message)
            set_variable(name="SEARCH_RESULTS_LENGTH", value=0)
            set_variable(name="SEARCH_TOTAL_PAGES", value=0)
            page_number = -1
            search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
            return render_template(html_config_template_sql,
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   search_message=search_message_sql,
                                   search_results_length=0,
                                   page_number=page_number,
                                   y_classes=y_classes_sql,
                                   total_pages=0,
                                   texts_list=[[]][0],
                                   texts_group_1=texts_group_1_sql,
                                   group1_table_limit_value=group_1_keep_top_sql,
                                   texts_group_2=texts_group_2_sql,
                                   group2_table_limit_value=predictions_number_sql,
                                   texts_group_3=texts_group_3_sql,
                                   total_summary=total_summary_sql,
                                   label_summary=label_summary_sql,
                                   recommendations_summary=[],
                                   overall_quality_score=overall_quality_score_sql,
                                   label_summary_string=label_summary_string_sql,
                                   initialize_flags=initialize_flags_sql,
                                   search_exclude_already_labeled=search_exclude_already_labeled_sql)
    else:
        info_message = "No search term entered"
        # text_list_full_sql = get_text_list(table_name="texts")
        text_list_list = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
        set_variable(name="SEARCH_MESSAGE", value=info_message)
        return render_template(html_config_template_sql,
                               selected_text_id=selected_text_id,
                               selected_text=selected_text,
                               label_selected=label_selected,
                               info_message=info_message,
                               search_message=info_message,
                               search_results_length=0,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=text_list_list[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/grouped_search_texts", methods=["POST"])
def grouped_search_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]

    y_classes_sql = get_y_classes()

    text_list_full_sql = get_text_list(table_name="texts")
    new_text = get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql,
                                       table_limit_sql=table_limit_sql)

    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")

    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    selected_label_search_texts = request.form["selected_label_search_texts"]
    vectorized_corpus_sql = get_pkl(name="VECTORIZED_CORPUS")
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    initialize_flags_sql = get_panel_flags()
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    predictions_probability_sql = get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")

    corpus_text_ids_sql = get_pkl(name="CORPUS_TEXT_IDS")
    random_state_sql = get_variable_value(name="RND_STATE")
    fit_classifier_verbose_sql = get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
    full_fit_if_labels_got_overridden_sql = get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")

    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_search_texts)
    add_log_click_value_sql(record=value_record)

    if (selected_label_search_texts == "None") or (selected_label_search_texts is None) \
            or (selected_label_search_texts == ""):
        if selected_label_search_texts == "":
            info_message += f"Select a 'Label'."

        # text_list_list = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
        total_summary_sql = get_total_summary_sql()
        label_summary_sql = get_label_summary_sql()
        label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
        search_results_texts_list_sql = get_text_list(table_name="searchResults")
        search_results_texts_list_list_sql = create_text_list_list(text_list_full_sql=search_results_texts_list_sql,
                                                                   sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=search_results_texts_list_list_sql[page_number],
                               texts_group_1=texts_group_1_sql,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=[],
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)

    else:
        texts_group_updated = get_text_list(table_name="searchResults")
        update_ids = []
        for obj in texts_group_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_search_texts
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            add_log_click_value_sql(record=value_record)

        text_list_full_sql, text_list_list_sql = \
            update_texts_list_by_id_sql(selected_label=selected_label_search_texts,
                                        update_ids=update_ids,
                                        sub_list_limit=table_limit_sql,
                                        labels_got_overridden_flag=[],
                                        update_in_place=True)

        set_text_list(label=selected_label_search_texts, table_name="searchResults")

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        labels_got_overridden_flag_sql = get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
        classifier_sql = fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
                                            corpus_text_ids=corpus_text_ids_sql,
                                            texts_list=text_list_full_sql,
                                            texts_list_labeled=texts_group_updated,
                                            y_classes=y_classes_sql,
                                            verbose=fit_classifier_verbose_sql,
                                            random_state=random_state_sql,
                                            n_jobs=-1,
                                            labels_got_overridden_flag=labels_got_overridden_flag_sql,
                                            full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden_sql)

        if classifier_sql and \
                (selected_label_search_texts is not None and
                 selected_label_search_texts != "" and
                 selected_label_search_texts != "None"):
            texts_group_2_sql = \
                get_top_predictions_sql(selected_class=selected_label_search_texts,
                                        fitted_classifier=classifier_sql,
                                        sparse_vectorized_corpus=vectorized_corpus_sql,
                                        corpus_text_ids=corpus_text_ids_sql,
                                        texts_list=text_list_full_sql,
                                        top=predictions_number_sql,
                                        cutoff_proba=predictions_probability_sql,
                                        y_classes=y_classes_sql,
                                        verbose=predictions_verbose_sql,
                                        exclude_already_labeled=group_2_exclude_already_labeled_sql)
        else:
            texts_group_2_sql = get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)
        # **********************************************************************************************

        search_results_texts_list_sql = get_text_list(table_name="searchResults")
        search_results_texts_list_list_sql = create_text_list_list(text_list_full_sql=search_results_texts_list_sql,
                                                                   sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_search_texts,
                               info_message="Labels assigned to group",
                               search_message=search_message_sql,
                               search_results_length=search_results_length_sql,
                               page_number=page_number,
                               y_classes=y_classes_sql,
                               total_pages=total_pages_sql,
                               texts_list=search_results_texts_list_list_sql[page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=group_1_keep_top_sql,
                               texts_group_2=texts_group_2_sql,
                               group2_table_limit_value=predictions_number_sql,
                               texts_group_3=texts_group_3_sql,
                               total_summary=total_summary_sql,
                               label_summary=label_summary_sql,
                               recommendations_summary=[],
                               overall_quality_score=overall_quality_score_sql,
                               label_summary_string=label_summary_string_sql,
                               initialize_flags=initialize_flags_sql,
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=search_exclude_already_labeled_sql)


@app.route("/clear_search_all_texts", methods=["POST"])
def clear_search_all_texts():
    set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    selected_text_id = request.form.get("selected_text_id", None)

    y_classes_sql = get_y_classes()
    table_limit_sql = get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = get_text_list(table_name="texts")
    texts_list_list_sql = create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    selected_text = get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    html_config_template_sql = get_variable_value(name="HTML_CONFIG_TEMPLATE")
    predictions_number_sql = get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = get_variable_value(name="OVERALL_QUALITY_SCORE")
    search_results_length_sql = get_variable_value(name="SEARCH_RESULTS_LENGTH")
    label_summary_sql = get_label_summary_sql()
    label_summary_string_sql = get_variable_value(name="LABEL_SUMMARY_STRING")
    total_summary_sql = get_total_summary_sql()
    texts_group_1_sql = get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = get_difficult_texts_sql()
    initialize_flags_sql = get_panel_flags()
    group_1_keep_top_sql = get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    label_selected = request.form.get("label_selected", None)
    info_message = "Search cleared"

    # config.SEARCH_MESSAGE.clear()
    # config.SEARCH_MESSAGE.append("Displaying all texts")
    set_variable(name="SEARCH_MESSAGE", value="Displaying all texts")

    # config.TEXTS_LIST.clear()
    # config.TEXTS_LIST.append(config.TEXTS_LIST_FULL[0])

    # config.TEXTS_LIST_LIST.clear()
    # config.TEXTS_LIST_LIST.append(config.TEXTS_LIST_LIST_FULL[0])

    # config.TOTAL_PAGES.clear()
    # config.TOTAL_PAGES.append(config.TOTAL_PAGES_FULL[0])
    total_pages_sql = get_variable_value(name="TOTAL_PAGES")

    # config.SEARCH_RESULT_LENGTH.clear()
    # config.SEARCH_RESULT_LENGTH.append(0)
    set_variable(name="SEARCH_RESULTS_LENGTH", value=0)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="clear_search",
                                                     click_object="button",
                                                     guid=None)
    add_log_click_record_sql(record=click_record)
    search_message_sql = get_variable_value(name="SEARCH_MESSAGE")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=search_message_sql,
                           search_results_length=search_results_length_sql,
                           page_number=0,
                           y_classes=y_classes_sql,
                           total_pages=total_pages_sql,
                           texts_list=texts_list_list_sql[0],
                           texts_group_1=texts_group_1_sql,
                           group1_table_limit_value=group_1_keep_top_sql,
                           texts_group_2=texts_group_2_sql,
                           group2_table_limit_value=predictions_number_sql,
                           texts_group_3=texts_group_3_sql,
                           total_summary=total_summary_sql,
                           label_summary=label_summary_sql,
                           recommendations_summary=[],
                           overall_quality_score=overall_quality_score_sql,
                           label_summary_string=label_summary_string_sql,
                           initialize_flags=initialize_flags_sql,
                           search_exclude_already_labeled=search_exclude_already_labeled_sql)


if __name__ == "__main__":
    app.run() # host="127.0.0.1:5000"

