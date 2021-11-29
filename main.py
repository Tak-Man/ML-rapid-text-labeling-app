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

connection = sqlite3.connect('database.db')
with open("schema.sql") as f:
    connection.executescript(f.read())


start_time = datetime.now()
print("*"*20, "Process started @", start_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)

app = Flask(__name__)
# Session(app)
app.secret_key = "super secret key"
app.config.from_object(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_PATH"] = config.MAX_CONTENT_PATH

app.jinja_env.add_extension('jinja2.ext.do')


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def load_new_data(source_file,
                  source_folder="./output/upload/",
                  shuffle_by="kmeans",
                  table_limit=50, texts_limit=1000, max_features=100,
                  y_classes=["Label 1", "Label 2", "Label 3", "Label 4"], rnd_state=258):

    data_df = utils.get_new_data(source_file=source_file,
                                 source_folder=source_folder,
                                 number_samples=None,
                                 random_state=config.RND_STATE)
    print("data_df :")
    print(data_df.head())
    corpus_text_ids = [str(x) for x in data_df[config.TEXT_ID_COL[0]].values]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)
    print("config.TEXT_VALUE_COL[0] :", config.TEXT_VALUE_COL[0])
    vectorized_corpus = vectorizer.fit_transform(data_df[config.TEXT_VALUE_COL[0]].values)

    if shuffle_by == "kmeans":
        kmeans = KMeans(n_clusters=len(y_classes), random_state=config.RND_STATE).fit(vectorized_corpus)
        kmeans_labels = kmeans.labels_
        texts_list, adj_text_ids = utils.convert_new_data_into_list_json(data_df,
                                                                         limit=texts_limit,
                                                                         shuffle_list=kmeans_labels,
                                                                         random_shuffle=False,
                                                                         random_state=rnd_state,
                                                                         id_col=config.TEXT_ID_COL[0],
                                                                         text_col=config.TEXT_VALUE_COL[0],
                                                                         label_col="label")
    else:
        texts_list, adj_text_ids = utils.convert_new_data_into_list_json(data_df,
                                                                         limit=texts_limit,
                                                                         shuffle_list=[],
                                                                         random_shuffle=True,
                                                                         random_state=rnd_state,
                                                                         id_col=config.TEXT_ID_COL[0],
                                                                         text_col=config.TEXT_VALUE_COL[0],
                                                                         label_col="label")

    texts_list_list = [texts_list[i:i + table_limit] for i in range(0, len(texts_list), table_limit)]
    total_pages = len(texts_list_list)

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids


def load_demo_data(dataset_name="Disaster Tweets Dataset", shuffle_by="kmeans",
                   table_limit=50, texts_limit=1000, max_features=100,
                   y_classes=["Earthquake", "Fire", "Flood", "Hurricane"], rnd_state=258):
    if dataset_name == "Disaster Tweets Dataset":
        consolidated_disaster_tweet_data_df = utils.get_disaster_tweet_demo_data(number_samples=None,
                                                                                 filter_data_types=["train"],
                                                                                 random_state=config.RND_STATE)
        corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)
        # https://stackoverflow.com/questions/69326639/sklearn-warnings-in-version-1-0
        vectorized_corpus = \
            vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"].values)

        if shuffle_by == "kmeans":
            kmeans = KMeans(n_clusters=len(y_classes), random_state=config.RND_STATE).fit(vectorized_corpus)
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

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids


def generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                            full_fit_if_labels_got_overridden=True, round_to=1,
                                            format_as_percentage=True):
    try:
        if len(config.CLASSIFIER_LIST) > 0:
            label_summary_df = pd.DataFrame.from_dict(config.LABEL_SUMMARY)
            y_classes_labeled = label_summary_df["name"].values
            all_classes_present = all(label in y_classes_labeled for label in config.Y_CLASSES[0])
            if all_classes_present:
                if config.FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS:
                    texts_group_updated = copy.deepcopy(config.TEXTS_LIST[0])
                    utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                         corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                         texts_list=config.TEXTS_LIST_FULL[0],
                                         texts_list_labeled=texts_group_updated,
                                         y_classes=config.Y_CLASSES[0],
                                         verbose=config.FIT_CLASSIFIER_VERBOSE,
                                         classifier_list=config.CLASSIFIER_LIST,
                                         random_state=config.RND_STATE,
                                         n_jobs=n_jobs,
                                         labels_got_overridden_flag=labels_got_overridden_flag,
                                         full_fit_if_labels_got_overridden=full_fit_if_labels_got_overridden)

                utils.get_all_predictions(fitted_classifier=config.CLASSIFIER_LIST[0],
                                          sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                          corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                          texts_list=config.TEXTS_LIST[0],
                                          top=config.GROUP_3_KEEP_TOP,
                                          y_classes=config.Y_CLASSES[0],
                                          verbose=config.PREDICTIONS_VERBOSE,
                                          round_to=round_to,
                                          format_as_percentage=format_as_percentage,
                                          similar_texts=config.TEXTS_GROUP_3,
                                          predictions_report=config.RECOMMENDATIONS_SUMMARY,
                                          overall_quality_score=config.OVERALL_QUALITY_SCORE)
                return 1, "The difficult texts list has been generated."
            else:
                return 1, """Examples of all labels are not present. 
                                  Label more texts then try generating the difficult text list."""
        else:
                return 1, "Label more texts then try generating the difficult text list."

    except:
        return 0, "An error occurred when trying to generate the difficult texts."


end_time = datetime.now()
print("*"*20, "Process ended @", end_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)
duration = end_time - start_time
print("*"*20, "Process duration -", duration, "*"*20)


@app.route('/label_entered', methods=['POST'])
def label_entered():
    if request.form["action"] == "add":
        label_entered = request.form["labelEntered"]
        print("label_entered :", label_entered)

        if len(config.Y_CLASSES) == 0:
            config.Y_CLASSES.append([label_entered])
        else:
            config.Y_CLASSES[0].append(label_entered)
            temp_y_classes = copy.deepcopy(config.Y_CLASSES[0])
            temp_y_classes = list(set(temp_y_classes))
            config.Y_CLASSES.clear()
            config.Y_CLASSES.append(temp_y_classes)


        entered_labels = ", ".join(config.Y_CLASSES[0])
    else:
        config.Y_CLASSES.clear()
        entered_labels = ""

    if len(config.PREP_DATA_MESSAGE1) > 0:
        prep_data_message1 = config.PREP_DATA_MESSAGE1[0]
    else:
        prep_data_message1 = None

    records_limit = 100

    return render_template("start.html",
                           dataset_list=config.DATASETS_AVAILABLE,
                           prep_data_message1=prep_data_message1,
                           prep_data_message2=config.PREP_DATA_MESSAGE2[0],
                           prepared_data_list=config.PREPARED_DATA_LIST[0][:records_limit],
                           entered_labels=entered_labels)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        text_id_col = request.form["textIdCol"]
        text_value_col = request.form["textValueCol"]

        config.TEXT_ID_COL.clear()
        config.TEXT_ID_COL.append(text_id_col)

        config.TEXT_VALUE_COL.clear()
        config.TEXT_VALUE_COL.append(text_value_col)

        config.Y_CLASSES.clear()

        if not text_id_col or not text_value_col:
            prep_data_message = "Select BOTH a text ID column and the column that contains the text"
            return render_template("start.html",
                                   dataset_list=config.DATASETS_AVAILABLE,
                                   prep_data_message=prep_data_message)
        else:
            result, dataset_df = utils.read_new_dataset(source_file_name=filename,
                                                        text_id_col=config.TEXT_ID_COL[0],
                                                        text_value_col=config.TEXT_VALUE_COL[0],
                                                        source_dir="./output/upload")
            if result == 1:
                prep_data_message1 = "Data set prepared. Inspect the data set below, and if it appears correct continue."
                config.PREP_DATA_MESSAGE1.clear()
                config.PREP_DATA_MESSAGE1.append(prep_data_message1)

                prepared_data_list = dataset_df.to_dict("records")
                config.PREPARED_DATA_LIST.clear()
                config.PREPARED_DATA_LIST.append(prepared_data_list)
                records_limit = 100
                total_records = len(config.PREPARED_DATA_LIST[0])

                prep_data_message2 = "Showing {} of {} records of the prepared data".format(records_limit, total_records)
                config.PREP_DATA_MESSAGE2.clear()
                config.PREP_DATA_MESSAGE2.append(prep_data_message2)

                config.DATASET_NAME.clear()
                config.DATASET_NAME.append(filename)

                config.DATASET_URL.clear()
                config.DATASET_URL.append("-")

                date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                config.DATE_TIME.clear()
                config.DATE_TIME.append(date_time)

                return render_template("start.html",
                                       dataset_list=config.DATASETS_AVAILABLE,
                                       prep_data_message1=config.PREP_DATA_MESSAGE1[0],
                                       prep_data_message2=config.PREP_DATA_MESSAGE2[0],
                                       prepared_data_list=config.PREPARED_DATA_LIST[0][:records_limit])
            else:
                prep_data_message = """Could not prepare data. Make sure to select a columnar dataset that is comma 
                                        separated, with texts which contain commas in quotation marks."""

                return render_template("start.html",
                                       dataset_list=config.DATASETS_AVAILABLE,
                                       prep_data_message=prep_data_message)


@app.route("/home")
def home():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM availableDatasets;")
    cur.execute("INSERT INTO availableDatasets SELECT * FROM fixedDatasets;")
    conn.commit()
    conn.close()
    dataset_name, dataset_url, date_time, y_classes, total_summary = config.has_save_data(source_dir="./output/save")
    if dataset_name:
            # temp_dict = {"name": dataset_name[0] + "-" + date_time[0],
            #              "description": "A partially labeled dataset having " + total_summary[2]["percentage"] +
            #                             " of " + total_summary[0]["number"] + " texts labeled.",
            #              "url": dataset_url[0]}
        # temp_datasets_available = copy.deepcopy(config.FIXED_DATASETS[:2])
        # temp_datasets_available.append(temp_dict)
        # config.DATASETS_AVAILABLE.clear()
        # config.DATASETS_AVAILABLE.extend(temp_datasets_available)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO availableDatasets (name, description, url) VALUES (?, ?, ?)",
                    (dataset_name[0] + "-" + date_time[0],
                     "A partially labeled dataset having " + total_summary[2]["percentage"] +
                     " of " + total_summary[0]["number"] + " texts labeled.",
                     dataset_url[0]))
        conn.commit()
        conn.close()

        # return render_template("start.html", dataset_list=config.DATASETS_AVAILABLE)
    # else:
        # config.DATASETS_AVAILABLE = config.FIXED_DATASETS
    conn = get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    return render_template("start.html", dataset_list=available_datasets)


@app.route("/")
def index():
    dataset_name, dataset_url, date_time, y_classes, total_summary = config.has_save_data(source_dir="./output/save")
    if dataset_name:
        temp_dict = {"name": dataset_name[0] + "-" + date_time[0],
                     "description": "A partially labeled dataset having " + total_summary[2]["percentage"] +
                                    " of " + total_summary[0]["number"] + " texts labeled.",
                     "url": dataset_url[0]}

        temp_datasets_available = copy.deepcopy(config.FIXED_DATASETS[:2])
        temp_datasets_available.append(temp_dict)
        config.DATASETS_AVAILABLE.clear()
        config.DATASETS_AVAILABLE.extend(temp_datasets_available)
    else:
        config.DATASETS_AVAILABLE = config.FIXED_DATASETS
    return render_template("start.html",
                           dataset_list=config.DATASETS_AVAILABLE)


@app.route("/dataset_selected", methods=["GET", "POST"])
def dataset_selected():
    dataset_name = request.args.get("dataset_name", None)
    config.DATASET_NAME.clear()
    config.DATASET_NAME.append(dataset_name)

    dataset_url = request.args.get("dataset_url", None)
    config.DATASET_URL.clear()
    config.DATASET_URL.append(dataset_url)

    if dataset_name in ["Disaster Tweets Dataset", "Disaster Tweets Dataset with 'Other'"]:
        config.Y_CLASSES.clear()
        if dataset_name == "Disaster Tweets Dataset with 'Other'":
            config.Y_CLASSES.append(["Earthquake", "Fire", "Flood", "Hurricane", "Other"])
        else:
            config.Y_CLASSES.append(["Earthquake", "Fire", "Flood", "Hurricane"])

        config.SHUFFLE_BY.clear()
        config.SHUFFLE_BY.append("random")  # kmeans random

        texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
            load_demo_data(dataset_name="Disaster Tweets Dataset", shuffle_by=config.SHUFFLE_BY[0],
                           table_limit=config.TABLE_LIMIT, texts_limit=config.TEXTS_LIMIT,
                           max_features=config.MAX_FEATURES,
                           y_classes=config.Y_CLASSES[0], rnd_state=config.RND_STATE)

        conn = get_db_connection()
        cur = conn.cursor()
        for text_record in texts_list:
            cur.execute("INSERT INTO texts (id, text, label) VALUES (?, ?, ?)",
                        (text_record["id"], text_record["text"], "-"))
        conn.commit()
        conn.close()

        for master, update in zip([config.TEXTS_LIST, config.TEXTS_LIST_LIST, config.ADJ_TEXT_IDS,
                                   config.TOTAL_PAGES, config.VECTORIZED_CORPUS, config.VECTORIZER_LIST,
                                   config.CORPUS_TEXT_IDS],
                                  [texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus,
                                   vectorizer, corpus_text_ids]):
            master.clear()
            master.append(update)

        config.TEXTS_LIST_FULL.clear()
        config.TEXTS_LIST_FULL.append(texts_list)
        print("len(config.TEXTS_LIST_FULL[0] :", len(config.TEXTS_LIST_FULL[0]))

        config.TEXTS_LIST_LIST_FULL.clear()
        config.TEXTS_LIST_LIST_FULL.append(texts_list_list)

        config.TOTAL_PAGES_FULL.clear()
        config.TOTAL_PAGES_FULL.append(total_pages)

        config.CLASSIFIER_LIST.clear()
        config.TEXTS_GROUP_1.clear()
        config.TEXTS_GROUP_2.clear()
        config.TEXTS_GROUP_3.clear()
        config.OVERALL_QUALITY_SCORE.clear()
        config.OVERALL_QUALITY_SCORE.append("-")
        return render_template("start.html",
                               dataset_list=config.DATASETS_AVAILABLE,
                               texts_list=config.TEXTS_LIST_LIST[0][0],
                               dataset_name=dataset_name,
                               dataset_labels=config.Y_CLASSES[0])

    else:
        load_status = utils.dataset_list(source_dir="./output/save/")
        print('config.SEARCH_MESSAGE :', config.SEARCH_MESSAGE)
        if load_status == 1:
            return render_template("start.html",
                                   dataset_list=config.DATASETS_AVAILABLE,
                                   texts_list=config.TEXTS_LIST_LIST[0][0],
                                   dataset_name=dataset_name,
                                   dataset_labels=config.Y_CLASSES[0])
        else:
            print("Error loading dataset.")


@app.route("/begin_labeling", methods=["POST"])
def begin_labeling():
    dataset_name = request.args.get("dataset_name", None)
    selected_config = request.form.get("selected_config", None)

    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config.DATE_TIME.clear()
    config.DATE_TIME.append(date_time)

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=config.DATASETS_AVAILABLE,
                               config1_message=config1_message)

    page_number = 0
    utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                           first_labeling_flag=config.FIRST_LABELING_FLAG,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                           label_summary_string=config.LABEL_SUMMARY_STRING)
    # config.SEARCH_MESSAGE.clear()
    # config.SEARCH_MESSAGE.append("Displaying all texts")

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    config.HTML_CONFIG_TEMPLATE.clear()
    config.HTML_CONFIG_TEMPLATE.append(html_config_template)

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=0,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=[],
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=[],
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=[],
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/begin_labeling_new_dataset", methods=["POST"])
def begin_labeling_new_dataset():
    print("In 'begin_labeling_new_dataset()")

    config.SHUFFLE_BY.clear()
    config.SHUFFLE_BY.append("random")  # kmeans random

    texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
        load_new_data(config.DATASET_NAME[0],
                      source_folder="./output/upload/",
                      shuffle_by=config.SHUFFLE_BY[0],
                      table_limit=config.TABLE_LIMIT,
                      texts_limit=config.TEXTS_LIMIT,
                      max_features=config.MAX_FEATURES,
                      y_classes=config.Y_CLASSES[0],
                      rnd_state=config.RND_STATE)

    for master, update in zip([config.TEXTS_LIST, config.TEXTS_LIST_LIST, config.ADJ_TEXT_IDS,
                               config.TOTAL_PAGES, config.VECTORIZED_CORPUS, config.VECTORIZER_LIST,
                               config.CORPUS_TEXT_IDS],
                              [texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus,
                               vectorizer, corpus_text_ids]):
        master.clear()
        master.append(update)

    config.TEXTS_LIST_FULL.clear()
    config.TEXTS_LIST_FULL.append(texts_list)

    config.TEXTS_LIST_LIST_FULL.clear()
    config.TEXTS_LIST_LIST_FULL.append(texts_list_list)

    config.TOTAL_PAGES_FULL.clear()
    config.TOTAL_PAGES_FULL.append(total_pages)

    config.CLASSIFIER_LIST.clear()

    dataset_name = config.DATASET_NAME[0]
    selected_config = request.form.get("selected_config_new_data", None)

    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config.DATE_TIME.clear()
    config.DATE_TIME.append(date_time)

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=config.DATASETS_AVAILABLE,
                               config1_message=config1_message)

    page_number = 0
    utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                           first_labeling_flag=config.FIRST_LABELING_FLAG,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                           label_summary_string=config.LABEL_SUMMARY_STRING)
    config.SEARCH_MESSAGE.clear()
    config.SEARCH_MESSAGE.append("Displaying all texts")

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    config.HTML_CONFIG_TEMPLATE.clear()
    config.HTML_CONFIG_TEMPLATE.append(html_config_template)

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=[],
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=[],
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=[],
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/text_labeling", methods=["GET", "POST"])
def text_labeling():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    app_section = request.args.get("app_section", None)
    selected_text_id = request.args.get("selected_text_id", None)
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    label_selected = request.args.get("label_selected", None)
    page_number = int(request.args.get("page_number", None))
    initialize_flags = request.args.get("initialize_flags", None)
    # print("initialize_flags :", initialize_flags)

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type="text_id_selected",
                                                     click_object="link",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    # Group 1 ************************************************************************************
    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                                                   corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                                                   text_id=selected_text_id,
                                                                   keep_original=config.KEEP_ORIGINAL)

    utils.get_top_similar_texts(all_texts_json=config.TEXTS_LIST_FULL[0],
                                similarities_series=similarities_series,
                                top=config.GROUP_1_KEEP_TOP[0],
                                exclude_already_labeled=config.GROUP_1_EXCLUDE_ALREADY_LABELED,
                                verbose=config.SIMILAR_TEXT_VERBOSE,
                                similar_texts=config.TEXTS_GROUP_1)

    # **********************************************************************************************

    # Group 2 ************************************************************************************

    if len(config.CLASSIFIER_LIST) > 0 and (label_selected is not None and label_selected != "" and label_selected != "None"):
        utils.get_top_predictions(selected_class=label_selected,
                                  fitted_classifier=config.CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                  corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message="Text selected",
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/go_to_page", methods=["GET", "POST"])
def go_to_page():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    app_section = request.args.get("app_section", None)
    selection = request.args.get("selection", None)
    page_number = int(request.args.get("page_number", None))
    label_selected = request.args.get("label_selected", None)

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type=selection,
                                                     click_object="link",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           label_selected=label_selected,
                           info_message="",
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=[],
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/single_text", methods=["POST"])
def single_text():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    new_id = request.form["selected_text_id"]
    new_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == new_id][0]
    page_number = int(request.form["page_number"])
    new_label = request.form["selected_label_single"]
    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="current_text",
                                                     click_type="single_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=new_label)
    utils.add_log_record(value_record, log=config.VALUE_LOG)
    value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=new_id)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    if new_label is None or new_label == "" or new_id == "None":
        if new_label == "":
            flash("Select a 'Label'.", "error")
            info_message += f"Select a 'Label'."

        if new_id == "None":
            info_message += "\n" + f"Select a 'Text ID'."

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])
    else:
        new_obj = {"id": new_id, "text": new_text, "label": new_label}

        utils.update_texts_list_by_id(texts_list=config.TEXTS_LIST_FULL[0],
                                      sub_list_limit=config.TABLE_LIMIT,
                                      updated_obj_lst=[new_obj],
                                      texts_list_list=config.TEXTS_LIST_LIST_FULL[0],
                                      labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG,
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                               label_summary_string=config.LABEL_SUMMARY_STRING)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                             texts_list=config.TEXTS_LIST_FULL[0],
                             texts_list_labeled=[new_obj],
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST,
                             random_state=config.RND_STATE,
                             n_jobs=-1,
                             labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG[0],
                             full_fit_if_labels_got_overridden=config.FULL_FIT_IF_LABELS_GOT_OVERRIDDEN)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (new_label is not None and new_label != "" and new_label != "None"):
            utils.get_top_predictions(selected_class=new_label,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)

        result, difficult_texts_message = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=new_label,
                               info_message="Label assigned",
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=[], # TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/grouped_1_texts", methods=["POST"])
def grouped_1_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    # new_text = request.form["selected_text"]
    selected_label_group1 = request.form["selected_label_group1"]
    info_message = ""

    new_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == new_id][0]

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group1)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    if (selected_label_group1 == "None") or (selected_label_group1 is None) or (selected_label_group1 == "") or \
            (len(config.TEXTS_GROUP_1) == 0):
        if selected_label_group1 == "":
            info_message += f"Select a 'Label'."

        if len(config.TEXTS_GROUP_1) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=[],
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

    else:
        texts_group_1_updated = copy.deepcopy(config.TEXTS_GROUP_1)

        for obj in texts_group_1_updated:
            obj["label"] = selected_label_group1
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            utils.add_log_record(value_record, log=config.VALUE_LOG)

        utils.update_texts_list_by_id(texts_list=config.TEXTS_LIST_FULL[0],
                                      sub_list_limit=config.TABLE_LIMIT,
                                      updated_obj_lst=texts_group_1_updated,
                                      texts_list_list=config.TEXTS_LIST_LIST_FULL[0],
                                      labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG,
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                               label_summary_string=config.LABEL_SUMMARY_STRING)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                             texts_list=config.TEXTS_LIST_FULL[0],
                             texts_list_labeled=texts_group_1_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST,
                             random_state=config.RND_STATE,
                             n_jobs=-1,
                             labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG[0],
                             full_fit_if_labels_got_overridden=config.FULL_FIT_IF_LABELS_GOT_OVERRIDDEN)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (selected_label_group1 is not None and
                 selected_label_group1 != "" and
                 selected_label_group1 != "None"):
            utils.get_top_predictions(selected_class=selected_label_group1,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)

        result, difficult_texts_message = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group1,
                               info_message="Labels assigned to group",
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/grouped_2_texts", methods=["POST"])
def grouped_2_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == new_id][0]
    selected_label_group2 = request.form["selected_label_group2"]

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group2)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    info_message = ""
    if (selected_label_group2 is None) or (selected_label_group2 == "None") or (selected_label_group2 == "") or \
            (len(config.TEXTS_GROUP_2) == 0):
        if selected_label_group2 == "":
            info_message += f"Select a 'Label'."

        if len(config.TEXTS_GROUP_2) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label_group2,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=[],
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

    else:
        texts_group_2_updated = copy.deepcopy(config.TEXTS_GROUP_2)
        for obj in texts_group_2_updated:
            obj["label"] = selected_label_group2
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            utils.add_log_record(value_record, log=config.VALUE_LOG)

        utils.update_texts_list_by_id(texts_list=config.TEXTS_LIST_FULL[0],
                                      sub_list_limit=config.TABLE_LIMIT,
                                      updated_obj_lst=texts_group_2_updated,
                                      texts_list_list=config.TEXTS_LIST_LIST_FULL[0],
                                      labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG,
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                               label_summary_string=config.LABEL_SUMMARY_STRING)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                             texts_list=config.TEXTS_LIST_FULL[0],
                             texts_list_labeled=texts_group_2_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST,
                             random_state=config.RND_STATE,
                             n_jobs=-1,
                             labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG[0],
                             full_fit_if_labels_got_overridden=config.FULL_FIT_IF_LABELS_GOT_OVERRIDDEN)

        if len(selected_label_group2) > 0 and \
                (selected_label_group2 is not None and
                 selected_label_group2 != "" and
                 selected_label_group2 != "None"):
            utils.get_top_predictions(selected_class=selected_label_group2,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)

        result, difficult_texts_message = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group2,
                               info_message="Labels assigned to group",
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/label_all", methods=["POST"])
def label_all():
    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    # new_text = request.form["selected_text"]
    selected_label = request.form["selected_label"]

    new_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == new_id][0]

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    if config.HTML_CONFIG_TEMPLATE[0] in ["text_labeling_2.html"]:
        scroll_to_id = "labelAllButton"
    else:
        scroll_to_id = None

    if config.NUMBER_UNLABELED_TEXTS[0] == 0:
        info_message = "There are no more unlabeled texts. If you are unhappy with the quality of the " \
                       "auto-labeling, then work through the 'Difficult Texts' to improve the quality."
        label_summary_message = info_message
        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=[],
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               scroll_to_id=scroll_to_id,
                               label_summary_message=label_summary_message,
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

    if len(config.CLASSIFIER_LIST) > 0:
        if config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] == 0:
            info_message = \
                f"Are you sure that you want to auto-label the " \
                f"remaining {config.NUMBER_UNLABELED_TEXTS[0]:,} texts?"
            label_summary_message = info_message

            temp_count = config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0]
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(temp_count + 1)

            return render_template(config.HTML_CONFIG_TEMPLATE[0],
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=config.SEARCH_MESSAGE[0],
                                   search_results_length=config.SEARCH_RESULT_LENGTH[0],
                                   page_number=page_number,
                                   y_classes=config.Y_CLASSES[0],
                                   total_pages=config.TOTAL_PAGES[0],
                                   texts_list=config.TEXTS_LIST_LIST[0][page_number],
                                   texts_group_1=config.TEXTS_GROUP_1,
                                   group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                                   texts_group_2=[],
                                   group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                                   texts_group_3=config.TEXTS_GROUP_3,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                                   label_summary_string=config.LABEL_SUMMARY_STRING[0],
                                   initialize_flags=config.INITIALIZE_FLAGS[0],
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

        elif config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] == 1:
            info_message = \
                f"Click 'Label All Texts' again to confirm your choice. The remaining unlabeled texts will be " \
                f"labeled using machine-learning. Inspect the 'Difficult Texts' to see how well the " \
                f"machine-learning model is trained and if you are satisfied then proceed."
            label_summary_message = info_message

            temp_count = config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0]
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(temp_count + 1)

            return render_template(config.HTML_CONFIG_TEMPLATE[0],
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=config.SEARCH_MESSAGE[0],
                                   search_results_length=config.SEARCH_RESULT_LENGTH[0],
                                   page_number=page_number,
                                   y_classes=config.Y_CLASSES[0],
                                   total_pages=config.TOTAL_PAGES[0],
                                   texts_list=config.TEXTS_LIST_LIST[0][page_number],
                                   texts_group_1=config.TEXTS_GROUP_1,
                                   group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                                   texts_group_2=[],
                                   group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                                   texts_group_3=config.TEXTS_GROUP_3,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                                   label_summary_string=config.LABEL_SUMMARY_STRING[0],
                                   initialize_flags=config.INITIALIZE_FLAGS[0],
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

        elif config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] > 1:
            info_message = \
                f"{config.NUMBER_UNLABELED_TEXTS[0]} texts were auto-labeled."

            updated_texts_list, updated_texts_list_list, labeled_text_ids = \
                utils.label_all(fitted_classifier=config.CLASSIFIER_LIST[0],
                                sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                texts_list=config.TEXTS_LIST_FULL[0],
                                label_only_unlabeled=True,
                                sub_list_limit=config.TABLE_LIMIT,
                                update_in_place=False,
                                texts_list_list=config.TEXTS_LIST_LIST_FULL[0])

            for labeled_text_id in labeled_text_ids:
                value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=labeled_text_id)
                utils.add_log_record(value_record, log=config.VALUE_LOG)

            utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                                   first_labeling_flag=config.FIRST_LABELING_FLAG,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                                   label_summary_string=config.LABEL_SUMMARY_STRING)

            label_summary_message = info_message
            return render_template(config.HTML_CONFIG_TEMPLATE[0],
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   search_message=config.SEARCH_MESSAGE[0],
                                   search_results_length=config.SEARCH_RESULT_LENGTH[0],
                                   page_number=page_number,
                                   y_classes=config.Y_CLASSES[0],
                                   total_pages=config.TOTAL_PAGES[0],
                                   texts_list=config.TEXTS_LIST_LIST[0][page_number],
                                   texts_group_1=[], # texts_group_1_updated,
                                   group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                                   texts_group_2=config.TEXTS_GROUP_2,
                                   group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                                   texts_group_3=config.TEXTS_GROUP_3,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                                   label_summary_string=config.LABEL_SUMMARY_STRING[0],
                                   initialize_flags=config.INITIALIZE_FLAGS[0],
                                   scroll_to_id=scroll_to_id,
                                   label_summary_message=label_summary_message,
                                   search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])
    # **********************************************************************************************
    else:
        info_message = "Label more texts before trying again."
    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=new_id,
                           selected_text=new_text,
                           label_selected=selected_label,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=[],
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           scroll_to_id=scroll_to_id,
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route('/export_records', methods=['GET', 'POST'])
def export_records():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="export_records",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    download_files = []

    print("After download_files")

    texts_df = pd.DataFrame.from_dict(config.TEXTS_LIST_FULL[0])
    texts_df.to_csv("./output/labeled_texts.csv", index=False)
    download_files.append("./output/labeled_texts.csv")

    print("After config.TEXTS_LIST_FULL")

    click_log_df = pd.DataFrame.from_dict(config.CLICK_LOG)
    click_log_df.to_csv("./output/click_log.csv", index=False)
    download_files.append("./output/click_log.csv")

    value_log_df = pd.DataFrame.from_dict(config.VALUE_LOG)
    value_log_df.to_csv("./output/value_log.csv", index=False)
    download_files.append("./output/value_log.csv")

    print("After config.VALUE_LOG")

    if len(config.CLASSIFIER_LIST) > 0:
        filename = "trained-classifier.pkl"
        pickle.dump(config.CLASSIFIER_LIST[0], open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

        filename = "fitted-vectorizer.pkl"
        pickle.dump(config.VECTORIZER_LIST[0], open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

    if len(config.TOTAL_SUMMARY) > 0:
        filename = "total-summary.csv"
        total_summary_df = pd.DataFrame.from_dict(config.TOTAL_SUMMARY)
        total_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    if len(config.LABEL_SUMMARY) > 0:
        filename = "label-summary.csv"
        label_summary_df = pd.DataFrame.from_dict(config.LABEL_SUMMARY)
        label_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    filename = "rapid-labeling-results-" + config.DATE_TIME[0] + ".zip"
    file_path = "./output/download/rapid-labeling-results-" + config.DATE_TIME[0] + ".zip"
    zip_file = zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED)
    for file in download_files:
        zip_file.write(file)
    zip_file.close()
    return send_file(file_path, mimetype='zip', download_name=filename, as_attachment=True)


@app.route('/save_state', methods=['GET', 'POST'])
def save_state():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="save_state",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    save_state = {}
    save_state["DATASET_NAME"] = config.DATASET_NAME
    save_state["DATASET_URL"] = config.DATASET_URL
    save_state["TOTAL_SUMMARY"] = config.TOTAL_SUMMARY
    save_state["LABEL_SUMMARY"] = config.LABEL_SUMMARY
    save_state["RECOMMENDATIONS_SUMMARY"] = config.RECOMMENDATIONS_SUMMARY
    save_state["TEXTS_LIST_LABELED"] = config.TEXTS_LIST_LABELED
    save_state["TEXTS_GROUP_1"] = config.TEXTS_GROUP_1
    save_state["TEXTS_GROUP_2"] = config.TEXTS_GROUP_2
    save_state["TEXTS_GROUP_3"] = config.TEXTS_GROUP_3
    save_state["SEARCH_MESSAGE"] = config.SEARCH_MESSAGE

    save_state["TEXTS_LIST"] = config.TEXTS_LIST
    save_state["TEXTS_LIST_FULL"] = config.TEXTS_LIST_FULL

    save_state["CORPUS_TEXT_IDS"] = config.CORPUS_TEXT_IDS
    print("len(config.TEXTS_LIST_LIST_FULL[0]) :", len(config.TEXTS_LIST_LIST_FULL[0]))
    save_state["TEXTS_LIST_LIST"] = config.TEXTS_LIST_LIST
    save_state["TEXTS_LIST_LIST_FULL"] = config.TEXTS_LIST_LIST_FULL

    save_state["TOTAL_PAGES_FULL"] = config.TOTAL_PAGES_FULL
    save_state["ADJ_TEXT_IDS"] = config.ADJ_TEXT_IDS
    save_state["TOTAL_PAGES"] = config.TOTAL_PAGES
    save_state["Y_CLASSES"] = config.Y_CLASSES
    save_state["SHUFFLE_BY"] = config.SHUFFLE_BY
    save_state["NUMBER_UNLABELED_TEXTS"] = config.NUMBER_UNLABELED_TEXTS
    save_state["HTML_CONFIG_TEMPLATE"] = config.HTML_CONFIG_TEMPLATE
    save_state["LABEL_SUMMARY_STRING"] = config.LABEL_SUMMARY_STRING

    with open("./output/save/save_state.json", "w") as outfile:
        json.dump(save_state, outfile)

    filenames = ["DATE_TIME", "DATASET_NAME", "CLICK_LOG", "VALUE_LOG", "VECTORIZED_CORPUS", "Y_CLASSES",
                 "CLASSIFIER_LIST", "VECTORIZER_LIST", "TOTAL_SUMMARY", "DATASET_URL"]
    for filename in filenames:
        temp_filename = filename + ".pkl"
        pickle.dump(eval("config." + filename), open("./output/save/" + temp_filename, "wb"))

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=0,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][0],
                           texts_group_1=[],
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=[],
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=[],
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route('/update_panels', methods=['GET', 'POST'])
def update_panels():
    panel_flag = eval(request.args.get("panel_flag", None))
    utils.update_panel_flags(update_flag=panel_flag, panel_flags=config.INITIALIZE_FLAGS[0])

    return panel_flag


@app.route('/label_selected', methods=['GET', 'POST'])
def label_selected():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    label_selected = request.args.get("label_selected")
    selected_text_id = request.args.get("selected_text_id")
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    page_number = int(request.args.get("page_number"))

    click_record, guid = utils.generate_click_record(click_location="label",
                                                     click_type="label_selected",
                                                     click_object="radio_button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=label_selected)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        config.TEXTS_GROUP_1.clear()
    else:
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                                                       corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                                                       text_id=selected_text_id,
                                                                       keep_original=config.KEEP_ORIGINAL)

        utils.get_top_similar_texts(all_texts_json=config.TEXTS_LIST_FULL[0],
                                    similarities_series=similarities_series,
                                    top=config.GROUP_1_KEEP_TOP[0],
                                    exclude_already_labeled=config.GROUP_1_EXCLUDE_ALREADY_LABELED,
                                    verbose=config.SIMILAR_TEXT_VERBOSE,
                                    similar_texts=config.TEXTS_GROUP_1)
    # **********************************************************************************************

    # Group 2 ************************************************************************************
    if len(config.CLASSIFIER_LIST) > 0 and label_selected != "None":
        utils.get_top_predictions(selected_class=label_selected,
                                  fitted_classifier=config.CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                  corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    info_message = f"'{label_selected}' selected"

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route('/generate_difficult_texts', methods=['POST'])
def generate_difficult_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    id = request.form["selected_text_id"]
    label_selected = request.form["selected_label"]

    text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == id][0]

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="generate_list",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    # Group 3 ************************************************************************************

    result, message = generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                              full_fit_if_labels_got_overridden=True, round_to=1,
                                                              format_as_percentage=True)
    info_message = message

    # **********************************************************************************************


    if config.HTML_CONFIG_TEMPLATE[0] in ["text_labeling_2.html"]:
        scroll_to_id = "labelAllButton"
        difficult_texts_message = info_message
    else:
        scroll_to_id = None
        difficult_texts_message = None

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=id,
                           selected_text=text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           scroll_to_id=scroll_to_id,
                           difficult_texts_message=difficult_texts_message,
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/set_group_1_record_limit", methods=["POST"])
def set_group_1_record_limit():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    label_selected = request.form.get("label_selected", None)
    table_limit = int(request.form.get("group1_table_limit", None))

    info_message = f"Set 'Similar Texts' display limit to {table_limit}"
    config.GROUP_1_KEEP_TOP.clear()
    config.GROUP_1_KEEP_TOP.append(table_limit)

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=table_limit)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        config.TEXTS_GROUP_1.clear()
    else:
        start_test_time = datetime.now()

        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                                                       corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                                                       text_id=selected_text_id,
                                                                       keep_original=config.KEEP_ORIGINAL)

        utils.get_top_similar_texts(all_texts_json=config.TEXTS_LIST_FULL[0],
                                    similarities_series=similarities_series,
                                    top=config.GROUP_1_KEEP_TOP[0],
                                    exclude_already_labeled=config.GROUP_1_EXCLUDE_ALREADY_LABELED,
                                    verbose=config.SIMILAR_TEXT_VERBOSE,
                                    similar_texts=config.TEXTS_GROUP_1)

        end_test_time = datetime.now()
        duration = end_test_time - start_test_time
    # **********************************************************************************************

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/set_group_2_record_limit", methods=["POST"])
def set_group_2_record_limit():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    label_selected = request.form.get("label_selected", None)
    table_limit = int(request.form.get("group2_table_limit", None))

    info_message = f"Set 'Similar Texts' display limit to {table_limit}"
    config.PREDICTIONS_NUMBER.clear()
    config.PREDICTIONS_NUMBER.append(table_limit)

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=table_limit)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    # Group 2 ************************************************************************************
    if len(config.CLASSIFIER_LIST) > 0:
        utils.get_top_predictions(selected_class=label_selected,
                                  fitted_classifier=config.CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                  corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=page_number,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][page_number],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/search_all_texts", methods=["POST"])
def search_all_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    label_selected = request.form.get("label_selected", None)
    include_search_term = request.form.get("include_search_term", None)
    exclude_search_term = request.form.get("exclude_search_term", None)
    allow_search_override_labels = request.form.get('searchAllTextRadio', None)
    print("allow_search_override_labels :", allow_search_override_labels)

    if allow_search_override_labels == "Yes":
        config.SEARCH_EXCLUDE_ALREADY_LABELED.clear()
        config.SEARCH_EXCLUDE_ALREADY_LABELED.append(False)
    else:
        config.SEARCH_EXCLUDE_ALREADY_LABELED.clear()
        config.SEARCH_EXCLUDE_ALREADY_LABELED.append(True)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="search_texts",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)
    value_record_1 = utils.generate_value_record(guid=guid, value_type="include_search_term", value=include_search_term)
    utils.add_log_record(value_record_1, log=config.VALUE_LOG)
    value_record_2 = utils.generate_value_record(guid=guid, value_type="exclude_search_term", value=exclude_search_term)
    utils.add_log_record(value_record_2, log=config.VALUE_LOG)

    if len(include_search_term) > 0 or len(exclude_search_term) > 0:
        search_results = utils.search_all_texts(all_text=config.TEXTS_LIST_FULL[0],
                                                include_search_term=include_search_term,
                                                exclude_search_term=exclude_search_term,
                                                search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0],
                                                include_behavior="conjunction",
                                                exclude_behavior="conjunction",
                                                all_upper=True)
        search_results_length = len(search_results)

        if search_results_length > 0:
            config.SEARCH_RESULT_LENGTH.clear()
            config.SEARCH_RESULT_LENGTH.append(search_results_length)

            config.TEXTS_LIST.clear()
            config.TEXTS_LIST.append(search_results)

            search_results_list = \
                [search_results[i:i + config.TABLE_LIMIT] for i in range(0, len(search_results), config.TABLE_LIMIT)]
            search_results_total_pages = len(search_results_list)
            search_result_page_number = 0

            config.TEXTS_LIST_LIST.clear()
            config.TEXTS_LIST_LIST.append(search_results_list)

            config.TOTAL_PAGES.clear()
            config.TOTAL_PAGES.append(search_results_total_pages)

            info_message = f"Showing results Include-'{include_search_term}', Exclude-'{exclude_search_term}'"
            config.SEARCH_MESSAGE.clear()
            config.SEARCH_MESSAGE.append(info_message)

            return render_template(config.HTML_CONFIG_TEMPLATE[0],
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   search_message=config.SEARCH_MESSAGE[0],
                                   search_results_length=config.SEARCH_RESULT_LENGTH[0],
                                   page_number=search_result_page_number,
                                   y_classes=config.Y_CLASSES[0],
                                   total_pages=config.TOTAL_PAGES[0],
                                   texts_list=config.TEXTS_LIST_LIST[0][search_result_page_number],
                                   texts_group_1=config.TEXTS_GROUP_1,
                                   group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                                   texts_group_2=config.TEXTS_GROUP_2,
                                   group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                                   texts_group_3=config.TEXTS_GROUP_3,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                                   label_summary_string=config.LABEL_SUMMARY_STRING[0],
                                   initialize_flags=config.INITIALIZE_FLAGS[0],
                                   search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])
        else:
            info_message = f"No search results for Include-'{include_search_term}', Exclude-'{exclude_search_term}'"
            config.SEARCH_MESSAGE.clear()
            config.SEARCH_MESSAGE.append(info_message)

            config.SEARCH_RESULT_LENGTH.clear()
            config.SEARCH_RESULT_LENGTH.append(0)

            config.TEXTS_LIST_LIST.clear()
            config.TEXTS_LIST_LIST.append([[]])

            config.TOTAL_PAGES.clear()
            config.TOTAL_PAGES.append(0)

            page_number = -1

            return render_template(config.HTML_CONFIG_TEMPLATE[0],
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   search_message=config.SEARCH_MESSAGE[0],
                                   search_results_length=config.SEARCH_RESULT_LENGTH[0],
                                   page_number=page_number,
                                   y_classes=config.Y_CLASSES[0],
                                   total_pages=config.TOTAL_PAGES[0],
                                   texts_list=config.TEXTS_LIST_LIST[0][0],
                                   texts_group_1=config.TEXTS_GROUP_1,
                                   group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                                   texts_group_2=config.TEXTS_GROUP_2,
                                   group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                                   texts_group_3=config.TEXTS_GROUP_3,
                                   total_summary=config.TOTAL_SUMMARY,
                                   label_summary=config.LABEL_SUMMARY,
                                   recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                                   label_summary_string=config.LABEL_SUMMARY_STRING[0],
                                   initialize_flags=config.INITIALIZE_FLAGS[0],
                                   search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])
    else:
        info_message = "No search term entered"
        config.SEARCH_MESSAGE.clear()
        config.SEARCH_MESSAGE.append(info_message)

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=selected_text_id,
                               selected_text=selected_text,
                               label_selected=label_selected,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=0,
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/grouped_search_texts", methods=["POST"])
def grouped_search_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == new_id][0]
    # search_results_length = request.form["search_results_length"]
    selected_label_search_texts = request.form["selected_label_search_texts"]
    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_search_texts)
    utils.add_log_record(value_record, log=config.VALUE_LOG)

    if (selected_label_search_texts == "None") or (selected_label_search_texts is None) \
            or (selected_label_search_texts == ""):
        if selected_label_search_texts == "":
            info_message += f"Select a 'Label'."

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=config.TEXTS_GROUP_1,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=[],
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])

    else:
        texts_group_updated = copy.deepcopy(config.TEXTS_LIST[0])

        for obj in texts_group_updated:
            obj["label"] = selected_label_search_texts
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            utils.add_log_record(value_record, log=config.VALUE_LOG)

        utils.update_texts_list_by_id(texts_list=config.TEXTS_LIST_FULL[0],
                                      sub_list_limit=config.TABLE_LIMIT,
                                      updated_obj_lst=texts_group_updated,
                                      texts_list_list=config.TEXTS_LIST_LIST_FULL[0],
                                      labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG,
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS,
                               label_summary_string=config.LABEL_SUMMARY_STRING)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                             texts_list=config.TEXTS_LIST_FULL[0],
                             texts_list_labeled=texts_group_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST,
                             random_state=config.RND_STATE,
                             n_jobs=-1,
                             labels_got_overridden_flag=config.LABELS_GOT_OVERRIDDEN_FLAG[0],
                             full_fit_if_labels_got_overridden=config.FULL_FIT_IF_LABELS_GOT_OVERRIDDEN)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (selected_label_search_texts is not None and
                 selected_label_search_texts != "" and
                 selected_label_search_texts != "None"):
            utils.get_top_predictions(selected_class=selected_label_search_texts,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=config.CORPUS_TEXT_IDS[0],
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)

        result, difficult_texts_message = \
            generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                    full_fit_if_labels_got_overridden=True, round_to=1,
                                                    format_as_percentage=True)
        # **********************************************************************************************

        return render_template(config.HTML_CONFIG_TEMPLATE[0],
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_search_texts,
                               info_message="Labels assigned to group",
                               search_message=config.SEARCH_MESSAGE[0],
                               search_results_length=config.SEARCH_RESULT_LENGTH[0],
                               page_number=page_number,
                               y_classes=config.Y_CLASSES[0],
                               total_pages=config.TOTAL_PAGES[0],
                               texts_list=config.TEXTS_LIST_LIST[0][page_number],
                               texts_group_1=[], # texts_group_1_updated,
                               group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                               texts_group_2=config.TEXTS_GROUP_2,
                               group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                               texts_group_3=config.TEXTS_GROUP_3,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                               label_summary_string=config.LABEL_SUMMARY_STRING[0],
                               initialize_flags=config.INITIALIZE_FLAGS[0],
                               difficult_texts_message=difficult_texts_message,
                               search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


@app.route("/clear_search_all_texts", methods=["POST"])
def clear_search_all_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = [text["text"] for text in config.TEXTS_LIST_FULL[0] if text["id"] == selected_text_id][0]
    label_selected = request.form.get("label_selected", None)
    info_message = "Search cleared"

    config.SEARCH_MESSAGE.clear()
    config.SEARCH_MESSAGE.append("Displaying all texts")

    config.TEXTS_LIST.clear()
    config.TEXTS_LIST.append(config.TEXTS_LIST_FULL[0])

    config.TEXTS_LIST_LIST.clear()
    config.TEXTS_LIST_LIST.append(config.TEXTS_LIST_LIST_FULL[0])

    config.TOTAL_PAGES.clear()
    config.TOTAL_PAGES.append(config.TOTAL_PAGES_FULL[0])

    config.SEARCH_RESULT_LENGTH.clear()
    config.SEARCH_RESULT_LENGTH.append(0)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="clear_search",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    return render_template(config.HTML_CONFIG_TEMPLATE[0],
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           search_message=config.SEARCH_MESSAGE[0],
                           search_results_length=config.SEARCH_RESULT_LENGTH[0],
                           page_number=0,
                           y_classes=config.Y_CLASSES[0],
                           total_pages=config.TOTAL_PAGES[0],
                           texts_list=config.TEXTS_LIST_LIST[0][0],
                           texts_group_1=config.TEXTS_GROUP_1,
                           group1_table_limit_value=config.GROUP_1_KEEP_TOP[0],
                           texts_group_2=config.TEXTS_GROUP_2,
                           group2_table_limit_value=config.PREDICTIONS_NUMBER[0],
                           texts_group_3=config.TEXTS_GROUP_3,
                           total_summary=config.TOTAL_SUMMARY,
                           label_summary=config.LABEL_SUMMARY,
                           recommendations_summary=config.RECOMMENDATIONS_SUMMARY,
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0],
                           label_summary_string=config.LABEL_SUMMARY_STRING[0],
                           initialize_flags=config.INITIALIZE_FLAGS[0],
                           search_exclude_already_labeled=config.SEARCH_EXCLUDE_ALREADY_LABELED[0])


if __name__ == "__main__":
    app.run() # host="127.0.0.1:5000"

