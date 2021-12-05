from flask import Flask, render_template, request, jsonify, flash, make_response, redirect, g, send_file, session
from flask_session import Session
import web_app_utilities as utils
import configuration as config
import copy
import pandas as pd
from datetime import datetime
import zipfile
import pickle
import json
from werkzeug.utils import secure_filename
import os



start_time = datetime.now()
print("*"*20, "Process started @", start_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)

app = Flask(__name__)
# Session(app)
# app.secret_key = "super secret key"
app.config.from_object(__name__)
app.config["UPLOAD_FOLDER"] = "./output/upload/"
# app.config["MAX_CONTENT_PATH"] = 10000

app.jinja_env.add_extension('jinja2.ext.do')


end_time = datetime.now()
print("*"*20, "Process ended @", end_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)
duration = end_time - start_time
print("*"*20, "Process duration -", duration, "*"*20)


@app.route('/label_entered', methods=['POST'])
def label_entered():
    if request.form["action"] == "add":
        label_entered = request.form["labelEntered"]
        print("label_entered :", label_entered)

        utils.add_y_classes([label_entered], begin_fresh=False)
        y_classes_sql = utils.get_y_classes()
        entered_labels = ", ".join(y_classes_sql)
    else:
        utils.add_y_classes([], begin_fresh=True)
        entered_labels = ""

    prep_data_message1_sql = utils.get_variable_value(name="PREP_DATA_MESSAGE1")
    if len(prep_data_message1_sql) > 0:
        prep_data_message1 = prep_data_message1_sql
    else:
        prep_data_message1 = None

    records_limit = 100

    prep_data_message2_sql = utils.get_variable_value(name="PREP_DATA_MESSAGE2")
    prepared_data_list_sql = utils.get_text_list(table_name="prepData")
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

        utils.set_variable(name="TEXT_ID_COL", value=text_id_col)
        utils.set_variable(name="TEXT_VALUE_COL", value=text_value_col)

        datasets_available_sql = utils.get_variable_value(name="DATASETS_AVAILABLE")

        utils.clear_y_classes()

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
                utils.set_variable(name="PREP_DATA_MESSAGE1", value=prep_data_message1)


                prepared_data_list = dataset_df.to_dict("records")
                utils.populate_texts_table_sql(texts_list=prepared_data_list, table_name="prepTexts")
                records_limit = 100
                prepared_texts_sql = utils.get_text_list(table_name="prepTexts")
                total_records = len(prepared_texts_sql)

                date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

                prep_data_message2 = "Showing {} of {} records of the prepared data".format(records_limit, total_records)
                utils.set_variable(name="PREP_DATA_MESSAGE2", value=prep_data_message2)
                utils.set_variable(name="DATASET_NAME", value=filename)
                utils.set_variable(name="DATASET_URL", value="-")
                utils.set_variable(name="DATE_TIME", value=date_time)

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
    available_datasets_sql = utils.get_available_datasets()
    return render_template("start.html", dataset_list=available_datasets_sql)


@app.route("/")
def index():
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    print("search_exclude_already_labeled_sql :", search_exclude_already_labeled_sql)
    available_datasets_sql = utils.get_available_datasets()
    return render_template("start.html", dataset_list=available_datasets_sql)


@app.route("/dataset_selected", methods=["GET", "POST"])
def dataset_selected():
    dataset_name = request.args.get("dataset_name", None)
    utils.set_variable(name="DATASET_NAME", value=dataset_name)
    dataset_name_sql = utils.get_variable_value(name="DATASET_NAME")

    dataset_url = request.args.get("dataset_url", None)
    utils.set_variable(name="DATASET_URL", value=dataset_url)

    conn = utils.get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")

    if dataset_name in ["Disaster Tweets Dataset", "Disaster Tweets Dataset with 'Other'"]:
        if dataset_name == "Disaster Tweets Dataset with 'Other'":
            utils.add_y_classes(y_classses_list=["Earthquake", "Fire", "Flood", "Hurricane", "Other"], begin_fresh=True)
        else:
            utils.add_y_classes(y_classses_list=["Earthquake", "Fire", "Flood", "Hurricane"], begin_fresh=True)

        y_classes_sql = utils.get_y_classes()
        utils.set_variable(name="SHUFFLE_BY", value="random")
        shuffle_by_sql = utils.get_variable_value(name="SHUFFLE_BY")
        texts_limit_sql = utils.get_variable_value(name="TEXTS_LIMIT")
        max_features_sql = utils.get_variable_value(name="MAX_FEATURES")
        random_state_sql = utils.get_variable_value(name="RND_STATE")

        texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
            utils.load_demo_data_sql(dataset_name="Disaster Tweets Dataset", shuffle_by=shuffle_by_sql,
                                     table_limit=table_limit_sql, texts_limit=texts_limit_sql,
                                     max_features=max_features_sql,
                                     y_classes=y_classes_sql, rnd_state=random_state_sql)

        utils.reset_log_click_record_sql()
        utils.reset_log_click_value_sql()

        utils.populate_texts_table_sql(texts_list=texts_list)

        text_list_full_sql = utils.get_text_list(table_name="texts")
        text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)

        utils.set_variable(name="TOTAL_PAGES", value=total_pages)
        utils.set_variable(name="SEARCH_MESSAGE", value="")

        utils.set_pkl(name="CLASSIFIER", pkl_data=None, reset=True)
        utils.set_texts_group_x(None, table_name="group1Texts")
        utils.set_texts_group_x(None, table_name="group2Texts")
        utils.reset_difficult_texts_sql()

        utils.set_variable(name="OVERALL_QUALITY_SCORE", value="-")

        return render_template("start.html",
                               dataset_list=available_datasets,
                               texts_list=text_list_list_sql[0],
                               dataset_name=dataset_name_sql,
                               dataset_labels=y_classes_sql)

    else:
        load_status = utils.load_save_state_sql(source_dir="./output/save/")
        if load_status == 1:
            y_classes_sql = utils.get_y_classes()
            text_list_full_sql = utils.get_text_list(table_name="texts")
            text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                             sub_list_limit=table_limit_sql)
            dataset_name_sql = utils.get_variable_value(name="DATASET_NAME")
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
    utils.set_variable(name="DATE_TIME", value=date_time)

    conn = utils.get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=available_datasets,
                               config1_message=config1_message)

    page_number = 0

    text_list_full_sql = utils.get_text_list(table_name="texts")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                     sub_list_limit=table_limit_sql)
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
        utils.generate_summary_sql(text_lists=text_list_full_sql)

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    utils.set_variable(name="HTML_CONFIG_TEMPLATE", value=html_config_template)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")

    y_classes_sql = utils.get_y_classes()
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    utils.set_variable(name="SEARCH_RESULTS_LENGTH", value=0)
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")

    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           info_message="No label selected",
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="SHUFFLE_BY", value="random")
    shuffle_by_sql = utils.get_variable_value(name="SHUFFLE_BY")
    dataset_name_sql = utils.get_variable_value(name="DATASET_NAME")
    y_classes_sql = utils.get_y_classes()
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    texts_limit_sql = utils.get_variable_value(name="TEXTS_LIMIT")
    max_features_sql = utils.get_variable_value(name="MAX_FEATURES")
    random_state_sql = utils.get_variable_value(name="RND_STATE")
    text_id_col_sql = utils.get_pkl(name="TEXT_ID_COL")
    text_value_col_sql = utils.get_variable_value(name="TEXT_VALUE_COL")

    conn = utils.get_db_connection()
    available_datasets = conn.execute('SELECT * FROM availableDatasets').fetchall()
    conn.close()

    texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer, corpus_text_ids = \
        utils.load_new_data(dataset_name_sql,
                            text_id_col=text_id_col_sql,
                            text_value_col=text_value_col_sql,
                            source_folder="./output/upload/",
                            shuffle_by=shuffle_by_sql,
                            table_limit=table_limit_sql,
                            texts_limit=texts_limit_sql,
                            max_features=max_features_sql,
                            y_classes=y_classes_sql,
                            rnd_state=random_state_sql)

    conn = utils.get_db_connection()
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

    utils.set_variable(name="TOTAL_PAGES", value=total_pages)
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")

    utils.set_pkl(name="CLASSIFIER", pkl_data=None, reset=True)

    dataset_name = utils.get_variable_value(name="DATASET_NAME")
    selected_config = request.form.get("selected_config_new_data", None)

    date_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    utils.set_variable(name="DATE_TIME", value=date_time)

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=available_datasets,
                               config1_message=config1_message)

    page_number = 0

    text_list_full_sql = utils.get_text_list(table_name="texts")
    text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
        utils.generate_summary_sql(text_lists=text_list_full_sql)

    utils.set_variable(name="SEARCH_MESSAGE", value="Displaying all texts")

    if selected_config == "config1":
        html_config_template = "text_labeling_1.html"
    elif selected_config == "config2":
        html_config_template = "text_labeling_2.html"
    else:
        html_config_template = "text_labeling_1.html"

    utils.set_variable(name="HTML_CONFIG_TEMPLATE", value=html_config_template)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    app_section = request.args.get("app_section", None)
    selected_text_id = request.args.get("selected_text_id", None)

    y_classes_sql = utils.get_y_classes()

    text_list_full_sql = utils.get_text_list(table_name="texts")
    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql, 
                                             table_limit_sql=table_limit_sql)

    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    keep_original_sql = utils.get_variable_value(name="KEEP_ORIGINAL")
    corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    group_1_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    similar_text_verbose_sql = utils.get_variable_value(name="SIMILAR_TEXT_VERBOSE")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")

    label_selected = request.args.get("label_selected", None)
    page_number = int(request.args.get("page_number", None))

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type="text_id_selected",
                                                     click_object="link",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    # Group 1 ************************************************************************************
    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                   corpus_text_ids=corpus_text_ids_sql,
                                                                   text_id=selected_text_id,
                                                                   keep_original=keep_original_sql)

    text_list_full_sql = utils.get_text_list(table_name="texts")

    texts_group_1_sql = utils.get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                        similarities_series=similarities_series,
                                                        top=group_1_keep_top_sql,
                                                        exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                        verbose=similar_text_verbose_sql)

    # **********************************************************************************************

    # Group 2 ************************************************************************************
    classifier_sql = utils.get_pkl(name="CLASSIFIER_LIST")
    if classifier_sql and (label_selected is not None and label_selected != "" and label_selected != "None"):
        texts_group_2_sql = \
            utils.get_top_predictions_sql(selected_class=label_selected,
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
        texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags = utils.get_panel_flags()
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    texts_group_3_sql = utils.get_difficult_texts_sql()

    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message="Text selected",
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    app_section = request.args.get("app_section", None)
    selection = request.args.get("selection", None)
    page_number = int(request.args.get("page_number", None))
    label_selected = request.args.get("label_selected", None)
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    y_classes_sql = utils.get_y_classes()
    text_list_full_sql = utils.get_text_list(table_name="texts")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                     sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type=selection,
                                                     click_object="link",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    return render_template(html_config_template_sql,
                           selected_text_id="None",
                           selected_text="Select a text to begin labeling.",
                           label_selected=label_selected,
                           info_message="",
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    new_id = request.form["selected_text_id"]

    y_classes_sql = utils.get_y_classes()
    text_list_full_sql = utils.get_text_list(table_name="texts")
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    new_text = utils.get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                       search_results_length_sql=search_results_length_sql,
                                       table_limit_sql=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    random_state_sql = utils.get_variable_value(name="RND_STATE")
    labels_got_overridden_sql = utils.get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
    full_fit_if_labels_got_overridden_sql = utils.get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    initialize_flags_sql = utils.get_panel_flags()
    update_in_place_sql = utils.get_variable_value(name="UPDATE_TEXTS_IN_PLACE")

    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")

    page_number = int(request.form["page_number"])
    new_label = request.form["selected_label_single"]
    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="current_text",
                                                     click_type="single_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=new_label)
    utils.add_log_click_value_sql(records=[value_record])
    value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=new_id)
    utils.add_log_click_value_sql(records=[value_record])

    if new_label is None or new_label == "" or new_id == "None":
        if new_label == "":
            flash("Select a 'Label'.", "error")
            info_message += f"Select a 'Label'."

        if new_id == "None":
            info_message += "\n" + f"Select a 'Text ID'."

        text_list_list = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message,
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
        corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
        vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
        fit_classifier_verbose_sql = utils.get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
        text_list_full_sql, texts_list_list_sql = \
            utils.update_texts_list_by_id_sql(selected_label=new_label, update_ids=[new_id],
                                              sub_list_limit=table_limit_sql, labels_got_overridden_flag=[],
                                              update_in_place=update_in_place_sql)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            utils.generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        classifier_sql = utils.fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
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
                utils.get_top_predictions_sql(selected_class=new_label,
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
            texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            utils.generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                          full_fit_if_labels_got_overridden=True, round_to=1,
                                                          format_as_percentage=True)

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=new_label,
                               info_message="Label assigned",
                               page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    # new_text = request.form["selected_text"]
    selected_label_group1 = request.form["selected_label_group1"]
    info_message = ""

    y_classes_sql = utils.get_y_classes()
    text_list_full_sql = utils.get_text_list(table_name="texts")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql,
                                             table_limit_sql=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")

    new_text = utils.get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    random_state_sql = utils.get_variable_value(name="RND_STATE")
    fit_classifier_verbose_sql = utils.get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
    full_fit_if_labels_got_overridden_sql = utils.get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
    labels_got_overridden_flag_sql = utils.get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
    initialize_flags_sql = utils.get_panel_flags()
    update_in_place_sql = utils.get_variable_value(name="UPDATE_TEXTS_IN_PLACE")

    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    corpus_text_ids = utils.get_pkl(name="CORPUS_TEXT_IDS")

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group1)
    utils.add_log_click_value_sql(records=[value_record])

    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    if (selected_label_group1 == "None") or (selected_label_group1 is None) or (selected_label_group1 == "") or \
            (len(texts_group_1_sql) == 0):
        if selected_label_group1 == "":
            info_message += f"Select a 'Label'."

        if len(texts_group_1_sql) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        total_summary_sql = utils.get_total_summary_sql()
        label_summary_sql = utils.get_label_summary_sql()
        label_summary_string_sql = utils.get_variable_value("LABEL_SUMMARY_STRING")
        overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
        texts_group_3_sql = utils.get_difficult_texts_sql()
        texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")

        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message,
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
        texts_group_1_updated = utils.get_texts_group_x(table_name="group1Texts")
        update_ids = list()
        value_records = list()
        for obj in texts_group_1_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_group1
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            value_records.append(value_record)

        utils.add_log_click_value_sql(records=value_records)

        text_list_full_sql, texts_list_list_sql = \
            utils.update_texts_list_by_id_sql(selected_label=selected_label_group1, update_ids=update_ids,
                                              sub_list_limit=table_limit_sql, labels_got_overridden_flag=[],
                                              update_in_place=update_in_place_sql)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            utils.generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        classifier_sql = \
            utils.fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
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
                utils.get_top_predictions_sql(selected_class=selected_label_group1,
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
            texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, recommendations_summary_sql, overall_quality_score_sql = \
            utils.generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                          full_fit_if_labels_got_overridden=True, round_to=1,
                                                          format_as_percentage=True)

        overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
        initialize_flags_sql = utils.get_panel_flags()
        texts_group_3_sql = utils.get_difficult_texts_sql()

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group1,
                               info_message="Labels assigned to group",
                               page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]

    y_classes_sql = utils.get_y_classes()

    text_list_full_sql = utils.get_text_list(table_name="texts")
    new_text = utils.get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql,
                                             table_limit_sql=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    corpus_text_ids = utils.get_pkl(name="CORPUS_TEXT_IDS")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    update_in_place_sql = utils.get_variable_value(name="UPDATE_TEXTS_IN_PLACE")

    selected_label_group2 = request.form["selected_label_group2"]

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_group2)
    utils.add_log_click_value_sql(records=[value_record])

    info_message = ""
    if (selected_label_group2 is None) or (selected_label_group2 == "None") or (selected_label_group2 == "") or \
            (len(selected_label_group2) == 0):
        if selected_label_group2 == "":
            info_message += f"Select a 'Label'."

        if len(selected_label_group2) == 0:
            info_message += "\n" + f"Select a 'Text ID'."

        text_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                         sub_list_limit=table_limit_sql)
        texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
        texts_group_3_sql = utils.get_difficult_texts_sql()
        total_summary_sql = utils.get_total_summary_sql()
        label_summary_sql = utils.get_label_summary_sql()
        label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
        overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label_group2,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message,
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
        texts_group_2_updated = copy.deepcopy(utils.get_texts_group_x(table_name="group2Texts"))
        update_ids = list()
        value_records = list()
        for obj in texts_group_2_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_group2
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            value_records.append(value_record)
        utils.add_log_click_value_sql(records=value_records)

        text_list_full_sql, texts_list_list_sql = \
            utils.update_texts_list_by_id_sql(selected_label=selected_label_group2,
                                              update_ids=update_ids,
                                              sub_list_limit=table_limit_sql,
                                              labels_got_overridden_flag=[],
                                              update_in_place=update_in_place_sql)

        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            utils.generate_summary_sql(text_lists=text_list_full_sql)

        # Group 2 **************************************************************************************
        vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
        corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
        fit_classifier_verbose_sql = utils.get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
        labels_got_overridden_flag_sql = utils.get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
        random_state_sql = utils.get_variable_value(name="RND_STATE")
        full_fit_if_labels_got_overridden_sql = utils.get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
        classifier_sql = utils.fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
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
                utils.get_top_predictions_sql(selected_class=selected_label_group2,
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
            texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            utils.generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                          full_fit_if_labels_got_overridden=True, round_to=1,
                                                          format_as_percentage=True)

        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_group2,
                               info_message="Labels assigned to group",
                               page_navigation_message=page_navigation_message,
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

    y_classes_sql = utils.get_y_classes()

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    # text_list_search_results_sql = utils.get_text_list(table_name="searchResults")
    text_list_full_sql = utils.get_text_list(table_name="texts")
    new_text = utils.get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")

    text_list_list_sql, text_list_appropriate_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql,
                                             table_limit_sql=table_limit_sql)

    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_appropriate_sql,
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)

    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    initialize_flags_sql = utils.get_panel_flags()

    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    label_summary_sql = utils.get_label_summary_sql()
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    number_unlabeled_texts_sql = utils.get_variable_value(name="NUMBER_UNLABELED_TEXTS")
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    update_in_place_sql = utils.get_variable_value(name="UPDATE_TEXTS_IN_PLACE")

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    if html_config_template_sql in ["text_labeling_2.html"]:
        scroll_to_id = "labelAllButton"
    else:
        scroll_to_id = None

    if number_unlabeled_texts_sql == 0:
        info_message = "There are no more unlabeled texts. If you are unhappy with the quality of the " \
                       "auto-labeling, then work through the 'Difficult Texts' to improve the quality."
        label_summary_message = info_message
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               label_selected=selected_label,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message,
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

    classifier_sql = utils.get_pkl(name="CLASSIFIER")
    if classifier_sql:
        confirm_label_all_texts_counts_sql = utils.get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
        if confirm_label_all_texts_counts_sql == 0:
            info_message = \
                f"Are you sure that you want to auto-label the " \
                f"remaining {number_unlabeled_texts_sql:,} texts?"
            label_summary_message = info_message

            temp_count = utils.get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
            utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=(temp_count + 1))

            return render_template(html_config_template_sql,
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   page_navigation_message=page_navigation_message,
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
            temp_count = utils.get_variable_value(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS")
            utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=(temp_count + 1))

            return render_template(html_config_template_sql,
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   page_navigation_message=page_navigation_message,
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

            classifier_sql = utils.get_pkl(name="CLASSIFIER")
            vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
            text_list_full_sql, texts_list_list_sql, labeled_text_ids = \
                utils.label_all_sql(fitted_classifier=classifier_sql,
                                    sparse_vectorized_corpus=vectorized_corpus_sql,
                                    corpus_text_ids=corpus_text_ids_sql,
                                    texts_list=text_list_full_sql,
                                    label_only_unlabeled=True,
                                    sub_list_limit=table_limit_sql,
                                    update_in_place=update_in_place_sql)

            for labeled_text_id in labeled_text_ids:
                value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=labeled_text_id)
                utils.add_log_click_value_sql(records=[value_record])

            total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
                utils.generate_summary_sql(text_lists=text_list_full_sql)

            label_summary_message = info_message
            texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")

            return render_template(html_config_template_sql,
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=selected_label,
                                   info_message=info_message,
                                   page_navigation_message=page_navigation_message,
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
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="export_records",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    download_files = []

    print("After download_files")

    text_list_full_sql = utils.get_text_list(table_name="texts")
    texts_df = pd.DataFrame.from_dict(text_list_full_sql)
    texts_df.to_csv("./output/labeled_texts.csv", index=False)
    download_files.append("./output/labeled_texts.csv")

    click_log_sql = utils.get_click_log()
    click_log_df = pd.DataFrame.from_dict(click_log_sql)
    click_log_df.to_csv("./output/click_log.csv", index=False)
    download_files.append("./output/click_log.csv")

    value_log_sql = utils.get_value_log()
    value_log_df = pd.DataFrame.from_dict(value_log_sql)
    value_log_df.to_csv("./output/value_log.csv", index=False)
    download_files.append("./output/value_log.csv")

    print("After value_log_sql")
    classifier_sql = utils.get_pkl(name="CLASSIFIER")
    if classifier_sql:
        filename = "trained-classifier.pkl"
        pickle.dump(classifier_sql, open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

        vectorizer_sql = utils.get_pkl(name="VECTORIZER")
        filename = "fitted-vectorizer.pkl"
        pickle.dump(vectorizer_sql, open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

    total_summary_sql = utils.get_total_summary_sql()
    if len(total_summary_sql) > 0:
        filename = "total-summary.csv"
        total_summary_df = pd.DataFrame.from_dict(total_summary_sql)
        total_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    label_summary_sql = utils.get_label_summary_sql()
    if len(label_summary_sql) > 0:
        filename = "label-summary.csv"
        label_summary_df = pd.DataFrame.from_dict(label_summary_sql)
        label_summary_df.to_csv("./output/" + filename, index=False)
        download_files.append("./output/" + filename)

    date_time_sql = utils.get_variable_value(name="DATE_TIME")
    filename = "rapid-labeling-results-" + date_time_sql + ".zip"
    file_path = "./output/download/rapid-labeling-results-" + date_time_sql + ".zip"
    zip_file = zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED)
    for file in download_files:
        zip_file.write(file)
    zip_file.close()
    return send_file(file_path, mimetype='zip', download_name=filename, as_attachment=True)


@app.route('/save_state', methods=['GET', 'POST'])
def save_state():
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)
    click_record, guid = utils.generate_click_record(click_location="summary",
                                                     click_type="save_state",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    y_classes_sql = utils.get_y_classes()
    text_list_full_sql = utils.get_text_list(table_name="texts")
    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, 
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    initialize_flags_sql = utils.get_panel_flags()

    classifier_sql = utils.get_pkl(name="CLASSIFIER")
    vectorizer_sql = utils.get_pkl(name="VECTORIZER")
    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    click_log_sql = utils.get_click_log()
    value_log_sql = utils.get_value_log()

    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()

    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")

    save_state = {}
    save_state["DATASET_NAME"] = utils.get_variable_value(name="DATASET_NAME")
    save_state["DATASET_URL"] = utils.get_variable_value(name="DATASET_URL")
    save_state["TOTAL_SUMMARY"] = utils.get_total_summary_sql()
    save_state["LABEL_SUMMARY"] = utils.get_label_summary_sql()
    save_state["RECOMMENDATIONS_SUMMARY"] = []
    # save_state["TEXTS_LIST_LABELED"] = config.TEXTS_LIST_LABELED
    save_state["TEXTS_GROUP_1"] = utils.get_texts_group_x(table_name="group1Texts")
    save_state["TEXTS_GROUP_2"] = utils.get_texts_group_x(table_name="group2Texts")
    save_state["TEXTS_GROUP_3"] = utils.get_difficult_texts_sql()
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    save_state["SEARCH_MESSAGE"] = search_message_sql

    save_state["TEXTS_LIST"] = utils.get_text_list(table_name="texts")
    save_state["TEXTS_LIST_FULL"] = [text_list_full_sql]

    save_state["CORPUS_TEXT_IDS"] = utils.get_pkl(name="CORPUS_TEXT_IDS")
    save_state["TEXTS_LIST_LIST"] = [texts_list_list_sql]
    save_state["TEXTS_LIST_LIST_FULL"] = [texts_list_list_sql]

    # save_state["TOTAL_PAGES_FULL"] = config.TOTAL_PAGES_FULL
    # save_state["ADJ_TEXT_IDS"] = config.ADJ_TEXT_IDS
    save_state["TOTAL_PAGES"] = utils.get_variable_value(name="TOTAL_PAGES")
    save_state["Y_CLASSES"] = utils.get_variable_value(name="Y_CLASSES")

    save_state["SHUFFLE_BY"] = utils.get_variable_value(name="SHUFFLE_BY")
    save_state["NUMBER_UNLABELED_TEXTS"] = utils.get_variable_value(name="NUMBER_UNLABELED_TEXTS")
    save_state["HTML_CONFIG_TEMPLATE"] = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    save_state["LABEL_SUMMARY_STRING"] = utils.get_variable_value(name="LABEL_SUMMARY_STRING")

    save_state["PREDICTIONS_NUMBER"] = utils.get_variable_value(name="PREDICTIONS_NUMBER")
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
                           page_navigation_message=page_navigation_message,
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
    panel_flag = eval(request.args.get("panel_flag", None))
    utils.update_panel_flags_sql(update_flag=panel_flag)
    return panel_flag


@app.route('/update_search_exclude_already_labeled_sql', methods=['POST'])
def update_search_exclude_already_labeled_sql():
    search_exclude_already_labeled = request.args.get("search_exclude_already_labeled", None)
    print("search_exclude_already_labeled :", search_exclude_already_labeled)
    utils.set_variable(name="SEARCH_EXCLUDE_ALREADY_LABELED", value=search_exclude_already_labeled)
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    print("search_exclude_already_labeled_sql :", search_exclude_already_labeled_sql)
    return search_exclude_already_labeled


@app.route('/label_selected', methods=['GET', 'POST'])
def label_selected():
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    label_selected = request.args.get("label_selected")
    selected_text_id = request.args.get("selected_text_id")

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = utils.get_text_list(table_name="texts")
    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql,
                                             table_limit_sql=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    y_classes_sql = utils.get_y_classes()
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    corpus_text_ids = utils.get_pkl(name="CORPUS_TEXT_IDS")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    keep_original_sql = utils.get_variable_value(name="KEEP_ORIGINAL")
    similar_text_verbose_sql = utils.get_variable_value(name="SIMILAR_TEXT_VERBOSE")

    page_number = int(request.args.get("page_number"))

    click_record, guid = utils.generate_click_record(click_location="label",
                                                     click_type="label_selected",
                                                     click_object="radio_button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=label_selected)
    utils.add_log_click_value_sql(records=[value_record])

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        utils.set_texts_group_x(None, "group2Texts")
        texts_group_1_sql = []
    else:
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                       corpus_text_ids=corpus_text_ids,
                                                                       text_id=selected_text_id,
                                                                       keep_original=keep_original_sql)

        texts_group_1_sql = utils.get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                            similarities_series=similarities_series,
                                                            top=group_1_keep_top_sql,
                                                            exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                            verbose=similar_text_verbose_sql)
    # **********************************************************************************************

    # Group 2 ************************************************************************************
    classifier_sql = utils.get_pkl(name="CLASSIFIER")
    if classifier_sql and label_selected != "None":
        texts_group_2_sql = \
            utils.get_top_predictions_sql(selected_class=label_selected,
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
        texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************

    info_message = f"'{label_selected}' selected"


    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    initialize_flags_sql = utils.get_panel_flags()
    label_summary_sql = utils.get_label_summary_sql()
    total_summary_sql = utils.get_total_summary_sql()
    texts_group_3_sql = utils.get_difficult_texts_sql()
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    id = request.form["selected_text_id"]
    label_selected = request.form["selected_label"]

    y_classes_sql = utils.get_y_classes()
    text_list_full_sql = utils.get_text_list(table_name="texts")

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    text = utils.get_selected_text(selected_text_id=id, text_list_full_sql=text_list_full_sql)

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="generate_list",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    # Group 3 ************************************************************************************

    result, message, texts_group_3_sql, overall_quality_score_sql = \
        utils.generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
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
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    y_classes_sql = utils.get_y_classes()
    group1_table_limit = int(request.form.get("group1_table_limit", None))

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = utils.get_text_list(table_name="texts")
    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    group_1_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_1_EXCLUDE_ALREADY_LABELED")
    keep_original_sql = utils.get_variable_value(name="KEEP_ORIGINAL")

    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    label_selected = request.form.get("label_selected", None)
    info_message = f"Set 'Similarity' display limit to {group1_table_limit}"
    utils.set_variable(name="GROUP_1_KEEP_TOP", value=group1_table_limit)
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")

    click_record, guid = utils.generate_click_record(click_location="similar_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=table_limit_sql)
    utils.add_log_click_value_sql(records=[value_record])

    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        utils.set_texts_group_x(None, table_name="group1Texts")
        texts_group_1_sql = []
    else:
        vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
        corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
        similar_text_verbose_sql = utils.get_pkl(name="SIMILAR_TEXT_VERBOSE")
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=vectorized_corpus_sql,
                                                                       corpus_text_ids=corpus_text_ids_sql,
                                                                       text_id=selected_text_id,
                                                                       keep_original=keep_original_sql)

        texts_group_1_sql = utils.get_top_similar_texts_sql(all_texts_json=text_list_full_sql,
                                                            similarities_series=similarities_series,
                                                            top=group_1_keep_top_sql,
                                                            exclude_already_labeled=group_1_exclude_already_labeled_sql,
                                                            verbose=similar_text_verbose_sql)

    # **********************************************************************************************
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    initialize_flags_sql = utils.get_panel_flags()

    texts_group_3_sql = utils.get_difficult_texts_sql()
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    # texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")

    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)

    y_classes_sql = utils.get_y_classes()
    group_2_table_limit = int(request.form.get("group2_table_limit", None))

    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = utils.get_text_list(table_name="texts")
    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, 
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")
    initialize_flags_sql = utils.get_panel_flags()
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")
    label_selected = request.form.get("label_selected", None)

    info_message = f"Set 'Recommendations' display limit to {group_2_table_limit}"
    utils.set_variable(name="PREDICTIONS_NUMBER", value=group_2_table_limit)
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")

    click_record, guid = utils.generate_click_record(click_location="recommended_texts",
                                                     click_type="set_table_limit",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])
    value_record = utils.generate_value_record(guid=guid, value_type="table_limit", value=group_2_table_limit)
    utils.add_log_click_value_sql(records=[value_record])

    # Group 2 ************************************************************************************
    classifier_sql = utils.get_pkl(name="CLASSIFIER")
    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    if classifier_sql:
        texts_group_2_sql = \
            utils.get_top_predictions_sql(selected_class=label_selected,
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
        texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    # **********************************************************************************************
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    total_summary_sql = utils.get_total_summary_sql()
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)

    text_list_full_sql = utils.get_text_list(table_name="texts")
    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    total_summary_sql = utils.get_total_summary_sql()
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    initialize_flags_sql = utils.get_panel_flags()
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    label_selected = request.form.get("label_selected", None)
    include_search_term = request.form.get("include_search_term", None)
    exclude_search_term = request.form.get("exclude_search_term", None)
    allow_search_override_labels = request.form.get('searchAllTextRadio', None)

    if allow_search_override_labels == "Yes":
        utils.set_variable(name="SEARCH_EXCLUDE_ALREADY_LABELED", value=False)
    else:
        utils.set_variable(name="SEARCH_EXCLUDE_ALREADY_LABELED", value=True)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="search_texts",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])
    value_record_1 = utils.generate_value_record(guid=guid, value_type="include_search_term", value=include_search_term)
    utils.add_log_click_value_sql(records=[value_record_1])
    value_record_2 = utils.generate_value_record(guid=guid, value_type="exclude_search_term", value=exclude_search_term)
    utils.add_log_click_value_sql(records=[value_record_2])

    y_classes_sql = utils.get_y_classes()

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

            utils.set_variable(name="SEARCH_TOTAL_PAGES", value=search_results_total_pages)

            info_message = f"Showing results Include-'{include_search_term}', Exclude-'{exclude_search_term}'"
            utils.set_variable(name="SEARCH_MESSAGE", value=info_message)
            utils.set_variable(name="SEARCH_RESULTS_LENGTH", value=search_results_length_sql)
            page_navigation_message_search = "Displaying {:,} texts ({} per page)".format(search_results_length_sql,
                                                                                          table_limit_sql)
            return render_template(html_config_template_sql,
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   page_navigation_message=page_navigation_message_search,
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
            utils.set_variable(name="SEARCH_MESSAGE", value=info_message)
            utils.set_variable(name="SEARCH_RESULTS_LENGTH", value=0)
            utils.set_variable(name="SEARCH_TOTAL_PAGES", value=0)
            page_number = -1
            search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
            return render_template(html_config_template_sql,
                                   selected_text_id=selected_text_id,
                                   selected_text=selected_text,
                                   label_selected=label_selected,
                                   info_message=info_message,
                                   page_navigation_message="No search results",
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
        # text_list_full_sql = utils.get_text_list(table_name="texts")
        text_list_list = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, 
                                                     sub_list_limit=table_limit_sql)
        utils.set_variable(name="SEARCH_MESSAGE", value=info_message)
        return render_template(html_config_template_sql,
                               selected_text_id=selected_text_id,
                               selected_text=selected_text,
                               label_selected=label_selected,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]

    y_classes_sql = utils.get_y_classes()

    text_list_full_sql = utils.get_text_list(table_name="texts")
    new_text = utils.get_selected_text(selected_text_id=new_id, text_list_full_sql=text_list_full_sql)
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")

    text_list_list_sql, text_list_full_sql, total_pages_sql = \
        utils.get_appropriate_text_list_list(text_list_full_sql=text_list_full_sql, total_pages_sql=total_pages_sql,
                                             search_results_length_sql=search_results_length_sql,
                                             table_limit_sql=table_limit_sql)

    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")

    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    selected_label_search_texts = request.form["selected_label_search_texts"]
    vectorized_corpus_sql = utils.get_pkl(name="VECTORIZED_CORPUS")
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    initialize_flags_sql = utils.get_panel_flags()
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    predictions_probability_sql = utils.get_variable_value(name="PREDICTIONS_PROBABILITY")
    predictions_verbose_sql = utils.get_variable_value(name="PREDICTIONS_VERBOSE")
    group_2_exclude_already_labeled_sql = utils.get_variable_value(name="GROUP_2_EXCLUDE_ALREADY_LABELED")

    corpus_text_ids_sql = utils.get_pkl(name="CORPUS_TEXT_IDS")
    random_state_sql = utils.get_variable_value(name="RND_STATE")
    fit_classifier_verbose_sql = utils.get_variable_value(name="FIT_CLASSIFIER_VERBOSE")
    full_fit_if_labels_got_overridden_sql = utils.get_variable_value(name="FULL_FIT_IF_LABELS_GOT_OVERRIDDEN")
    update_in_place_sql = utils.get_variable_value(name="UPDATE_TEXTS_IN_PLACE")

    page_navigation_message_search = "Displaying {:,} texts ({} per page)".format(search_results_length_sql,
                                                                                  table_limit_sql)

    info_message = ""

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])

    value_record = utils.generate_value_record(guid=guid, value_type="label", value=selected_label_search_texts)
    utils.add_log_click_value_sql(records=[value_record])

    if (selected_label_search_texts == "None") or (selected_label_search_texts is None) \
            or (selected_label_search_texts == ""):
        if selected_label_search_texts == "":
            info_message += f"Select a 'Label'."

        # text_list_list = utils.create_text_list_list(text_list_full_sql=text_list_full_sql, sub_list_limit=table_limit_sql)
        total_summary_sql = utils.get_total_summary_sql()
        label_summary_sql = utils.get_label_summary_sql()
        label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
        search_results_texts_list_sql = utils.get_text_list(table_name="searchResults")
        search_results_texts_list_list_sql = \
            utils.create_text_list_list(text_list_full_sql=search_results_texts_list_sql, 
                                        sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id=new_id,
                               selected_text=new_text,
                               info_message=info_message,
                               page_navigation_message=page_navigation_message_search,
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
        test_start_time = datetime.now()
        texts_group_updated = utils.get_text_list(table_name="searchResults")
        test_end_time = datetime.now()
        test_duration = test_end_time - test_start_time
        print("texts_group_updated - test_start_time :", test_start_time.strftime("%H:%M:%S"))
        print("texts_group_updated - test_end_time :", test_end_time.strftime("%H:%M:%S"))
        print("texts_group_updated - test_duration :", test_duration)

        test_start_time = datetime.now()
        update_ids = list()
        value_records = list()
        for obj in texts_group_updated:
            update_ids.append(obj["id"])
            obj["label"] = selected_label_search_texts
            value_record = utils.generate_value_record(guid=guid, value_type="text_id", value=obj["id"])
            value_records.append(value_record)
        utils.add_log_click_value_sql(records=[value_record])

        test_end_time = datetime.now()
        test_duration = test_end_time - test_start_time
        print("update_ids - test_start_time :", test_start_time.strftime("%H:%M:%S"))
        print("update_ids - test_end_time :", test_end_time.strftime("%H:%M:%S"))
        print("update_ids - test_duration :", test_duration)

        test_start_time = datetime.now()
        text_list_full_sql, text_list_list_sql = \
            utils.update_texts_list_by_id_sql(selected_label=selected_label_search_texts,
                                              update_ids=update_ids,
                                              sub_list_limit=table_limit_sql,
                                              labels_got_overridden_flag=[],
                                              update_in_place=update_in_place_sql)
        test_end_time = datetime.now()
        test_duration = test_end_time - test_start_time
        print("update_texts_list_by_id_sql - test_start_time :", test_start_time.strftime("%H:%M:%S"))
        print("update_texts_list_by_id_sql - test_end_time :", test_end_time.strftime("%H:%M:%S"))
        print("update_texts_list_by_id_sql - test_duration :", test_duration)

        test_start_time = datetime.now()
        utils.set_text_list(label=selected_label_search_texts, table_name="searchResults")
        test_end_time = datetime.now()
        test_duration = test_end_time - test_start_time
        print("set_text_list - test_start_time :", test_start_time.strftime("%H:%M:%S"))
        print("set_text_list - test_end_time :", test_end_time.strftime("%H:%M:%S"))
        print("set_text_list - test_duration :", test_duration)

        test_start_time = datetime.now()
        total_summary_sql, label_summary_sql, number_unlabeled_texts_sql, label_summary_string_sql = \
            utils.generate_summary_sql(text_lists=text_list_full_sql)
        test_end_time = datetime.now()
        test_duration = test_end_time - test_start_time
        print("generate_summary_sql - test_start_time :", test_start_time.strftime("%H:%M:%S"))
        print("generate_summary_sql - test_end_time :", test_end_time.strftime("%H:%M:%S"))
        print("generate_summary_sql - test_duration :", test_duration)
        # Group 2 **************************************************************************************
        labels_got_overridden_flag_sql = utils.get_variable_value(name="LABELS_GOT_OVERRIDDEN_FLAG")
        classifier_sql = utils.fit_classifier_sql(sparse_vectorized_corpus=vectorized_corpus_sql,
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
                utils.get_top_predictions_sql(selected_class=selected_label_search_texts,
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
            texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")

        result, difficult_texts_message, texts_group_3_sql, overall_quality_score_sql = \
            utils.generate_all_predictions_if_appropriate(n_jobs=-1, labels_got_overridden_flag=True,
                                                          full_fit_if_labels_got_overridden=True, round_to=1,
                                                          format_as_percentage=True)
        # **********************************************************************************************

        search_results_texts_list_sql = utils.get_text_list(table_name="searchResults")
        search_results_texts_list_list_sql = \
            utils.create_text_list_list(text_list_full_sql=search_results_texts_list_sql,
                                        sub_list_limit=table_limit_sql)
        return render_template(html_config_template_sql,
                               selected_text_id="None", # new_id,
                               selected_text="Select a text below to label.", # new_text,
                               label_selected=selected_label_search_texts,
                               info_message="Labels assigned to group",
                               page_navigation_message=page_navigation_message_search,
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
    utils.set_variable(name="CONFIRM_LABEL_ALL_TEXTS_COUNTS", value=0)

    selected_text_id = request.form.get("selected_text_id", None)

    y_classes_sql = utils.get_y_classes()
    table_limit_sql = utils.get_variable_value(name="TABLE_LIMIT")
    text_list_full_sql = utils.get_text_list(table_name="texts")
    texts_list_list_sql = utils.create_text_list_list(text_list_full_sql=text_list_full_sql,
                                                      sub_list_limit=table_limit_sql)
    page_navigation_message = "Displaying {:,} texts ({} per page)".format(len(text_list_full_sql), table_limit_sql)
    selected_text = utils.get_selected_text(selected_text_id=selected_text_id, text_list_full_sql=text_list_full_sql)
    html_config_template_sql = utils.get_variable_value(name="HTML_CONFIG_TEMPLATE")
    predictions_number_sql = utils.get_variable_value(name="PREDICTIONS_NUMBER")
    overall_quality_score_sql = utils.get_variable_value(name="OVERALL_QUALITY_SCORE")
    search_results_length_sql = utils.get_variable_value(name="SEARCH_RESULTS_LENGTH")
    label_summary_sql = utils.get_label_summary_sql()
    label_summary_string_sql = utils.get_variable_value(name="LABEL_SUMMARY_STRING")
    total_summary_sql = utils.get_total_summary_sql()
    texts_group_1_sql = utils.get_texts_group_x(table_name="group1Texts")
    texts_group_2_sql = utils.get_texts_group_x(table_name="group2Texts")
    texts_group_3_sql = utils.get_difficult_texts_sql()
    initialize_flags_sql = utils.get_panel_flags()
    group_1_keep_top_sql = utils.get_variable_value(name="GROUP_1_KEEP_TOP")
    search_exclude_already_labeled_sql = utils.get_variable_value(name="SEARCH_EXCLUDE_ALREADY_LABELED")

    label_selected = request.form.get("label_selected", None)
    info_message = "Search cleared"

    # config.SEARCH_MESSAGE.clear()
    # config.SEARCH_MESSAGE.append("Displaying all texts")
    utils.set_variable(name="SEARCH_MESSAGE", value="Displaying all texts")

    # config.TEXTS_LIST.clear()
    # config.TEXTS_LIST.append(config.TEXTS_LIST_FULL[0])

    # config.TEXTS_LIST_LIST.clear()
    # config.TEXTS_LIST_LIST.append(config.TEXTS_LIST_LIST_FULL[0])

    # config.TOTAL_PAGES.clear()
    # config.TOTAL_PAGES.append(config.TOTAL_PAGES_FULL[0])
    total_pages_sql = utils.get_variable_value(name="TOTAL_PAGES")

    # config.SEARCH_RESULT_LENGTH.clear()
    # config.SEARCH_RESULT_LENGTH.append(0)
    utils.set_variable(name="SEARCH_RESULTS_LENGTH", value=0)

    click_record, guid = utils.generate_click_record(click_location="all_texts",
                                                     click_type="clear_search",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_click_record_sql(records=[click_record])
    search_message_sql = utils.get_variable_value(name="SEARCH_MESSAGE")
    return render_template(html_config_template_sql,
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=info_message,
                           page_navigation_message=page_navigation_message,
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

