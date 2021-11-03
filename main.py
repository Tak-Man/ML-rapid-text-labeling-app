from flask import Flask, render_template, request, jsonify, flash, make_response, redirect, g, send_file
import web_app_utilities as utils
import configuration as config
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
# from flask.ext.session import Session
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import zipfile
import pickle


start_time = datetime.now()
print("*"*20, "Process started @", start_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)

app = Flask(__name__)
app.secret_key = "super secret key"
app.config.from_object(__name__)
# Session(app)


consolidated_disaster_tweet_data_df = utils.get_disaster_tweet_demo_data(number_samples=None,
                                                                         filter_data_types=["train"],
                                                                         random_state=config.RND_STATE)

CORPUS_TEXT_IDS = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]


def load_demo_data(dataset="Disaster Tweets Dataset", shuffle_by="kmeans",
                   table_limit=50, texts_limit=1000, max_features=100,
                   y_classes=["Earthquake", "Fire", "Flood", "Hurricane"], rnd_state=258):
    if dataset == "Disaster Tweets Dataset":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)
        vectorized_corpus = \
            vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"])
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

    return texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer


end_time = datetime.now()
print("*"*20, "Process ended @", end_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)
duration = end_time - start_time
print("*"*20, "Process duration -", duration, "*"*20)


@app.route("/")
def index():
    return render_template("start.html",
                           dataset_list=config.DATASETS_AVAILABLE)


@app.route("/dataset_selected", methods=["GET"])
def dataset_selected():
    dataset_name = request.args.get("dataset_name", None)

    if dataset_name == "Disaster Tweets Dataset":
        config.Y_CLASSES.clear()
        config.Y_CLASSES.append(["Earthquake", "Fire", "Flood", "Hurricane", "Other"])
        config.SHUFFLE_BY.clear()
        config.SHUFFLE_BY.append("random")  # kmeans random

        texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus, vectorizer = \
            load_demo_data(dataset="Disaster Tweets Dataset", shuffle_by=config.SHUFFLE_BY[0],
                           table_limit=config.TABLE_LIMIT, texts_limit=config.TEXTS_LIMIT,
                           max_features=config.MAX_FEATURES,
                           y_classes=config.Y_CLASSES[0], rnd_state=config.RND_STATE)

        for master, update in zip([config.TEXTS_LIST, config.TEXTS_LIST_LIST, config.ADJ_TEXT_IDS,
                                   config.TOTAL_PAGES, config.VECTORIZED_CORPUS, config.VECTORIZER_LIST],
                                  [texts_list, texts_list_list, adj_text_ids, total_pages, vectorized_corpus,
                                   vectorizer]):
            master.clear()
            master.append(update)

        config.TEXTS_LIST_FULL.clear()
        config.TEXTS_LIST_FULL.append(texts_list)

        config.TEXTS_LIST_LIST_FULL.clear()
        config.TEXTS_LIST_LIST_FULL.append(texts_list_list)

        config.TOTAL_PAGES_FULL.clear()
        config.TOTAL_PAGES_FULL.append(total_pages)

        return render_template("start.html",
                               dataset_list=config.DATASETS_AVAILABLE,
                               texts_list=config.TEXTS_LIST_LIST[0][0],
                               dataset_name=dataset_name,
                               dataset_labels=config.Y_CLASSES[0])

    # elif dataset_name == "":




@app.route("/begin_labeling", methods=["POST"])
def begin_labeling():
    dataset_name = request.args.get("dataset_name", None)
    selected_config = request.form.get("selected_config", None)

    if not dataset_name or not selected_config:
        config1_message = "Select a dataset AND configuration above"
        return render_template("start.html",
                               dataset_list=config.DATASETS_AVAILABLE,
                               config1_message=config1_message)

    if dataset_name == "Disaster Tweets Dataset":
        page_number = 0
        utils.generate_summary(text_lists=config.TEXTS_LIST[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)
        config.SEARCH_MESSAGE.clear()
        config.SEARCH_MESSAGE.append("Displaying all texts")

        if selected_config == "config1":
            html_config_template = "text_labeling_1.html"
        elif selected_config == "config2":
            html_config_template = "text_labeling_2.html"
        else:
            html_config_template = "text_labeling_1.html"

        return render_template(html_config_template,
                               selected_text_id="None",
                               selected_text="Select a text below to begin labeling.",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

# @app.route("/")
# def index():
#     page_number = 0
#
#     utils.generate_summary(text_lists=TEXTS_LIST[0],
#                            first_labeling_flag=FIRST_LABELING_FLAG,
#                            total_summary=TOTAL_SUMMARY,
#                            label_summary=LABEL_SUMMARY)
#
#     return render_template("text_labeling_1.html",
#                            selected_text_id="None",
#                            selected_text="Select a text below to begin labeling.",
#                            info_message="",
#                            page_number=0,
#                            y_classes=Y_CLASSES[0],
#                            total_pages=TOTAL_PAGES[0],
#                            texts_list=TEXTS_LIST_LIST[0][page_number],
#                            texts_group_1=[],
#                            group1_table_limit_value=GROUP_1_KEEP_TOP[0],
#                            texts_group_2=[],
#                            group2_table_limit_value=PREDICTIONS_NUMBER[0],
#                            texts_group_3=[],
#                            total_summary=TOTAL_SUMMARY,
#                            label_summary=LABEL_SUMMARY)


@app.route("/text_labeling_1.html", methods=["GET", "POST"])
def text_labeling():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    app_section = request.args.get("app_section", None)
    selected_text_id = request.args.get("selected_text_id", None)
    selected_text = request.args.get("selected_text", None)
    label_selected = request.args.get("label_selected", None)
    page_number = int(request.args.get("page_number", None))

    click_record, guid = utils.generate_click_record(click_location=app_section,
                                                     click_type="text_id_selected",
                                                     click_object="link",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    # Group 1 ************************************************************************************
    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                                                   corpus_text_ids=CORPUS_TEXT_IDS,
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
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


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

    return render_template("text_labeling_1.html",
                           selected_text_id="None",
                           selected_text="Select a text below to begin labeling.",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/single_text", methods=["POST"])
def single_text():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    new_id = request.form["selected_text_id"]
    new_text = request.form["selected_text"]
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

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])
    else:
        # old_obj = {"id": new_id, "text": new_text, "label": "-"}
        new_obj = {"id": new_id, "text": new_text, "label": new_label}

        # utils.update_texts_list(texts_list=TEXTS_LIST[0],
        #                         sub_list_limit=TABLE_LIMIT,
        #                         old_obj_lst=[old_obj],
        #                         new_obj_lst=[new_obj],
        #                         texts_list_list=TEXTS_LIST_LIST)

        utils.update_texts_list_by_id(texts_list=config.TEXTS_LIST_FULL[0],
                                      sub_list_limit=config.TABLE_LIMIT,
                                      updated_obj_lst=[new_obj],
                                      texts_list_list=config.TEXTS_LIST_LIST_FULL[0],
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=CORPUS_TEXT_IDS,
                             texts_list_labeled=[new_obj],
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (new_label is not None and new_label != "" and new_label != "None"):
            utils.get_top_predictions(selected_class=new_label,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=CORPUS_TEXT_IDS,
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)

        # **********************************************************************************************

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/grouped_1_texts", methods=["POST"])
def grouped_1_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = request.form["selected_text"]
    selected_label_group1 = request.form["selected_label_group1"]
    info_message = ""

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

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

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
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=CORPUS_TEXT_IDS,
                             texts_list_labeled=texts_group_1_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (selected_label_group1 is not None and
                 selected_label_group1 != "" and
                 selected_label_group1 != "None"):
            utils.get_top_predictions(selected_class=selected_label_group1,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=CORPUS_TEXT_IDS,
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)
        # **********************************************************************************************

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/grouped_2_texts", methods=["POST"])
def grouped_2_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = request.form["selected_text"]
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

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

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
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=CORPUS_TEXT_IDS,
                             texts_list_labeled=texts_group_2_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST)

        if len(selected_label_group2) > 0 and \
                (selected_label_group2 is not None and
                 selected_label_group2 != "" and
                 selected_label_group2 != "None"):
            utils.get_top_predictions(selected_class=selected_label_group2,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=CORPUS_TEXT_IDS,
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)
        # **********************************************************************************************

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/label_all", methods=["POST"])
def label_all():
    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = request.form["selected_text"]
    selected_label = request.form["selected_label"]

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="group_label_assigned",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    if config.NUMBER_UNLABELED_TEXTS[0] == 0:
        info_message = "There are no more unlabeled texts. If you are unhappy with the quality of the " \
                       "auto-labeling, then work through the 'Difficult Texts' to improve the quality."
        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

    if len(config.CLASSIFIER_LIST) > 0:
        if config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] == 0:
            info_message = \
                f"Are you sure that you want to auto-label the " \
                f"remaining {config.NUMBER_UNLABELED_TEXTS[0]:,} texts?"

            temp_count = config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0]
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(temp_count + 1)

            return render_template("text_labeling_1.html",
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
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

        elif config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] == 1:
            info_message = \
                f"Click 'Label All Texts' again to confirm your choice. The remaining unlabeled texts will be " \
                f"labeled using machine-learning. Inspect the 'Difficult Texts' to see how well the " \
                f"machine-learning model is trained and if you are satisfied then proceed."

            temp_count = config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0]
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
            config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(temp_count + 1)

            return render_template("text_labeling_1.html",
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
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

        elif config.CONFIRM_LABEL_ALL_TEXTS_COUNTS[0] > 1:
            info_message = \
                f"{config.NUMBER_UNLABELED_TEXTS[0]} texts were auto-labeled."

            updated_texts_list, updated_texts_list_list, labeled_text_ids = \
                utils.label_all(fitted_classifier=config.CLASSIFIER_LIST[0],
                                sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                corpus_text_ids=CORPUS_TEXT_IDS,
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
                                   number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)

            return render_template("text_labeling_1.html",
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
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0])
    # **********************************************************************************************
    else:
        info_message = "Label more texts before trying again."
    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/text/<id>", methods=["GET", "PUT", "DELETE"])
def single_record(id):
    if request.method == "GET":
        for text in config.TEXTS_LIST:
            if text in config.TEXTS_LIST:
                return jsonify(text)
            pass

    if request.method == "PUT":
        for text in config.TEXTS_LIST:
            if text["id"] == id:
                text["text"] = request.form["selected_text"]
                text["label"] = request.form["assigned_label"]
                updated_text = {"id": id, "text": text["text"], "label": text["label"]}
                return jsonify(updated_text)

    if request.method == "DELETE":
        for index, book in enumerate(config.TEXTS_LIST):
            if text["id"] == id:
                config.TEXTS_LIST.pop(index)
                return jsonify(config.TEXTS_LIST)


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

    click_log_df = pd.DataFrame.from_dict(config.CLICK_LOG)
    click_log_df.to_csv("./output/click_log.csv", index=False)
    download_files.append("./output/click_log.csv")

    value_log_df = pd.DataFrame.from_dict(config.VALUE_LOG)
    value_log_df.to_csv("./output/value_log.csv", index=False)
    download_files.append("./output/value_log.csv")

    texts_df = pd.DataFrame.from_dict(config.TEXTS_LIST_FULL[0])
    texts_df.to_csv("./output/labeled_texts.csv", index=False)
    download_files.append("./output/labeled_texts.csv")

    if len(config.CLASSIFIER_LIST) > 0:
        filename = "trained-classifier.sav"
        pickle.dump(config.CLASSIFIER_LIST[0], open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

        filename = "fitted-vectorizer.sav"
        pickle.dump(config.VECTORIZER_LIST[0], open("./output/" + filename, "wb"))
        download_files.append("./output/" + filename)

        filename = "labels.txt"
        with open("./output/" + filename, "w") as f:
            for label in config.Y_CLASSES[0]:
                f.write(str(label) + "\n")

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

    filename = "rapid-labeling-results.zip"
    file_path = "./output/download/rapid-labeling-results.zip"
    zip_file = zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED)
    for file in download_files:
        zip_file.write(file)
    zip_file.close()
    return send_file(file_path, mimetype = 'zip', attachment_filename=filename, as_attachment=True)


@app.route('/label_selected', methods=['GET', 'POST'])
def label_selected():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    label_selected = request.args.get("label_selected")
    selected_text_id = request.args.get("selected_text_id")
    selected_text = request.args.get("selected_text")
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
                                                                       corpus_text_ids=CORPUS_TEXT_IDS,
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
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    info_message = f"'{label_selected}' selected"

    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route('/generate_difficult_texts', methods=['POST'])
def generate_difficult_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    id = request.form["selected_text_id"]
    text = request.form["selected_text"]
    label_selected = request.form["selected_label"]

    click_record, guid = utils.generate_click_record(click_location="difficult_texts",
                                                     click_type="generate_list",
                                                     click_object="button",
                                                     guid=None)
    utils.add_log_record(click_record, log=config.CLICK_LOG)

    # Group 3 ************************************************************************************
    if len(config.CLASSIFIER_LIST) > 0:
        utils.get_all_predictions(fitted_classifier=config.CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  texts_list=config.TEXTS_LIST[0],
                                  top=config.GROUP_3_KEEP_TOP,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  round_to=1,
                                  format_as_percentage=True,
                                  similar_texts=config.TEXTS_GROUP_3,
                                  predictions_report=config.RECOMMENDATIONS_SUMMARY,
                                  overall_quality_score=config.OVERALL_QUALITY_SCORE)
        info_message = label_selected

    # **********************************************************************************************
    else:
        info_message = "Label more texts then try generating the difficult text list."
    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/set_group_1_record_limit", methods=["POST"])
def set_group_1_record_limit():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = request.form.get("selected_text", None)
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
                                                                       corpus_text_ids=CORPUS_TEXT_IDS,
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

    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/set_group_2_record_limit", methods=["POST"])
def set_group_2_record_limit():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = request.form.get("selected_text", None)
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
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  texts_list=config.TEXTS_LIST_FULL[0],
                                  top=config.PREDICTIONS_NUMBER[0],
                                  cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                  y_classes=config.Y_CLASSES[0],
                                  verbose=config.PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                  similar_texts=config.TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/search_all_texts", methods=["POST"])
def search_all_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form.get("page_number", None))
    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = request.form.get("selected_text", None)
    label_selected = request.form.get("label_selected", None)
    include_search_term = request.form.get("include_search_term", None)
    exclude_search_term = request.form.get("exclude_search_term", None)

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
                                                exclude_already_labeled=False,
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

            return render_template("text_labeling_1.html",
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
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0])
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

            return render_template("text_labeling_1.html",
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
                                   overall_quality_score=config.OVERALL_QUALITY_SCORE[0])
    else:
        info_message = "No search term entered"
        config.SEARCH_MESSAGE.clear()
        config.SEARCH_MESSAGE.append(info_message)

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/grouped_search_texts", methods=["POST"])
def grouped_search_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    page_number = int(request.form["page_number"])
    new_id = request.form["selected_text_id"]
    new_text = request.form["selected_text"]
    search_results_length = request.form["search_results_length"]
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

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])

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
                                      update_in_place=False)

        utils.generate_summary(text_lists=config.TEXTS_LIST_FULL[0],
                               first_labeling_flag=config.FIRST_LABELING_FLAG,
                               total_summary=config.TOTAL_SUMMARY,
                               label_summary=config.LABEL_SUMMARY,
                               number_unlabeled_texts=config.NUMBER_UNLABELED_TEXTS)

        # Group 2 **************************************************************************************
        utils.fit_classifier(sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                             corpus_text_ids=CORPUS_TEXT_IDS,
                             texts_list_labeled=texts_group_updated,
                             y_classes=config.Y_CLASSES[0],
                             verbose=config.FIT_CLASSIFIER_VERBOSE,
                             classifier_list=config.CLASSIFIER_LIST)

        if len(config.CLASSIFIER_LIST) > 0 and \
                (selected_label_search_texts is not None and
                 selected_label_search_texts != "" and
                 selected_label_search_texts != "None"):
            utils.get_top_predictions(selected_class=selected_label_search_texts,
                                      fitted_classifier=config.CLASSIFIER_LIST[0],
                                      sparse_vectorized_corpus=config.VECTORIZED_CORPUS[0],
                                      corpus_text_ids=CORPUS_TEXT_IDS,
                                      texts_list=config.TEXTS_LIST_FULL[0],
                                      top=config.PREDICTIONS_NUMBER[0],
                                      cutoff_proba=config.PREDICTIONS_PROBABILITY,
                                      y_classes=config.Y_CLASSES[0],
                                      verbose=config.PREDICTIONS_VERBOSE,
                                      exclude_already_labeled=config.GROUP_2_EXCLUDE_ALREADY_LABELED,
                                      similar_texts=config.TEXTS_GROUP_2)
        # **********************************************************************************************

        return render_template("text_labeling_1.html",
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
                               overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


@app.route("/clear_search_all_texts", methods=["POST"])
def clear_search_all_texts():
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.clear()
    config.CONFIRM_LABEL_ALL_TEXTS_COUNTS.append(0)

    selected_text_id = request.form.get("selected_text_id", None)
    selected_text = request.form.get("selected_text", None)
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

    return render_template("text_labeling_1.html",
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
                           overall_quality_score=config.OVERALL_QUALITY_SCORE[0])


if __name__ == "__main__":
    app.run() # host="127.0.0.1:5000"

