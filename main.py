from flask import Flask, render_template, request, jsonify, flash, make_response, redirect
import web_app_utilities as utils
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
# from flask.ext.session import Session
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.cluster import KMeans


start_time = datetime.now()
print("*"*20, "Process started @", start_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)

app = Flask(__name__)
app.secret_key = "super secret key"
app.config.from_object(__name__)
# Session(app)

TEXTS_LIMIT = 100000
TABLE_LIMIT = 50
MAX_FEATURES = 100
KEEP_ORIGINAL = True
GROUP_1_KEEP_TOP = 10
GROUP_1_EXCLUDE_ALREADY_LABELED = True
RND_STATE = 2584
PREDICTIONS_PROBABILITY = 0.95
PREDICTIONS_NUMBER = 50
PREDICTIONS_VERBOSE = True
GROUP_3_KEEP_TOP = 50

TOTAL_SUMMARY = []
LABEL_SUMMARY = []
TEXTS_LIST_LABELED = []
TEXTS_GROUP_1 = []
TEXTS_GROUP_2 = []
TEXTS_GROUP_3 = []
CLASSIFIER_LIST = []
FIRST_LABELING_FLAG = True
Y_CLASSES = ["Earthquake", "Fire", "Flood", "Hurricane"]
SHUFFLE_BY = "random" # kmeans random

consolidated_disaster_tweet_data_df = utils.get_demo_data(number_samples=None,
                                                          filter_data_types=["train"],
                                                          random_state=RND_STATE)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=MAX_FEATURES)
VECTORIZED_CORPUS = \
    vectorizer.fit_transform(consolidated_disaster_tweet_data_df["tweet_text"])

CORPUS_TEXT_IDS = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]
# print("type(vectorized_corpus) :", type(VECTORIZED_CORPUS))
# print("vectorized_corpus.shape :", VECTORIZED_CORPUS.shape)


if SHUFFLE_BY == "kmeans":
    kmeans = KMeans(n_clusters=len(Y_CLASSES), random_state=RND_STATE).fit(VECTORIZED_CORPUS)
    KMEANS_LABELS = kmeans.labels_
    print("KMEANS_LABELS :", KMEANS_LABELS[:10])
    print("Counter(KMEANS_LABELS) :", Counter(KMEANS_LABELS))
    print("len(KMEANS_LABELS) :", len(KMEANS_LABELS))
    TEXTS_LIST, adj_text_ids = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                      limit=TEXTS_LIMIT,
                                                                      keep_labels=False,
                                                                      shuffle_list=KMEANS_LABELS,
                                                                      random_shuffle=False,
                                                                      random_state=RND_STATE)
else:
    TEXTS_LIST, adj_text_ids = utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df,
                                                                      limit=TEXTS_LIMIT,
                                                                      keep_labels=False,
                                                                      shuffle_list=[],
                                                                      random_shuffle=True,
                                                                      random_state=RND_STATE)

TEXTS_LIST_LIST = [TEXTS_LIST[i:i + TABLE_LIMIT] for i in range(0, len(TEXTS_LIST), TABLE_LIMIT)]
TOTAL_PAGES = len(TEXTS_LIST_LIST)
# print("TOTAL_PAGES :", TOTAL_PAGES)


end_time = datetime.now()
print("*"*20, "Process ended @", end_time.strftime("%m/%d/%Y, %H:%M:%S"), "*"*20)
duration = end_time - start_time
print("*"*20, "Process duration -", duration, "*"*20)


@app.route("/")
def index():
    page_number = 0

    utils.generate_summary(text_lists=TEXTS_LIST,
                           first_labeling_flag=FIRST_LABELING_FLAG,
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)

    return render_template("text_labeling_1.html",
                           selected_text_id="None",
                           selected_text="Select a text below to begin labeling.",
                           info_message="",
                           page_number=0,
                           y_classes=Y_CLASSES,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=[],
                           texts_group_2=[],
                           texts_group_3=[],
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)


@app.route("/text_labeling_1.html", methods=["GET", "POST"])
def text_labeling():
    selected_text_id = request.args.get("selected_text_id", None)
    selected_text = request.args.get("selected_text", None)
    label_selected = request.args.get("label_selected", None)
    page_number = int(request.args.get("page_number", None))
    # print("selected_text_id :", selected_text_id)
    # print("selected_text :", selected_text)
    print("label_selected :", label_selected)

    # Group 1 ************************************************************************************
    similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                                                   corpus_text_ids=CORPUS_TEXT_IDS,
                                                                   text_id=selected_text_id,
                                                                   keep_original=KEEP_ORIGINAL)

    utils.get_top_similar_texts(all_texts_json=TEXTS_LIST,
                                similarities_series=similarities_series,
                                top=GROUP_1_KEEP_TOP,
                                exclude_already_labeled=GROUP_1_EXCLUDE_ALREADY_LABELED,
                                similar_texts=TEXTS_GROUP_1)
    # print("len(TEXTS_GROUP_1) :", len(TEXTS_GROUP_1))
    # **********************************************************************************************

    # Group 2 ************************************************************************************
    print("label_selected :", label_selected)
    if len(CLASSIFIER_LIST) > 0 and (label_selected is not None and label_selected != "" and label_selected != "None"):
        utils.get_top_predictions(selected_class=label_selected,
                                  fitted_classifier=CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  tweet_data_df=consolidated_disaster_tweet_data_df,
                                  texts_list=TEXTS_LIST,
                                  top=PREDICTIONS_NUMBER,
                                  cutoff_proba=PREDICTIONS_PROBABILITY,
                                  fat_df=False,
                                  y_classes=Y_CLASSES,
                                  verbose=PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=True,
                                  similar_texts=TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template("text_labeling_1.html",
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message="Text selected",
                           page_number=page_number,
                           y_classes=Y_CLASSES,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=TEXTS_GROUP_1,
                           texts_group_2=TEXTS_GROUP_2,
                           texts_group_3=TEXTS_GROUP_3,
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)


@app.route("/go_to_page", methods=["GET", "POST"])
def go_to_page():
    page_number = int(request.args.get("page_number", None))
    # print("page_number :", page_number)
    label_selected = request.args.get("label_selected", None)
    return render_template("text_labeling_1.html",
                           selected_text_id="None",
                           selected_text="Select a text below to begin labeling.",
                           label_selected=label_selected,
                           info_message="",
                           page_number=page_number,
                           y_classes=Y_CLASSES,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=[],
                           texts_group_2=TEXTS_GROUP_2,
                           texts_group_3=TEXTS_GROUP_3,
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)


@app.route("/single_text", methods=["POST"])
def single_text():
    # if request.method == "GET":
    #     if len(TEXTS_LIST) > 0:
    #         # return jsonify(TEXTS_LIST)
    #         id = request.form["selected_text_id"]
    #         text = request.form["selected_text"]
    #         return render_template("text_labeling_1.html",
    #                                selected_text_id=id,
    #                                selected_text=text,
    #                                info_message="GET",
    #                                texts_list=jsonify(TEXTS_LIST),
    #                                texts_group_1=TEXTS_GROUP_1,
    #                                texts_group_2=TEXTS_GROUP_2,
    #                                total_summary=TOTAL_SUMMARY,
    #                                label_summary=LABEL_SUMMARY)
    #     else:
    #         "Nothing Found", 404
    if request.method == "POST":
        print(f"In 'single_text()'")
        # print(f"selected_label_single : '{request.form.get('selected_label_single')}'")
        new_id = request.form["selected_text_id"]
        new_text = request.form["selected_text"]
        page_number = int(request.form["page_number"])
        new_label = request.form["selected_label_single"]
        info_message = ""

        print(f"new_id-{new_id}, new_text-{new_text}, page_number-{page_number}, new_label :", new_label)

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
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=TEXTS_GROUP_2,
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)
        else:
            # old_obj = {"id": new_id, "text": new_text, "label": "-"}
            new_obj = {"id": new_id, "text": new_text, "label": new_label}

            # utils.update_texts_list(texts_list=TEXTS_LIST,
            #                         sub_list_limit=TABLE_LIMIT,
            #                         old_obj_lst=[old_obj],
            #                         new_obj_lst=[new_obj],
            #                         texts_list_list=TEXTS_LIST_LIST)

            utils.update_texts_list_by_id(texts_list=TEXTS_LIST,
                                          sub_list_limit=TABLE_LIMIT,
                                          updated_obj_lst=[new_obj],
                                          texts_list_list=TEXTS_LIST_LIST,
                                          update_in_place=False)

            utils.generate_summary(text_lists=TEXTS_LIST,
                                   first_labeling_flag=FIRST_LABELING_FLAG,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)

            # Group 2 **************************************************************************************
            # print("len(CLASSIFIER_LIST) :", len(CLASSIFIER_LIST))
            utils.fit_classifier(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                 corpus_text_ids=CORPUS_TEXT_IDS,
                                 texts_list_labeled=[new_obj],
                                 y_classes=Y_CLASSES,
                                 classifier_list=CLASSIFIER_LIST)

            # print("new_label :", new_label)
            if len(CLASSIFIER_LIST) > 0 and \
                    (new_label is not None and new_label != "" and new_label != "None"):
                utils.get_top_predictions(selected_class=new_label,
                                          fitted_classifier=CLASSIFIER_LIST[0],
                                          sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                          corpus_text_ids=CORPUS_TEXT_IDS,
                                          tweet_data_df=consolidated_disaster_tweet_data_df,
                                          texts_list=TEXTS_LIST,
                                          top=PREDICTIONS_NUMBER,
                                          cutoff_proba=PREDICTIONS_PROBABILITY,
                                          fat_df=False,
                                          y_classes=Y_CLASSES,
                                          verbose=PREDICTIONS_VERBOSE,
                                          exclude_already_labeled=True,
                                          similar_texts=TEXTS_GROUP_2)

            # **********************************************************************************************

            # print(f">> Testing - new_label-'{new_label}'")
            return render_template("text_labeling_1.html",
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=new_label,
                                   info_message="Label assigned",
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=[], # TEXTS_GROUP_1,
                                   texts_group_2=TEXTS_GROUP_2,
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)


@app.route("/grouped_1_texts", methods=["POST"])
def grouped_1_texts():
    if request.method == "POST":
        page_number = int(request.form["page_number"])
        new_id = request.form["selected_text_id"]
        new_text = request.form["selected_text"]

        selected_label_group1 = request.form["selected_label_group1"]
        # print("selected_label_group1 :", selected_label_group1)
        # print()

        info_message = ""
        if (selected_label_group1 == "None") or (selected_label_group1 is None) or (selected_label_group1 == "") or \
                (len(TEXTS_GROUP_1) == 0):
            if selected_label_group1 == "":
                info_message += f"Select a 'Label'."

            if len(TEXTS_GROUP_1) == 0:
                info_message += "\n" + f"Select a 'Text ID'."

            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   info_message=info_message,
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=[],
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)

        else:
            texts_group_1_updated = copy.deepcopy(TEXTS_GROUP_1)
            for obj in texts_group_1_updated:
                obj["label"] = selected_label_group1

            utils.update_texts_list_by_id(texts_list=TEXTS_LIST,
                                          sub_list_limit=TABLE_LIMIT,
                                          updated_obj_lst=texts_group_1_updated,
                                          texts_list_list=TEXTS_LIST_LIST,
                                          update_in_place=False)

            utils.generate_summary(text_lists=TEXTS_LIST,
                                   first_labeling_flag=FIRST_LABELING_FLAG,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)

            # Group 2 **************************************************************************************
            # print("len(CLASSIFIER_LIST) :", len(CLASSIFIER_LIST))
            utils.fit_classifier(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                 corpus_text_ids=CORPUS_TEXT_IDS,
                                 texts_list_labeled=texts_group_1_updated,
                                 y_classes=Y_CLASSES,
                                 classifier_list=CLASSIFIER_LIST)

            if len(CLASSIFIER_LIST) > 0 and \
                    (selected_label_group1 is not None and
                     selected_label_group1 != "" and
                     selected_label_group1 != "None"):
                utils.get_top_predictions(selected_class=selected_label_group1,
                                          fitted_classifier=CLASSIFIER_LIST[0],
                                          sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                          corpus_text_ids=CORPUS_TEXT_IDS,
                                          tweet_data_df=consolidated_disaster_tweet_data_df,
                                          texts_list=TEXTS_LIST,
                                          top=PREDICTIONS_NUMBER,
                                          cutoff_proba=PREDICTIONS_PROBABILITY,
                                          fat_df=False,
                                          y_classes=Y_CLASSES,
                                          verbose=PREDICTIONS_VERBOSE,
                                          exclude_already_labeled=True,
                                          similar_texts=TEXTS_GROUP_2)
            # **********************************************************************************************

            return render_template("text_labeling_1.html",
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=selected_label_group1,
                                   info_message="Labels assigned to group",
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=[], # texts_group_1_updated,
                                   texts_group_2=TEXTS_GROUP_2,
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)


@app.route("/grouped_2_texts", methods=["POST"])
def grouped_2_texts():
    if request.method == "POST":
        page_number = int(request.form["page_number"])
        new_id = request.form["selected_text_id"]
        new_text = request.form["selected_text"]

        selected_label_group2 = request.form["selected_label_group2"]
        print("selected_label_group2 :", selected_label_group2)
        print()

        info_message = ""
        if (selected_label_group2 is None) or (selected_label_group2 == "None") or (selected_label_group2 == "") or \
                (len(TEXTS_GROUP_2) == 0):
            if selected_label_group2 == "":
                info_message += f"Select a 'Label'."

            if len(TEXTS_GROUP_2) == 0:
                info_message += "\n" + f"Select a 'Text ID'."

            return render_template("text_labeling_1.html",
                                   selected_text_id=new_id,
                                   selected_text=new_text,
                                   label_selected=selected_label_group2,
                                   info_message=info_message,
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=TEXTS_GROUP_1,
                                   texts_group_2=[],
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)

        else:
            texts_group_2_updated = copy.deepcopy(TEXTS_GROUP_2)
            for obj in texts_group_2_updated:
                obj["label"] = selected_label_group2

            utils.update_texts_list_by_id(texts_list=TEXTS_LIST,
                                          sub_list_limit=TABLE_LIMIT,
                                          updated_obj_lst=texts_group_2_updated,
                                          texts_list_list=TEXTS_LIST_LIST,
                                          update_in_place=False)

            utils.generate_summary(text_lists=TEXTS_LIST,
                                   first_labeling_flag=FIRST_LABELING_FLAG,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)

            # Group 2 **************************************************************************************
            print("len(CLASSIFIER_LIST) :", len(CLASSIFIER_LIST))
            utils.fit_classifier(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                 corpus_text_ids=CORPUS_TEXT_IDS,
                                 texts_list_labeled=texts_group_2_updated,
                                 y_classes=Y_CLASSES,
                                 classifier_list=CLASSIFIER_LIST)

            if len(selected_label_group2) > 0 and \
                    (selected_label_group2 is not None and
                     selected_label_group2 != "" and
                     selected_label_group2 != "None"):
                utils.get_top_predictions(selected_class=selected_label_group2,
                                          fitted_classifier=CLASSIFIER_LIST[0],
                                          sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                          corpus_text_ids=CORPUS_TEXT_IDS,
                                          tweet_data_df=consolidated_disaster_tweet_data_df,
                                          texts_list=TEXTS_LIST,
                                          top=PREDICTIONS_NUMBER,
                                          cutoff_proba=PREDICTIONS_PROBABILITY,
                                          fat_df=False,
                                          y_classes=Y_CLASSES,
                                          verbose=PREDICTIONS_VERBOSE,
                                          exclude_already_labeled=True,
                                          similar_texts=TEXTS_GROUP_2)
            # **********************************************************************************************

            return render_template("text_labeling_1.html",
                                   selected_text_id="None", # new_id,
                                   selected_text="Select a text below to label.", # new_text,
                                   label_selected=selected_label_group2,
                                   info_message="Labels assigned to group",
                                   page_number=page_number,
                                   y_classes=Y_CLASSES,
                                   total_pages=TOTAL_PAGES,
                                   texts_list=TEXTS_LIST_LIST[page_number],
                                   texts_group_1=[], # texts_group_1_updated,
                                   texts_group_2=TEXTS_GROUP_2,
                                   texts_group_3=TEXTS_GROUP_3,
                                   total_summary=TOTAL_SUMMARY,
                                   label_summary=LABEL_SUMMARY)


@app.route("/text/<id>", methods=["GET", "PUT", "DELETE"])
def single_record(id):
    if request.method == "GET":
        for text in TEXTS_LIST:
            if text in TEXTS_LIST:
                return jsonify(text)
            pass

    if request.method == "PUT":
        for text in TEXTS_LIST:
            if text["id"] == id:
                text["text"] = request.form["selected_text"]
                text["label"] = request.form["assigned_label"]
                updated_text = {"id": id, "text": text["text"], "label": text["label"]}
                return jsonify(updated_text)

    if request.method == "DELETE":
        for index, book in enumerate(TEXTS_LIST):
            if text["id"] == id:
                TEXTS_LIST.pop(index)
                return jsonify(TEXTS_LIST)


@app.route('/export_records', methods=['GET', 'POST'])
def export_records():
    texts_df = pd.DataFrame.from_dict(TEXTS_LIST)
    export_response = make_response(texts_df.to_csv(index=False))
    export_response.headers["Content-Disposition"] = "attachment; filename=export_records.csv"
    export_response.headers["Content-Type"] = "text/csv"
    return export_response


@app.route('/label_selected', methods=['GET', 'POST'])
def label_selected():
    # print(">> In 'label_selected'")
    label_selected = request.args.get("label_selected")
    # print(f"label_selected : {label_selected}")

    selected_text_id = request.args.get("selected_text_id")
    selected_text = request.args.get("selected_text")
    page_number = int(request.args.get("page_number"))

    print(f"selected_text_id : {selected_text_id}, selected_text : {selected_text}, page_number : {page_number}")


    # Group 1 ************************************************************************************
    if selected_text_id == "None":
        TEXTS_GROUP_1.clear()
    else:
        similarities_series = utils.get_all_similarities_one_at_a_time(sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                                                       corpus_text_ids=CORPUS_TEXT_IDS,
                                                                       text_id=selected_text_id,
                                                                       keep_original=KEEP_ORIGINAL)

        utils.get_top_similar_texts(all_texts_json=TEXTS_LIST,
                                    similarities_series=similarities_series,
                                    top=GROUP_1_KEEP_TOP,
                                    exclude_already_labeled=GROUP_1_EXCLUDE_ALREADY_LABELED,
                                    similar_texts=TEXTS_GROUP_1)
        # print("len(TEXTS_GROUP_1) :", len(TEXTS_GROUP_1))
    # **********************************************************************************************

    # Group 2 ************************************************************************************
    if len(CLASSIFIER_LIST) > 0:
        utils.get_top_predictions(selected_class=label_selected,
                                  fitted_classifier=CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  tweet_data_df=consolidated_disaster_tweet_data_df,
                                  texts_list=TEXTS_LIST,
                                  top=PREDICTIONS_NUMBER,
                                  cutoff_proba=PREDICTIONS_PROBABILITY,
                                  fat_df=False,
                                  y_classes=Y_CLASSES,
                                  verbose=PREDICTIONS_VERBOSE,
                                  exclude_already_labeled=True,
                                  similar_texts=TEXTS_GROUP_2)
    # **********************************************************************************************

    return render_template("text_labeling_1.html",
                           selected_text_id=selected_text_id,
                           selected_text=selected_text,
                           label_selected=label_selected,
                           info_message=label_selected,
                           page_number=page_number,
                           y_classes=Y_CLASSES,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=TEXTS_GROUP_1,
                           texts_group_2=TEXTS_GROUP_2,
                           texts_group_3=TEXTS_GROUP_3,
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)


@app.route('/generate_difficult_texts', methods=['POST'])
def generate_difficult_texts():
    page_number = int(request.form["page_number"])
    id = request.form["selected_text_id"]
    text = request.form["selected_text"]
    label_selected = request.form["selected_label"]

    # Group 3 ************************************************************************************
    if len(CLASSIFIER_LIST) > 0:
        utils.get_all_predictions(fitted_classifier=CLASSIFIER_LIST[0],
                                  sparse_vectorized_corpus=VECTORIZED_CORPUS,
                                  corpus_text_ids=CORPUS_TEXT_IDS,
                                  tweet_data_df=consolidated_disaster_tweet_data_df,
                                  top=GROUP_3_KEEP_TOP,
                                  fat_df=False,
                                  y_classes=Y_CLASSES,
                                  verbose=PREDICTIONS_VERBOSE,
                                  round_to=1,
                                  format_as_percentage=True,
                                  similar_texts=TEXTS_GROUP_3)
        info_message = label_selected
    # **********************************************************************************************
    else:
        info_message = "Label more texts then try generating the difficult text list."
    return render_template("text_labeling_1.html",
                           selected_text_id=id,
                           selected_text=text,
                           label_selected=label_selected,
                           info_message=info_message,
                           page_number=page_number,
                           y_classes=Y_CLASSES,
                           total_pages=TOTAL_PAGES,
                           texts_list=TEXTS_LIST_LIST[page_number],
                           texts_group_1=TEXTS_GROUP_1,
                           texts_group_2=TEXTS_GROUP_2,
                           texts_group_3=TEXTS_GROUP_3,
                           total_summary=TOTAL_SUMMARY,
                           label_summary=LABEL_SUMMARY)



if __name__ == "__main__":
    app.run() # host="127.0.0.1:5000"

