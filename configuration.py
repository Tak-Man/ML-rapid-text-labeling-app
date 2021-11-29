import pickle
import os


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


FIXED_DATASETS = [{"name": "Disaster Tweets Dataset",
                   "description": """
                                    The HumAID Twitter dataset consists of several thousands of manually annotated 
                                    tweets that has been collected during 19 major natural disaster events 
                                    including earthquakes, hurricanes, wildfires, and floods, which happened 
                                    from 2016 to 2019 across different parts of the World. The annotations in 
                                    the provided datasets consists of following humanitarian categories. The 
                                    dataset consists only english tweets and it is the largest dataset for 
                                    crisis informatics so far.
                                  """,
                   "url": "https://crisisnlp.qcri.org/humaid_dataset.html#"},

                  {"name": "Disaster Tweets Dataset with 'Other'",
                   "description": """
                                    The same dataset as the 'Disaster Tweets Dataset' but the label 'Other'
                                    is included to account for texts that do not neatly fall under the labels
                                    'Earthquake', 'Hurricane', 'Fire', 'Flood'.
                                   """,
                   "url": "https://crisisnlp.qcri.org/humaid_dataset.html#"}]

SAVES_DIR = "./output/"
MAX_CONTENT_PATH = 10000
UPLOAD_FOLDER = "./output/upload/"
TEXTS_LIMIT = 100000
TABLE_LIMIT = 50
MAX_FEATURES = 800
KEEP_ORIGINAL = True
GROUP_1_EXCLUDE_ALREADY_LABELED = True
GROUP_2_EXCLUDE_ALREADY_LABELED = True
RND_STATE = 2584
PREDICTIONS_PROBABILITY = 0.85
PREDICTIONS_VERBOSE = False
SIMILAR_TEXT_VERBOSE = False
FIT_CLASSIFIER_VERBOSE = False
GROUP_3_KEEP_TOP = 50
FIRST_LABELING_FLAG = True
FULL_FIT_IF_LABELS_GOT_OVERRIDDEN = False
FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS = False
INITIALIZE_FLAGS = [{0: 1, 1: 0, 2: 0, 3: 0}]
GROUP_1_KEEP_TOP = [10]
PREDICTIONS_NUMBER = [50]
SEARCH_RESULT_LENGTH = [0]
OVERALL_QUALITY_SCORE = ["-"]
CONFIRM_LABEL_ALL_TEXTS_COUNTS = [0]
LABELS_GOT_OVERRIDDEN_FLAG = [False]
SEARCH_EXCLUDE_ALREADY_LABELED = [True]

TEXT_ID_COL = []
TEXT_VALUE_COL = []
CORPUS_TEXT_IDS = []
PREP_DATA_MESSAGE1 = []
PREP_DATA_MESSAGE2 = []
PREPARED_DATA_LIST = []
DATASETS_AVAILABLE = []
DATASET_NAME = []
DATASET_URL = []
TOTAL_SUMMARY = []
LABEL_SUMMARY = []
RECOMMENDATIONS_SUMMARY = []
TEXTS_LIST_LABELED = []
TEXTS_GROUP_1 = []
TEXTS_GROUP_2 = []
TEXTS_GROUP_3 = []
CLASSIFIER_LIST = []
VECTORIZER_LIST = []
SEARCH_MESSAGE = ["Displaying all texts"]
TEXTS_LIST = []
TEXTS_LIST_FULL = []
TEXTS_LIST_LIST = []
TEXTS_LIST_LIST_FULL = []
TOTAL_PAGES_FULL = []
ADJ_TEXT_IDS = []
TOTAL_PAGES = []
VECTORIZED_CORPUS = []
Y_CLASSES = []
SHUFFLE_BY = []
NUMBER_UNLABELED_TEXTS = []
CLICK_LOG = []
VALUE_LOG = []
DATE_TIME = []
HTML_CONFIG_TEMPLATE = []
LABEL_SUMMARY_STRING = []


