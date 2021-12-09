DROP TABLE IF EXISTS decimalValues;

CREATE TABLE decimalValues (
    name TEXT NOT NULL UNIQUE,
    value DECIMAL(10, 5) NOT NULL
);

INSERT INTO decimalValues (name, value) VALUES ("OVERALL_QUALITY_SCORE_DECIMAL", 0.0);
INSERT INTO decimalValues (name, value) VALUES ("OVERALL_QUALITY_SCORE_DECIMAL_PREVIOUS", 0.0);

DROP TABLE IF EXISTS texts;

CREATE TABLE texts (
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL
);

DROP TABLE IF EXISTS prepTexts;

CREATE TABLE prepTexts (
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL
);

DROP TABLE IF EXISTS searchResults;

CREATE TABLE searchResults (
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL
);

DROP TABLE IF EXISTS clickRecord;

CREATE TABLE clickRecord (
    click_id TEXT NOT NULL,
    click_location TEXT NOT NULL,
    click_type TEXT NOT NULL,
    click_object TEXT NOT NULL,
    click_date_time TEXT NOT NULL
);

DROP TABLE IF EXISTS valueRecord;

CREATE TABLE valueRecord (
    click_id TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value TEXT NOT NULL
);


DROP TABLE IF EXISTS group1Texts;

CREATE TABLE group1Texts (
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL
);

DROP TABLE IF EXISTS group2Texts;

CREATE TABLE group2Texts (
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL
);

DROP TABLE IF EXISTS fixedDatasets;

CREATE TABLE fixedDatasets (
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    url TEXT NOT NULL
);

INSERT INTO fixedDatasets (name, description, url)
VALUES ('Disaster Tweets Dataset', 'The HumAID Twitter dataset consists of several thousands of manually annotated tweets that has been collected during 19 major natural disaster events including earthquakes, hurricanes, wildfires, and floods, which happened from 2016 to 2019 across different parts of the World. The annotations in the provided datasets consists of following humanitarian categories. The dataset consists only english tweets and it is the largest dataset for crisis informatics so far.',
        'https://crisisnlp.qcri.org/humaid_dataset.html#');

INSERT INTO fixedDatasets (name, description, url)
VALUES ("Disaster Tweets Dataset with 'Other'",
        "The same dataset as the 'Disaster Tweets Dataset' but the label 'Other' is included to account for texts that do not neatly fall under the labels 'Earthquake', 'Hurricane', 'Fire', 'Flood'.",
        'https://crisisnlp.qcri.org/humaid_dataset.html#');

DROP TABLE IF EXISTS availableDatasets;

CREATE TABLE availableDatasets (
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    url TEXT NOT NULL
);

DROP TABLE IF EXISTS yClasses;

CREATE TABLE yClasses (
    classId INTEGER,
    className TEXT NOT NULL
);


DROP TABLE IF EXISTS variables;

CREATE TABLE variables (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT INTO variables (name, value) VALUES ("SAVES_DIR", "./output/");
INSERT INTO variables (name, value) VALUES ("TEXTS_LIMIT", 100000);
INSERT INTO variables (name, value) VALUES ("TABLE_LIMIT", 50);
INSERT INTO variables (name, value) VALUES ("MAX_FEATURES", 800);
INSERT INTO variables (name, value) VALUES ("KEEP_ORIGINAL", "True");
INSERT INTO variables (name, value) VALUES ("GROUP_1_EXCLUDE_ALREADY_LABELED", "True");
INSERT INTO variables (name, value) VALUES ("GROUP_2_EXCLUDE_ALREADY_LABELED", "True");
INSERT INTO variables (name, value) VALUES ("RND_STATE", 2584);
INSERT INTO variables (name, value) VALUES ("PREDICTIONS_PROBABILITY", 0.85);
INSERT INTO variables (name, value) VALUES ("PREDICTIONS_VERBOSE", "False");
INSERT INTO variables (name, value) VALUES ("SIMILAR_TEXT_VERBOSE", "False");
INSERT INTO variables (name, value) VALUES ("FIT_CLASSIFIER_VERBOSE", "False");
INSERT INTO variables (name, value) VALUES ("FIRST_LABELING_FLAG", "True");
INSERT INTO variables (name, value) VALUES ("FULL_FIT_IF_LABELS_GOT_OVERRIDDEN", "False");
INSERT INTO variables (name, value) VALUES ("FORCE_FULL_FIT_FOR_DIFFICULT_TEXTS", "False");
INSERT INTO variables (name, value) VALUES ("PREDICTIONS_NUMBER", 50);
INSERT INTO variables (name, value) VALUES ("SEARCH_RESULT_LENGTH", 0);
INSERT INTO variables (name, value) VALUES ("GROUP_1_KEEP_TOP", 10);
INSERT INTO variables (name, value) VALUES ("OVERALL_QUALITY_SCORE", "-");
INSERT INTO variables (name, value) VALUES ("GROUP_3_KEEP_TOP", 50);
INSERT INTO variables (name, value) VALUES ("CONFIRM_LABEL_ALL_TEXTS_COUNTS", 0);
INSERT INTO variables (name, value) VALUES ("LABELS_GOT_OVERRIDDEN_FLAG", "False");
INSERT INTO variables (name, value) VALUES ("ALLOW_SEARCH_TO_OVERRIDE_EXISTING_LABELS", "No");
INSERT INTO variables (name, value) VALUES ("UPDATE_TEXTS_IN_PLACE", "False");


DROP TABLE IF EXISTS initializeFlags;

CREATE TABLE initializeFlags (
    name INT PRIMARY KEY,
    value INT NOT NULL
);

INSERT INTO initializeFlags (name, value) VALUES (0, 1);
INSERT INTO initializeFlags (name, value) VALUES (1, 0);
INSERT INTO initializeFlags (name, value) VALUES (2, 0);
INSERT INTO initializeFlags (name, value) VALUES (3, 0);


DROP TABLE IF EXISTS totalSummary;

CREATE TABLE totalSummary (
    name INT PRIMARY KEY,
    number INT NOT NULL,
    percentage TEXT NOT NULL
);


DROP TABLE IF EXISTS labelSummary;

CREATE TABLE labelSummary (
    name INT PRIMARY KEY,
    number INT NOT NULL,
    percentage TEXT NOT NULL
);
