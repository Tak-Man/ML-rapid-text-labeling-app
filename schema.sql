DROP TABLE IF EXISTS texts;

CREATE TABLE texts (
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