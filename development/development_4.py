import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv")
    print("consolidated_disaster_tweet_data_df :")
    print(consolidated_disaster_tweet_data_df.head())
    print()

    all_texts_json, adj_text_ids = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=100000)
    print("all_texts_json :")
    print(all_texts_json[:5])
    print(all_texts_json[-5:])
    print()

    update_corpus_text_ids = ['798262465234542592', '771464543796985856', '798021801540321280']

    corpus_text_ids = [str(x) for x in consolidated_disaster_tweet_data_df["tweet_id"].values]

    # *****************************************************************************************************************
    texts_list_list = []
    updated_obj_lst = []

    updated_obj_lst = [{"id": '798262465234542592', "text": "Blank text", "label": "Other Label"},
                       {"id": '771464543796985856', "text": "Blank text", "label": "Another Other Label"},
                       {"id": '798021801540321280', "text": "Blank text", "label": "Other Label"}]

    updated_texts_list, updated_texts_list_list = \
        utils.update_texts_list_by_id(texts_list=all_texts_json,
                                      sub_list_limit=10,
                                      updated_obj_lst=updated_obj_lst,
                                      texts_list_list=texts_list_list,
                                      update_in_place=True)
    print("updated_texts_list :")
    print(updated_texts_list[:5])
    print(updated_texts_list[-5:])
    # *****************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)