import sys
sys.path.append("../ML-rapid-text-labeling-app")
import web_app_utilities as utils
from datetime import datetime
import copy


if __name__ == "__main__":
    start_time = datetime.now()
    print(">> Start time :", start_time.strftime("%m/%d/%Y %H:%M:%S"), "*"*100)

    consolidated_disaster_tweet_data_df = \
        utils.get_disaster_tweet_demo_data(source_file="../data/consolidated_disaster_tweet_data.tsv")

    all_texts_json, adj_text_ids = \
        utils.convert_demo_data_into_list_json(consolidated_disaster_tweet_data_df, limit=100000)
    print("all_texts_json :")
    print(all_texts_json[:5])
    print()

    # *****************************************************************************************************************
    test_labels = ["Earthquake", "Flood"]
    update_ids = ['798262465234542592', '771464543796985856', '797835622471733248',
                  '798021801540321280', '798727277794033664']
    for test_label in test_labels:
        print("test_label :", test_label)
        texts_group_updated = []
        for text in all_texts_json:
            if text["id"] in update_ids:
                texts_group_updated.append({"id": text["id"], "text": text["text"], "label": test_label})

        print("texts_group_updated :")
        print(texts_group_updated)

        updated_texts_list, updated_texts_list_list, overridden = \
            utils.update_texts_list_by_id(texts_list=all_texts_json,
                                          sub_list_limit=50,
                                          updated_obj_lst=texts_group_updated,
                                          texts_list_list=[],
                                          labels_got_overridden_flag=[],
                                          update_in_place=True)

        print("updated_texts_list :")
        print(updated_texts_list[:5])
        print("updated_texts_list :")
        print(updated_texts_list[-5:])
        print("overridden :", overridden)
        print()
    # *****************************************************************************************************************

    end_time = datetime.now()
    duration = end_time - start_time

    print(">> End time :", end_time.strftime("%m/%d/%Y @ %H:%M:%S"), "*"*100)
    print(">> Duration :", duration, "*"*100)