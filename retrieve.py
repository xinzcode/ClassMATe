import json

from query_llama import query_llama
from utils import remove_label_bio, remove_duplicates_preserve_order


def query(generator, data, text, label, qtype):
    if qtype == 1:
        prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of "glass science paragraph" from the guideline, noting that if it is not mentioned, "The paragraph pertains to glass science." will be used as the definition. Use the following format to answer: ```Answer: the definition```'
    elif qtype == 5:
        prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of "experimental fact sentence" from the guideline, noting that if it is not mentioned, "The sentences that describe a relevant solid oxide fuel cell experiment." will be used as the definition. Use the following format to answer: ```Answer: the definition```'
    elif qtype == 4:
        prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of "{label}" from the guideline, including example description. Use the following format to answer: ```Answer: the definition```'
    elif qtype == 2:
        # prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of "{label}" from the guideline, including example description. Use the following format to answer: ```Answer: the definition```'
        prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of the relation extraction label "{label}" from the guideline, including example description. If it is too long please summarise. If the “{label}” is not mentioned, provide a definition you think might fit based on the content of the guideline. Use the following format to answer: ```Answer: the definition```'
    else:
        prompt = f'This is an annotation guideline for Materials Science Text Mining Tasks: \n"{text}" \nPlease provide the definition of "{label}" from the guideline, including example description. If it is too long please summarise. If the “{label}” is not mentioned, provide a definition you think might fit based on the content of the guideline. Use the following format to answer: ```Answer: the definition```'
    output_list, _, _ = query_llama(generator, [prompt], data, False)
    description = output_list[0]
    # print(description)
    if "Answer:" in description:
        description = description.split("Answer:")[1].strip()
    return description


def retrieve(res_dict, qtypes, dataset, generator, data):
    with open(f'dataset/guidelines/txt_cleaned/{dataset}.txt', "r", encoding="utf-8") as txt_file:
        text = txt_file.read()

    label_description_dict = {}

    for qtype in list(set(qtypes)):
        # NER
        if qtype == 0:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['t_type_set']])
            for label in candidates:
                label_description_dict[label] = query(generator, data, text, label, qtype)

        # RC
        if qtype == 2:
            candidates = res_dict['r_type_set']
            for label in candidates:
                label_description_dict[label] = query(generator, data, text, label, qtype)

        # EE
        if qtype == 3:
            candidates = res_dict['e_role_set']
            for label in candidates:
                label_description_dict[label] = query(generator, data, text, label, qtype)
                label_description_dict["event"] = ""

        # SF
        if qtype == 6:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['sf_type_set']])
            for label in candidates:
                label_description_dict[label] = query(generator, data, text, label, qtype)

        # PC (no candidates)
        if qtype == 1:
            label_description_dict["pc"] = query(generator, data, text, "", qtype)

        # SAR
        if qtype == 4:
            candidates = res_dict['sar_set']
            for label in candidates:
                label_description_dict[label] = query(generator, data, text, label, qtype)

        # SC (no candidates)
        if qtype == 5:
            label_description_dict["sc"] = query(generator, data, text, "", qtype)

    with open(f"{data.gold_description_path}/{dataset}_description.json", "w") as json_file:
        json.dump(label_description_dict, json_file, indent=4)

