import json
import random
from collections import defaultdict, OrderedDict
import jinja2
from query_llama import query_llama, load_llama
from task_eval import most_similar_answer


def query_bi(need_qtype, qtypes, pred_list, text_list, qw_list, generator, data, dataset):
    select_pred_list = [pred for pred, qtype in zip(pred_list, qtypes) if qtype == need_qtype]
    if len(select_pred_list)==0:
        return pred_list
    select_text_list = [text for text, qtype in zip(text_list, qtypes) if qtype == need_qtype]
    select_qw_list = [qw for qw, qtype in zip(qw_list, qtypes) if qtype == need_qtype]
    prompt_list = []
    for text, qw, pred in zip(select_text_list, select_qw_list, select_pred_list):
        if need_qtype == 0 or need_qtype == 6:
            # prompt = f'This is a material science literature text: {text} Please tell me the type of "{qw}" in the literature text, all types are in Options. Options: {"B-" + pred + ", I-" + pred}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            template_loader = jinja2.FileSystemLoader(searchpath="./dataset/templates")
            template_env = jinja2.Environment(loader=template_loader)
            bio_template = template_env.get_template("BIO")
            question_describe = f'Based on the above classes, please determine which entity class the "{qw}" in the above text belongs to. Use the following format to answer: ```Answer: [ONLY the entity class name that you think most correct; not a complete sentence]```'
            prompt = bio_template.render(text=text, entity=pred, question_describe=question_describe)
            prompt_list.append(prompt)
    output_list, _, _ = query_llama(generator, prompt_list, data, True)
    new_select_pred_list = []
    for res,pred in zip(output_list,select_pred_list):
        if "b-"+pred.lower() in res.lower():
            new_select_pred_list.append("b-"+pred.lower())
        else:
            new_select_pred_list.append("i-"+pred.lower())
    n = 0
    for i, qtype in enumerate(qtypes):
        if qtype == need_qtype:
            pred_list[i] = new_select_pred_list[n]
            n += 1
    return pred_list


def query_for_bi(dataset, qtypes, pred_list, text_list, qw_list, generator, data):
    if dataset == 0:
        pred_list = query_bi(0, qtypes, pred_list, text_list, qw_list, generator, data, dataset)
    if dataset == 1:
        pred_list = query_bi(0, qtypes, pred_list, text_list, qw_list, generator, data, dataset)
        pred_list = query_bi(6, qtypes, pred_list, text_list, qw_list, generator, data, dataset)
    return pred_list


def remove_duplicates_preserve_order(lst):
    return list(OrderedDict.fromkeys(lst))


def remove_label_bio(label):
    if "-" in label and label.lower().split("-")[0]=="b" or label.lower().split("-")[0]=="i":
            return label[2:]
    else:
        return label


def get_class_name(label):
    if "-" in label:
        label = "".join([word.capitalize() for word in label.split("-")])
    elif "_" in label:
        label = "".join([word.capitalize() for word in label.split("_")])
    else:
        label = label.capitalize()
    return label


def text_suitable(text, prompt, max_seq_len, max_text_len):
    over_len = len(prompt) - (max_seq_len - 50)
    if over_len > 0:
        text = text[:max_text_len - over_len]
    return text


def shuffle_list(list1, list2, list3, list4):
    zipped = list(zip(list1, list2, list3, list4))
    random.shuffle(zipped)
    list1, list2, list3, list4 = zip(*zipped)
    return list(list1), list(list2), list(list3),  list(list4)


def load_class_defs(status, code_path, dataset):
    if status == 0:
        class_defs = json.load(open(code_path + f'/{dataset}_class_base.json'))
    elif status == 1:
        class_defs = json.load(open(code_path + f'/{dataset}_class_with_description.json'))
    elif status == 2:
        class_defs = json.load(open(code_path + f'/{dataset}_class_with_description_properties.json'))

    return class_defs


def save_new_label_descriptions(label_description_dict, dataset, data):
    with open(f"{data.learning_description_path}/{dataset}_description.json", "w") as json_file:
        json.dump(label_description_dict, json_file, indent=4)


def load_learned_label_descriptions(dataset, data):
    class_defs = json.load(open(data.learning_description_path + f'/{dataset}_description.json'))
    return class_defs
