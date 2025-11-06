import copy
import json
import random
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from build_class import gen_class
from build_prompt import build_code_prompt
from query_llama import query_llama
from result_parse import llama_code_parse
from task_eval import most_similar_answer
from utils import query_for_bi, remove_duplicates_preserve_order, remove_label_bio, load_class_defs, \
    save_new_label_descriptions


def split_and_sample(dataset, data, text_list, qw_list, gold_list, qtypes, sample_size, seed):
    random.seed(seed)
    grouped_data = {}
    for text, qw, gold, qtype in zip(text_list, qw_list, gold_list, qtypes):
        if qtype not in grouped_data:
            grouped_data[qtype] = {'text': [], 'qw': [], 'gold': [], 'qtype': []}
        grouped_data[qtype]['text'].append(text)
        grouped_data[qtype]['qw'].append(qw)
        grouped_data[qtype]['gold'].append(gold)
        grouped_data[qtype]['qtype'].append(qtype)

    sampled_data = {}
    for qtype, data in grouped_data.items():
        # if qtype == 0:
        #     sample_size = 100

        n = len(data['text'])
        if n >= sample_size:
            indices = random.sample(range(n), sample_size)
        else:
            indices = range(n)
        sampled_data[qtype] = {
            'text': [data['text'][i] for i in indices],
            'qw': [data['qw'][i] for i in indices],
            'gold': [data['gold'][i] for i in indices],
            'qtype': [data['qtype'][i] for i in indices]
        }
    return sampled_data


def code_inference2(generator, dataset, class_defs, batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data):
    prompt_list = build_code_prompt(batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data.task_type_dict, class_defs, data.max_text_len, dataset, data.max_seq_len)
    output_list, output_logprobs_list, output_logprobs2_list = query_llama(generator, prompt_list, data, True)
    pred_list = llama_code_parse(output_list, batch_qtype_list, dataset)
    # pred_list_o = copy.deepcopy(pred_list)
    # pred_list = query_for_bi(dataset, batch_qtype_list, pred_list, batch_text_list, batch_qw_list, generator, data)
    return pred_list, output_logprobs_list, output_logprobs2_list


def gen_new_description(qtype, category, old_label_description, samples_qw_list, samples_text_list, samples_pred_list, samples_pred_logprob_list, samples_gold_list, data, generator):

    if qtype == 0:
        prompt = f'I am currently working on a prompt-based entity classification task for materials science text, below is a label ("{category}") and corresponding description, a set of uncertainty samples obtained based on the current label and description.\nPlease observe and learn from the given uncertainty samples to optimise the existing label description so that they can be better used to classify entities (note that the existing description may be inaccurate, if the there is a big gap with the sample you can generate a new one based on the Samples), and there should be only one new label description in total and no more than 50 words. Use the following format to answer: ```Answer: only the new description```.'
        prompt_label = f'\n\nLabel:\n{category} : {old_label_description}'
        prompt_sample = f'\n\nSamples:\n'
        for samples_qw, samples_text in zip(samples_qw_list, samples_text_list):
            prompt_sample += f'Context: {samples_text}\nEntity: {samples_qw}\n\n'

    elif qtype == 6:
        prompt = f'I am currently working on a prompt-based entity classification task for materials science text, below is a label ("{category}") and corresponding description, a set of uncertainty samples obtained based on the current label and description.\nPlease observe and learn from the given uncertainty samples to optimise the existing label description so that they can be better used to classify entities (note that the existing description may be inaccurate, if the there is a big gap with the sample you can generate a new one based on the Samples), and there should be only one new label description in total and no more than 50 words. Use the following format to answer: ```Answer: only the new description```.'
        prompt_label = f'\n\nLabel:\n{category} : {old_label_description}'
        prompt_sample = f'\n\nSamples:\n'
        for samples_qw, samples_text in zip(samples_qw_list, samples_text_list):
            prompt_sample += f'Context: {samples_text}\nEntity: {samples_qw}\n\n'

    if qtype == 3:
        prompt = f'I am currently working on a event parameter extraction task for materials science text, below is a label ("{category}") and corresponding description, a set of uncertainty samples obtained based on the current label and description.\nPlease observe and learn from the given uncertainty samples to optimise the existing label description so that they can be better used to extract entities (note that the existing description may be inaccurate, if the there is a big gap with the sample you can generate a new one based on the Samples), and there should be only one new label description in total and no more than 50 words. Use the following format to answer: ```Answer: only the new description```.'
        prompt_label = f'\n\nLabel:\n{category} : {old_label_description}'
        prompt_sample = f'\n\nSamples:\n'
        for samples_qw, samples_text in zip(samples_qw_list, samples_text_list):
            prompt_sample += f'Context: {samples_text}\nEntity: {samples_qw}\n\n'

    # rc
    elif qtype == 2:
        prompt = f'I am currently working on a relation classification task in materials science text, below is a relation label ("{category}") and corresponding description, a set of uncertainty samples obtained based on the current label (which should be of "{category}").\nPlease observe and learn from the given uncertainty samples to optimise the existing label description so that they can be better used to classify relations (note that the existing description may be inaccurate, if the there is a big gap with the sample you can generate a new one based on the Samples), and the new label description should not exceed 30 words. Use the following format to answer: ```Answer: only the new description```.'
        prompt_label = f'\n\nLabel:\n{category} : {old_label_description}'
        prompt_sample = f'\n\nSamples:\n'
        for samples_qw, samples_text in zip(samples_qw_list, samples_text_list):
            args1, args2 = samples_qw.split(",")
            prompt_sample += f'Context: {samples_text}\nEntity1: {args1}, Entity2: {args2}\nRelation: {category}\n\n'

    # sar
    elif qtype == 4:
        prompt = f'I am currently working on a synthesis action classification task for materials science text, below is a synthesis action label ("{category}") and corresponding description, a set of uncertainty samples obtained based on the current label (which should be of "{category}").\nPlease observe and learn from the given uncertainty samples to optimise the existing label description so that they can be better used to classify entities (note that the existing description may be inaccurate, if the there is a big gap with the sample you can generate a new one based on the Samples), and there should be only one new label description in total and no more than 50 words. Use the following format to answer: ```Answer: only the new description```.'
        prompt_label = f'\n\nLabel:\n{category} : {old_label_description}'
        prompt_sample = f'\n\nSamples:\n'
        for samples_qw, samples_text in zip(samples_qw_list, samples_text_list):
            prompt_sample += f'Context: {samples_text}\nEntity: {samples_qw}\n\n'

    # pc
    elif qtype == 1:
        prompt = f'I am currently working on a glass science text classification task, below is a label and corresponding description, a set of uncertainty samples obtained based on the current label and description.\nPlease observe and learn from each uncertainty sample given and optimize the existing label descriptions to be more accurate when reclassifying. Then aggregate based on these optimized label descriptions into one. The format of the final aggregated label description should be: ```GlassScienceText: the aggregated label description```.'
        prompt_label = f'\n\nLabel:\nGlassScienceText : {old_label_description}'
        prompt_sample = f'\n\nUncertainty Samples:\n'
        for samples_qw, samples_text, samples_gold in zip(samples_qw_list, samples_text_list, samples_gold_list):
            prompt_sample += f'Text: {samples_text}\n'
    # sc
    elif qtype == 5:
        prompt = f'I am currently working on a experimental fact text classification task, below is a label and corresponding description, a set of uncertainty samples obtained based on the current label and description.\nPlease observe and learn from each uncertainty sample given and optimize the existing label descriptions to be more accurate when reclassifying. Then aggregate based on these optimized label descriptions into one. The format of the final aggregated label description should be: ```ExperimentalFactText: the aggregated label description```.'
        prompt_label = f'\n\nLabel:\nExperimentalFactText : {old_label_description}'
        prompt_sample = f'\n\nUncertainty Samples:\n'
        for samples_qw, samples_text, samples_gold in zip(samples_qw_list, samples_text_list, samples_gold_list):
            prompt_sample += f'Text: {samples_text}\n'

    output_list, _, _ = query_llama(generator, [prompt+prompt_label+prompt_sample], data, False)
    description = output_list[0]
    print(description)
    if qtype == 1:
        if "GlassScienceText:" in description:
            description = description.split("GlassScienceText:")[-1].strip()
            if "\n" in description:
                description = description.split("\n")[0].strip()
    elif qtype == 5:
        if "ExperimentalFactText: " in description:
            description = description.split("ExperimentalFactText: ")[-1].strip()
            if "\n" in description:
                description = description.split("\n")[0].strip()
    # elif qtype == 0:
    #     if category+": " in description:
    #         description = description.split(category+": ")[-1].strip()
    #         if "\n" in description:
    #             description = description.split("\n")[0].strip()
    # elif qtype == 6:
    #     if category+": " in description:
    #         description = description.split(category+": ")[-1].strip()
    #         if "\n" in description:
    #             description = description.split("\n")[0].strip()
    #     if "**Aggregated Label Description:**\n" in description:
    #         description = description.split("**Aggregated Label Description:**\n")[-1].strip()
    #         if "\n" in description:
    #             description = description.split("\n")[0].strip()
    #     if "**Aggregated label description:**\n" in description:
    #         description = description.split("**Aggregated label description:**\n")[-1].strip()
    #         if "\n" in description:
    #             description = description.split("\n")[0].strip()
    else:
        if "Answer:" in description:
            description = description.split("Answer:")[1].strip()
            if "\n" in description:
                description = description.split("\n")[0].strip()
            if ":" in description:
                description = description.split(":")[-1].strip()

    print("New description: ", description)
    return description


def group_data_by_gold_label(top_n_data, qtype, res_dict):
    if qtype!=3:
        # 创建一个字典，按 'gold_list' 中的类别进行分组
        grouped_data = defaultdict(lambda: {
            'pred_list': [],
            'pred_logprob_list': [],
            'text_list': [],
            'qw_list': [],
            'gold_list': []
        })
        top_n_data['gold_list'] = [remove_label_bio(label) for label in top_n_data['gold_list']]
        # 遍历数据，将它们根据 gold_list 中的值进行分组
        for i in range(len(top_n_data['gold_list'])):
            gold_value = top_n_data['gold_list'][i]
            original_pred = top_n_data['pred_list'][i]
            if qtype == 0:
                candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['t_type_set']])
            elif qtype == 1 or qtype == 5:
                candidates = ['yes', 'no']
            elif qtype == 2:
                candidates = res_dict['r_type_set']
            elif qtype == 4:
                candidates = res_dict['sar_set']
            elif qtype == 6:
                candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['sf_type_set']])
            true_pred = most_similar_answer(original_pred, candidates)
            grouped_data[gold_value]['pred_list'].append(true_pred)
            grouped_data[gold_value]['pred_logprob_list'].append(top_n_data['pred_logprob_list'][i])
            grouped_data[gold_value]['text_list'].append(top_n_data['text_list'][i])
            grouped_data[gold_value]['qw_list'].append(top_n_data['qw_list'][i])
            grouped_data[gold_value]['gold_list'].append(top_n_data['gold_list'][i])
        filtered_grouped_data = {group: data for group, data in grouped_data.items() if len(data['gold_list']) >= 3}
        # # 打印分组后的数据
        # for category, samples in grouped_data.items():
        #     print(f"Category: {category}")
        #     print(f"  pred_list: {samples['pred_list']}")
        #     print(f"  pred_logprob_list: {samples['pred_logprob_list']}")
        #     print(f"  text_list: {samples['text_list']}")
        #     print(f"  qw_list: {samples['qw_list']}")
        #     print()
    else:
        grouped_data = defaultdict(lambda: {
            'pred_list': [],
            'pred_logprob_list': [],
            'text_list': [],
            'qw_list': [],
            'gold_list': []
        })
        candidates = res_dict['e_role_set']

        # 遍历数据，将它们根据 gold_list 中的值进行分组
        for i in range(len(top_n_data['gold_list'])):
            for candidate in candidates:
                if candidate in top_n_data['gold_list'][i]:
                    grouped_data[candidate]['pred_list'].append(top_n_data['pred_list'][i])
                    grouped_data[candidate]['pred_logprob_list'].append(top_n_data['pred_logprob_list'][i])
                    grouped_data[candidate]['text_list'].append(top_n_data['text_list'][i])
                    grouped_data[candidate]['qw_list'].append(top_n_data['qw_list'][i])
                    grouped_data[candidate]['gold_list'].append(top_n_data['gold_list'][i])
        filtered_grouped_data = {group: data for group, data in grouped_data.items() if len(data['gold_list']) >= 3}

    return filtered_grouped_data


def cal_uncertainty(pred_list, pred_logprob_list, text_list, qw_list, gold_list):
    samples = []
    for pred, logprob, text, qw, gold in zip(pred_list, pred_logprob_list, text_list, qw_list, gold_list):
        samples.append((pred, logprob, text, qw, remove_label_bio(gold.lower().strip())))
    sorted_data = sorted(samples, key=lambda x: x[1])
    all_logprob_data = [item[1] for item in sorted_data]
    average_score = sum(all_logprob_data) / len(all_logprob_data)
    print(f"all_average logprob: {average_score}")
    return average_score


def uncertainty_sample(pred_list, pred_logprob_list, text_list, qw_list, gold_list, sample_num):
    samples = []
    for pred, logprob, text, qw, gold in zip(pred_list, pred_logprob_list, text_list, qw_list, gold_list):
        samples.append((pred, logprob, text, qw, remove_label_bio(gold.lower().strip())))

    # 按照 pred_logprob_list 中的值排序，并选出最小的n个
    sorted_data = sorted(samples, key=lambda x: x[1])
    top_n = sorted_data[:sample_num]

    top_n_data = {
            "pred_list": [item[0] for item in top_n],
            "pred_logprob_list": [item[1] for item in top_n],
            "text_list": [item[2] for item in top_n],
            "qw_list": [item[3] for item in top_n],
            "gold_list": [item[4] for item in top_n]
    }
    # print(f"top_n logprobs: {top_n_data['pred_logprob_list']}")

    right_num = 0
    for item in top_n:
        # print(item)
        # print(item[0],item[4])
        if item[0]==item[4]:
            right_num+=1
    print("acc:", right_num/len(top_n))

    return top_n_data


def task_learning(qtype, generator, pred_list, pred_logprob_list, task_text_list, task_qw_list, task_gold_list, task_qtype_list, res_dict, data, label_description_dict, qtypes, sample_num, real_stop_labels):

    top_n_preds = uncertainty_sample(pred_list, pred_logprob_list, task_text_list, task_qw_list, task_gold_list, sample_num)
    updated_labels = []
    # print(f"active learning...")
    if qtype == 0 or qtype == 6 or qtype == 4:
        grouped_data = group_data_by_gold_label(top_n_preds, qtype, res_dict)
        for label, samples in grouped_data.items():
            if label in real_stop_labels:
                print(label, "has learned")
                continue
            old_label_description = label_description_dict[label]
            new_label_description = gen_new_description(qtype, label, old_label_description, samples['qw_list'], samples['text_list'], samples['pred_list'], samples['pred_logprob_list'], samples['gold_list'], data, generator)
            label_description_dict[label] = new_label_description
            updated_labels.append(label)
    elif qtype == 2:
        grouped_data = group_data_by_gold_label(top_n_preds, qtype, res_dict)
        for label, samples in grouped_data.items():
            old_label_description = label_description_dict[label]
            new_label_description = gen_new_description(qtype, label, old_label_description, samples['qw_list'], samples['text_list'], samples['pred_list'], samples['pred_logprob_list'], samples['gold_list'], data, generator)
            label_description_dict[label] = new_label_description

    elif qtype == 1:
        grouped_data = group_data_by_gold_label(top_n_preds, qtype, res_dict)
        for label, samples in grouped_data.items():
            print(label)
            if label == "yes":
                old_label_description = label_description_dict['pc']
                new_label_description = gen_new_description(qtype, 'pc', old_label_description, samples['qw_list'], samples['text_list'], samples['pred_list'], samples['pred_logprob_list'], samples['gold_list'], data, generator)
                label_description_dict['pc'] = new_label_description
    elif qtype == 5:
        grouped_data = group_data_by_gold_label(top_n_preds, qtype, res_dict)
        for label, samples in grouped_data.items():
            print(label)
            if label == "yes":
                old_label_description = label_description_dict['sc']
                new_label_description = gen_new_description(qtype, 'sc', old_label_description, samples['qw_list'], samples['text_list'], samples['pred_list'], samples['pred_logprob_list'], samples['gold_list'], data, generator)
                label_description_dict['sc'] = new_label_description
    elif qtype == 3:
        grouped_data = group_data_by_gold_label(top_n_preds, qtype, res_dict)
        for label, samples in grouped_data.items():
            old_label_description = label_description_dict[label]
            new_label_description = gen_new_description(qtype, label, old_label_description, samples['qw_list'], samples['text_list'], samples['pred_list'], samples['pred_logprob_list'], samples['gold_list'], data, generator)
            label_description_dict[label] = new_label_description


    return label_description_dict, updated_labels


def start_learning(generator, dataset, class_defs, text_list, qw_list, gold_list, res_dict, data, label_description_dict, qtypes):

    sampled_data = split_and_sample(dataset, data, text_list, qw_list, gold_list, qtypes, data.learning_size, data.seed)

    best_label_description_dict = copy.deepcopy(label_description_dict)
    best_class_defs = copy.deepcopy(class_defs)
    for qtype in list(set(qtypes)):
        print(f"\ntask-{qtype}")
        task_text_list, task_qw_list, task_gold_list, task_qtype_list = sampled_data[qtype]['text'], sampled_data[qtype]['qw'], sampled_data[qtype]['gold'], sampled_data[qtype]['qtype']
        print("uncertainty evaluate...")
        pred_list, pred_logprob_list, pred_logprob2_list = code_inference2(generator, dataset, class_defs, task_text_list, task_qw_list, res_dict, task_qtype_list, data)
        if qtype != 3:
            best_score = cal_uncertainty(pred_list, pred_logprob_list, text_list, qw_list, gold_list)
        else:
            best_score = cal_uncertainty(pred_list, pred_logprob2_list, text_list, qw_list, gold_list)

        real_stop_labels = []  # 已经优化好了的
        stop_labels = []
        if qtype == 4:
            learning_iteration = 5
        else:
            learning_iteration = data.learning_iteration
        for i in range(learning_iteration+1):
            print(f"\niteration-{i}")
            current_label_description_dict = copy.deepcopy(best_label_description_dict)
            max_sample_num = 20
            if qtype==3:
                max_sample_num=10

            for sample_num in range(5, max_sample_num + 1, 5):
                print(f"\nsample_num-{sample_num}")

                print(f"uncertainty learning ...")
                old_label_description_dict = copy.deepcopy(current_label_description_dict)
                current_label_description_dict, updated_labels = task_learning(qtype, generator, pred_list, pred_logprob_list, task_text_list, task_qw_list, task_gold_list, task_qtype_list, res_dict, data, current_label_description_dict, qtypes, sample_num, real_stop_labels)
                if current_label_description_dict == old_label_description_dict:
                    continue
                print("update class...")
                gen_class(res_dict, qtypes, dataset, generator, data, current_label_description_dict)
                current_class_defs = load_class_defs(data.status, data.code_path, dataset)

                print("uncertainty evaluate...")
                pred_list, pred_logprob_list, pred_logprob2_list = code_inference2(generator, dataset, current_class_defs, task_text_list, task_qw_list, res_dict, task_qtype_list, data)
                if qtype != 3:
                    current_score = cal_uncertainty(pred_list, pred_logprob_list, text_list, qw_list, gold_list)
                else:
                    current_score = cal_uncertainty(pred_list, pred_logprob2_list, text_list, qw_list, gold_list)

                if current_score > best_score:
                    print("Find!")
                    best_label_description_dict = copy.deepcopy(current_label_description_dict)
                    best_class_defs = copy.deepcopy(current_class_defs)
                    best_score = current_score
                    stop_labels += updated_labels
                    break
                else:
                    for label in stop_labels:
                        real_stop_labels.append(label)
                        stop_labels.remove(label)
                print()

    save_new_label_descriptions(best_label_description_dict, dataset, data)
    return best_class_defs