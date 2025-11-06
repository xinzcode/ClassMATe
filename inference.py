import json
import os
import sys
import time
from collections import defaultdict
import torch


from build_prompt import build_code_prompt, build_text_prompt
from data import Data
from convert_data import convert_data
from build_class import gen_class
from learning import start_learning
from query_llama import query_llama, load_llama
import random
import numpy as np
from result_parse import llama_code_parse, llama_text_parse
from retrieve import retrieve
from task_eval import task_evaluate
from utils import query_for_bi, shuffle_list, load_class_defs, load_learned_label_descriptions
import warnings

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12350'
os.environ['RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WORLD_SIZE'] = '1'


# 忽略 FutureWarning 和 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from transformers.utils import logging

logging.set_verbosity_error()  # 只显示严重错误，屏蔽所有警告


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def code_inference(generator, dataset, class_defs, batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data):
    prompt_list = build_code_prompt(batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data.task_type_dict, class_defs, data.max_text_len, dataset, data.max_seq_len)
    output_list, _, _ = query_llama(generator, prompt_list, data, True)
    pred_list = llama_code_parse(output_list, batch_qtype_list, dataset)
    return pred_list


def text_inference(generator, dataset, batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data):
    if data.use_description:
        if data.use_properties:
            label_properties_dict = json.load(open(data.gold_description_path + f'/{dataset}_properties.json'))
            prompt_list = build_text_prompt(batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data.task_type_dict, data.max_text_len, data.max_seq_len, data.use_description, data.use_properties, label_description_dict, label_properties_dict)
        else:
            prompt_list = build_text_prompt(batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data.task_type_dict, data.max_text_len, data.max_seq_len, data.use_description, data.use_properties, label_description_dict)
    else:
        prompt_list = build_text_prompt(batch_text_list, batch_qw_list, res_dict, batch_qtype_list, data.task_type_dict, data.max_text_len, data.max_seq_len, data.use_description, data.use_properties)
    output_list, _, _ = query_llama(generator, prompt_list, data, True)
    pred_list = llama_text_parse(output_list, batch_qtype_list, dataset)
    return pred_list


def load_generator(data):
    if data.model == "llama3":
        generator = load_llama(data)
    elif data.model == "gpt4" or data.model == "gpt3.5":
        generator = load_llama()
    else:
        print("error model!")
        sys.exit()
    return generator


if __name__ == "__main__":
    start_time = time.time()
    # setting
    data = Data()
    for attr, value in data.__dict__.items():
        print(f"{attr}: {value}")
    seed_torch(data.seed)

    # load llm
    generator = load_generator(data)

    qtype_micro_dict, qtype_macro_dict = defaultdict(list), defaultdict(list)
    all_rate = []
    for dataset in data.datasets:
        print("\nDataset", dataset)

        # load data
        text_list, qw_list, gold_list, res_dict, qtypes = convert_data(data, dataset)
        if len(text_list) == 0:
            continue
        text_list, qw_list, gold_list, qtypes = shuffle_list(text_list, qw_list, gold_list, qtypes)

        # retrieve
        if data.retrieve:
            print("retrieve...")
            retrieve(res_dict, qtypes, dataset, generator, data)
        label_description_dict = json.load(open(data.gold_description_path + f'/{dataset}_description.json'))

        # gen class
        if data.gen_class:
            print("gen class...")
            gen_class(res_dict, qtypes, dataset, generator, data, label_description_dict)
        class_defs = load_class_defs(data.status, data.code_path, dataset)

        # learning
        if data.learning:
            if data.code_style:
                class_defs = start_learning(generator, dataset, class_defs, text_list, qw_list, gold_list, res_dict, data, label_description_dict, qtypes)

        # inference
        if data.code_style:
            if not data.learning:
                class_defs = load_class_defs(data.status, data.code_path, dataset)
            if data.use_learned:
                class_defs = load_learned_label_descriptions(dataset, data)
            pred_list = code_inference(generator, dataset, class_defs, text_list, qw_list, res_dict, qtypes, data)
        else:
            pred_list = text_inference(generator, dataset, text_list, qw_list, res_dict, qtypes, data)

        # evaluate
        if data.print_gold_pred:
            for gold, pred in zip(gold_list, pred_list):
                print(gold+"\n"+pred+"\n\n")
        qtype_micro_dict_updated, qtype_macro_dict_updated = task_evaluate(res_dict, qtypes, gold_list, pred_list,  qtype_micro_dict, qtype_macro_dict)
        qtype_micro_dict, qtype_macro_dict = qtype_micro_dict_updated, qtype_macro_dict_updated

    # score
    print("\nAll")
    all_micro, all_macro = 0.0, 0.0
    for key in qtype_micro_dict.keys():
        micro = np.mean(qtype_micro_dict[key])
        macro = np.mean(qtype_macro_dict[key])
        print('task = {} micro-f1 = {} macro-f1 = {} '.format(key, format(micro, '.4f'), format(macro, '.4f')))
        all_micro += micro
        all_macro += macro
    print('all tasks micro-f1 = {} macro-f1 = {} '.format(format(all_micro/len(qtype_micro_dict.keys()), '.4f'), format(all_macro/len(qtype_micro_dict.keys()), '.4f')))

    end_time = time.time()
    print(time.strftime("\ntime = %H h %M m %S s ", time.gmtime(end_time - start_time)))
