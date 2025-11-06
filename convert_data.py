import json
import random
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np


def extract_entities(ks, vs):
    # 初始化变量
    current_entity = []  # 用于存储当前实体的词
    current_label = None  # 当前实体的标签
    text_list = []  # 最终的文本列表
    entity_list = []  # 存储实体词列表
    label_list = []  # 存储实体对应的标签

    # 遍历字典中的键值对
    for k, v in zip(ks, vs):
        if v.startswith("B-"):  # 开始一个新的实体
            # 如果有未保存的实体，先保存它
            if current_entity:
                text_list.append(" ".join(current_entity))  # 合并成完整文本
                entity_list.append(current_entity)
                label_list.append(current_label)
            # 初始化新的实体
            current_entity = [k]
            current_label = v[2:]  # 去掉 "B-" 前缀
        elif v.startswith("I-") and current_label == v[2:]:  # 当前是 I 并且标签匹配
            current_entity.append(k)  # 加入当前实体
        else:  # 非 B- 或 I- 的情况
            # 如果有未保存的实体，先保存它
            if current_entity:
                text_list.append(" ".join(current_entity))
                entity_list.append(current_entity)
                label_list.append(current_label)
            # 重置当前实体
            current_entity = []
            current_label = None

    # 处理最后一个实体
    if current_entity:
        text_list.append(" ".join(current_entity))
        entity_list.append(current_entity)
        label_list.append(current_label)

    # 输出结果
    # for text, entity, label in zip(text_list, entity_list, label_list):
    #     print(f"Text: {text}, Entity: {entity}, Label: {label}")
    return text_list, label_list


def load_matscholar(dataset_path, task_type_dict):
    path = '{}/matscholar.json'.format(dataset_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []
    tokens = df['tokens'].tolist()
    labels = df['labels'].tolist()

    for text, ann in zip(tokens, labels):
        # NER
        ks, vs = [], []
        for k, v in ann.items():
            ks.append(k)
            vs.append(v)
        ks, vs = extract_entities(ks, vs)
        # print()
        # print(text)
        for k, v in zip(ks, vs):
            # print(k, v)
            text_list.append(text)
            ques_word_list.append(k)
            answer_list.append(v)
            qtype_list.append(task_type_dict['ner'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_sofc_token(dataset_path, task_type_dict):
    path = '{}/sofc_token.json'.format(dataset_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []
    tokens = df['tokens'].tolist()
    token_labels = df['token_labels'].tolist()
    slot_labels = df['slot_labels'].tolist()

    for token,label,slot in zip(tokens,token_labels,slot_labels):
        text = ' '.join(token)

        ks, vs = extract_entities(token, label)
        # print()
        # print(text)
        for k, v in zip(ks, vs):
            # print(k, v)
            text_list.append(text)
            ques_word_list.append(k)
            answer_list.append(v)
            qtype_list.append(task_type_dict['ner'])

        ks, vs = extract_entities(token, slot)
        # print()
        # print(text)
        for k, v in zip(ks, vs):
            # print(k, v)
            text_list.append(text)
            ques_word_list.append(k)
            answer_list.append(v)
            qtype_list.append(task_type_dict['sf'])

    return text_list, ques_word_list, answer_list, qtype_list


def load_synthesis_procedures(dataset_path, task_type_dict):
    path = '{}/synthesis_procedures.json'.format(dataset_path)
    f = open(path)
    data_dict = json.load(f)
    items = data_dict['data']
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []

    for item in items:
        text = item['text']
        # NER
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            entity_name = text[l:r]
            entity_type = item['t_type_dict'][key]
            text_list.append(text)
            ques_word_list.append(entity_name)
            answer_list.append(entity_type)
            qtype_list.append(task_type_dict['ner'])
        # RC
        for key,value in item['r_args_dict'].items():
            args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                args.append(text[l:r].replace(',',''))
            relation_type = item['r_type_dict'][key]
            text_list.append(text)
            ques_word_list.append(','.join(args))
            answer_list.append(relation_type)
            qtype_list.append(task_type_dict['rc'])
        # EE
        for key,value in item['e_trig_dict'].items():
            event_struc = []
            l,r = item['t_span_dict'][value]
            trigger = text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                event_struc.append(str(text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            if len(event_struc)==0:
                event_struc.append('none:none')
                continue
            text_list.append(text)
            ques_word_list.append(trigger)
            answer_list.append(','.join(event_struc))
            qtype_list.append(task_type_dict['ee'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_sc_comics(dataset_path, task_type_dict):
    path = '{}/sc_comics.json'.format(dataset_path)
    f = open(path)
    data_dict = json.load(f)
    items = data_dict['data']
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []

    for item in items:
        text = item['text']
        # NER
        for key,value in item['t_span_dict'].items():
            l,r = value[0],value[1]
            entity_name = text[l:r]
            entity_type = item['t_type_dict'][key]
            text_list.append(text)
            ques_word_list.append(entity_name)
            answer_list.append(entity_type)
            qtype_list.append(task_type_dict['ner'])
        # RC
        for key,value in item['r_args_dict'].items():
            args = []
            for e in value:
                if e[0]=='T':
                    l,r = item['t_span_dict'][e]
                else:
                    l,r = item['t_span_dict'][item['e_trig_dict'][e]]
                args.append(text[l:r].replace(',',''))
            relation_type = item['r_type_dict'][key]
            text_list.append(text)
            ques_word_list.append(args[0] + "," + args[1])
            answer_list.append(relation_type)
            qtype_list.append(task_type_dict['rc'])
        # EE
        for key,value in item['e_trig_dict'].items():
            event_struc = []
            l,r = item['t_span_dict'][value]
            trigger = text[l:r]
            for arg in item['e_args_dict'][key]:
                l,r = item['t_span_dict'][arg[1]]
                event_struc.append(str(text[l:r]).replace(',','').replace(':','') + ":" + str(arg[0]).replace(',','').replace(':',''))
            if len(event_struc)==0:
                event_struc.append('none:none')
                continue
            text_list.append(text)
            ques_word_list.append(trigger)
            answer_list.append(','.join(event_struc))
            qtype_list.append(task_type_dict['ee'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_glass(dataset_path, task_type_dict):
    path = '{}/glass_non_glass.json'.format(dataset_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []
    abstracts = df['Abstract'].tolist()
    labels = df['Label'].tolist()

    for text,label in zip(abstracts,labels):
        # PC
        text_list.append(text)
        ques_word_list.append("")
        answer = 'yes' if int(label)==1 else 'no'
        answer_list.append(answer)
        qtype_list.append(task_type_dict['pc'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_re(dataset_path, task_type_dict):
    path = '{}/structured_re.json'.format(dataset_path)
    data = open(path).read().strip().split('\n')
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []

    for line in data:
        # RC
        j = json.loads(line)
        text = j['sentText']
        for rel in j['relationMentions']:
            args = [rel['arg1Text'],rel['arg2Text']]
            answer = rel['relText']
            text_list.append(text)
            ques_word_list.append(','.join(args))
            answer_list.append(answer)
            qtype_list.append(task_type_dict['rc'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_synthesis_actions(dataset_path, task_type_dict):
    path = '{}/synthesis_actions.json'.format(dataset_path)
    data = json.load(open(path))
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []

    for j in data:
        # SAR
        text = j['sentence']
        for ann in j['annotations']:
            token = ann['token']
            tag = ann['tag']
            if len(tag)==0:
                continue
            text_list.append(text)
            ques_word_list.append(token)
            answer_list.append(tag)
            qtype_list.append(task_type_dict['sar'])
    return text_list, ques_word_list, answer_list, qtype_list


def load_sofc_sent(dataset_path, task_type_dict):
    path = '{}/sofc_sent.json'.format(dataset_path)
    df = pd.read_json(path, orient='split')
    text_list = []
    ques_word_list = []
    answer_list = []
    qtype_list = []
    sents = df['sents'].tolist()
    labels = df['sent_labels'].tolist()

    for sent,label in zip(sents,labels):
        # SC
        text_list.append(sent)
        ques_word_list.append("")
        answer = 'yes' if int(label)==1 else 'no'
        answer_list.append(answer)
        qtype_list.append(task_type_dict['sc'])
    return text_list, ques_word_list, answer_list, qtype_list


def format_data(dataset_path, dataset, task_type_dict, tasks, lower):

    processed_data = dict()
    processed_data['texts'] = []
    processed_data['ques_words'] = []
    processed_data['answers'] = []
    processed_data['qtypes'] = []
    if dataset==0:
        text_list, ques_word_list, answer_list, qtype_list = load_matscholar(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==1:
        text_list, ques_word_list, answer_list, qtype_list = load_sofc_token(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==2:
        text_list, ques_word_list, answer_list, qtype_list = load_synthesis_procedures(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==3:
        text_list, ques_word_list, answer_list, qtype_list = load_sc_comics(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==4:
        text_list, ques_word_list, answer_list, qtype_list = load_glass(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==5:
        text_list, ques_word_list, answer_list, qtype_list = load_re(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==6:
        text_list, ques_word_list, answer_list, qtype_list = load_synthesis_actions(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    if dataset==7:
        text_list, ques_word_list, answer_list, qtype_list = load_sofc_sent(dataset_path, task_type_dict)
        processed_data['texts'] += text_list
        processed_data['ques_words'] += ques_word_list
        processed_data['answers'] += answer_list
        processed_data['qtypes'] += qtype_list
        # print('dataset = {} size = {} qtype = {}'.format(dataset, len(processed_data['texts']), np.unique(np.array(qtype_list))))
        # print('text len = {} ques_word len = {} answer len = {} qtype len = {}'.format(len(text_list), len(ques_word_list), len(answer_list), len(qtype_list)))
    df = pd.DataFrame(processed_data)
    df = df.loc[df['qtypes'].isin(tasks)]
    if lower:
        df['texts'] = df['texts'].apply(lambda x:x.lower())
        df['ques_words'] = df['ques_words'].apply(lambda x:x.lower())
        df['answers'] = df['answers'].apply(lambda x:x.lower())

    return df


def get_res_dict(df):
    res_dict = dict()
    for qtype in df['qtypes'].unique():
        tmp_df = df[df.qtypes==qtype]
        if qtype==0:
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['t_type_set'] = answer_set
            res_dict['t_type_dict'] = answer_map
            print(res_dict['t_type_set'])
        if qtype==1:
            res_dict['pc_type_dict'] = {'yes':1,'no':0}
            res_dict['pc_type_set'] = ['yes','no']
            print(res_dict['pc_type_set'])
        if qtype==2:
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['r_type_set'] = answer_set
            res_dict['r_type_dict'] = answer_map
            print(res_dict['r_type_set'])
        if qtype==3:
            answers = tmp_df['answers'].tolist()
            unique_types = list() # set 改为list 保留顺序
            for answer in answers:
                for args in answer.split(','):
                    args = args.split(':')[1]
                    if args not in unique_types:
                        unique_types.append(args)
            answer_set = [x.lower().strip().replace(' ', '') for x in unique_types]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['e_role_set'] = answer_set
            res_dict['e_role_dict'] = answer_map
            print(res_dict['e_role_set'])
        if qtype==4:
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['sar_set'] = answer_set
            res_dict['sar_dict'] = answer_map
            print(res_dict['sar_set'])
        if qtype==5:
            res_dict['sc_type_dict'] = {'yes':1,'no':0}
            res_dict['sc_type_set'] = ['yes','no']
            print(res_dict['sc_type_set'])
        if qtype==6:
            answer_set = [x.lower().strip().replace(' ', '') for x in tmp_df['answers'].unique()]
            answer_map = dict(zip(answer_set,list(range(len(answer_set)))))
            res_dict['sf_type_set'] = answer_set
            res_dict['sf_type_dict'] = answer_map
            print(res_dict['sf_type_set'])
    return res_dict


def split_data(df,  default_train_num, train_size = 0.9, even_split = False, seed = 42):
    if (even_split is False):
        train_df_list = []
        test_df_list = []
        for qtype in df['qtypes'].unique():
            tmp_df = df[df.qtypes == qtype]
            tmp_train_df, tmp_test_df = train_test_split(tmp_df, train_size=train_size, random_state=seed)
            train_df_list.append(tmp_train_df)
            test_df_list.append(tmp_test_df)
            test_size = len(tmp_test_df)
            train_size2 = len(tmp_train_df)
            print('qtype = {} test datasize = {} train datasize = {}'.format(qtype, test_size, train_size2))
        train_df = pd.concat(train_df_list, ignore_index=True)
        test_df = pd.concat(test_df_list, ignore_index=True)
    else:
        train_df_list = []
        test_df_list = []
        for qtype in df['qtypes'].unique():
            if qtype == 1:
                train_size = 1-(1-train_size)*10
            test_size, train_size2 = 0, 0
            tmp_df = df[df.qtypes == qtype]
            if (qtype in [0, 1, 2, 4, 5, 6]):
                unique_answers = list(tmp_df['answers'].unique())
                for answer in unique_answers:
                    tmp_df_2 = tmp_df.loc[tmp_df['answers'] == answer]
                    if len(tmp_df_2) * train_size > 1:
                        tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=train_size, random_state=seed)
                        train_df_list.append(tmp_train_df)
                        test_df_list.append(tmp_test_df)
                        test_size += len(tmp_test_df)
                        train_size2 += len(tmp_train_df)
                    else:
                        tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=default_train_num, random_state=seed)
                        train_df_list.append(tmp_train_df)
                        test_df_list.append(tmp_test_df)
                        test_size += len(tmp_test_df)
                        train_size2 += len(tmp_train_df)
            else:
                answers = tmp_df['answers'].tolist()
                unique_types = list()
                for answer in answers:
                    for args in answer.split(','):
                        if args.split(':')[1] not in unique_types:
                            unique_types.append(args.split(':')[1])

                def is_type(x, unique_type):
                    isin = False
                    for args in x.split(','):
                        role_type = args.split(':')[1]
                        if (role_type == unique_type):
                            return True
                    return isin

                for unique_type in unique_types:
                    tmp_df_2 = tmp_df.loc[tmp_df['answers'].apply(lambda x: is_type(x, unique_type))]
                    if len(tmp_df_2) * train_size > 1:
                        tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=train_size, random_state=seed)
                        train_df_list.append(tmp_train_df)
                        test_df_list.append(tmp_test_df)
                        test_size+=len(tmp_test_df)
                        train_size2 += len(tmp_train_df)
                    else:
                        tmp_train_df, tmp_test_df = train_test_split(tmp_df_2, train_size=default_train_num, random_state=seed)
                        train_df_list.append(tmp_train_df)
                        test_df_list.append(tmp_test_df)
                        test_size += len(tmp_test_df)
                        train_size2 += len(tmp_train_df)
            print('qtype = {} test datasize = {} train datasize = {}'.format(qtype, test_size, train_size2))
        train_df = pd.concat(train_df_list, ignore_index=True)
        test_df = pd.concat(test_df_list, ignore_index=True)

    return train_df, test_df


def df2json(df, output_path):
    result, qtypes = [], []
    for row in df.itertuples(index=False):
        text, ques_word, answer, qtype = row[0], row[1], row[2], row[3]
        item = {
            "text": text,
            "ques_words": ques_word,
            "answers": answer,
            "qtype": qtype
        }
        qtypes.append(qtype)
        result.append(item)
    with open(output_path, 'w') as dump_f:
        json.dump(result, dump_f, indent=2)
    return qtypes


def load_data(path):
    with open(path) as json_file:
        test_data = json.load(json_file)
    text_list = [item["text"] for item in test_data]
    qw_list = [item["ques_words"] for item in test_data]
    gold_list = [item["answers"] for item in test_data]
    qtypes = [item["qtype"] for item in test_data]
    return text_list, qw_list, gold_list, qtypes


def convert_data(data, dataset):

    # format data and get candidates
    df = format_data(data.dataset_path, dataset, data.task_type_dict, data.tasks, data.lower)
    if df.empty:
        return [], [], [], {}, []
    res_dict = get_res_dict(df)

    # split train test
    train_size = data.train_size
    even_split = data.even_split

    train_df, test_df = split_data(df, data.default_train_num, train_size, even_split, data.seed)

    # to json
    df2json(train_df, data.gen_train_data_path)
    df2json(test_df, data.gen_test_data_path)

    # load json
    test_text_list, test_qw_list, test_gold_list, test_qtypes = load_data(data.gen_test_data_path)
    train_text_list, _, _, train_qtypes = load_data(data.gen_train_data_path)

    return test_text_list, test_qw_list, test_gold_list, res_dict, test_qtypes

