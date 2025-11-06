from build_class import remove_label_bio
import jinja2
from collections import defaultdict, OrderedDict
from utils import remove_duplicates_preserve_order, text_suitable


def get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict):
    new_candidates = []
    for label in candidates:
        label_description = label_description_dict[remove_label_bio(label)]
        if use_properties:
            label_properties = label_properties_dict[remove_label_bio(label)]
            new_can = label + "(" + label_description + ".  The related properties include: " + label_properties + ")"
        else:
            new_can = label + "(" + label_description + ")"
        new_candidates.append(new_can)
    return new_candidates


def build_text_prompt(text_list, qw_list, res_dict, qtypes, task_type_dict, max_text_len, max_seq_len, use_description, use_properties, label_description_dict=None, label_properties_dict=None):
    prompt_list = []
    for text,qw,qtype in zip(text_list,qw_list,qtypes):
        text=text[:max_text_len]
        if qtype==task_type_dict['ner']:
            candidates = res_dict['t_type_set']
            if use_description:
                candidates = get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict)
            choices = ', '.join(candidates)
            entity_name = qw
            question = f'This is a material science literature text: {text} Please tell me the entity type of "{entity_name}" in the literature text, all entity types are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            text = text_suitable(text, question, max_seq_len, max_text_len)
            question = f'This is a material science literature text: {text} Please tell me the entity type of "{entity_name}" in the literature text, all entity types are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype==task_type_dict['rc']:
            candidates = res_dict['r_type_set']
            if use_description:
                candidates = get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict)
            choices = ', '.join(candidates)
            args1, args2 = qw.split(",")
            question = f'This is a material science literature text: {text} Please tell me the relationship between "{args1}" and "{args2}" in the literature text, all relationships are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype == task_type_dict['ee']:
            candidates = res_dict['e_role_set']
            if use_description:
                candidates = get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict)
            choices = ', '.join(candidates)
            trigger = qw
            question = f'This is a material science literature text: {text} Please provide the event arguments in the literature text according to the given trigger word: "{trigger}", all required arguments are in Args. Args: {choices}. Use the following format to answer: ```Answer: [ONLY the "required arg1 : arg word1; required arg2 : arg word2"; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype == task_type_dict['sf']:
            candidates = res_dict['sf_type_set']
            if use_description:
                candidates = get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict)
            choices = ', '.join(candidates)
            entity_name = qw
            question = f'This is a material science literature text: {text} Please tell me the solt type of "{entity_name}" in the literature text, all solt types are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype == task_type_dict['pc']:
            candidates = res_dict['pc_type_set']
            extend_str = ""
            if use_description:
                label_description = label_description_dict["pc"]
                if use_properties:
                    label_properties = label_properties_dict["pc"]
                    extend_str = "(" + label_description + ".  The related properties include: " + label_properties + ")"
                else:
                    extend_str = "(" + label_description + ")"
            choices = ', '.join(candidates)
            question = f'This is a material science literature paragraph: {text} Please tell me whether the given paragraph pertains to glass science{extend_str}, all answer options are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype == task_type_dict['sar']:
            candidates = res_dict['sar_set']
            if use_description:
                candidates = get_new_candidates(candidates, label_description_dict, use_properties, label_properties_dict)
            choices = ', '.join(candidates)
            entity_name = qw
            question = f'This is a material science literature text: {text} Please tell me the synthesis action of "{entity_name}" in the literature text, all synthesis actions are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
        elif qtype == task_type_dict['sc']:
            candidates = res_dict['sc_type_set']
            extend_str = ""
            if use_description:
                label_description = label_description_dict["sc"]
                if use_properties:
                    label_properties = label_properties_dict["sc"]
                    extend_str = "(" + label_description + ".  The related properties include: " + label_properties + ")"
                else:
                    extend_str = "(" + label_description + ")"
            choices = ', '.join(candidates)
            question = f'This is a material science literature sentences: {text} Please tell me whether the given sentences is a experimental fact sentence{extend_str}, all answer options are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt_list.append(question)
    return prompt_list



def build_code_prompt(text_list, qw_list, res_dict, qtypes, task_type_dict, class_defs, max_text_len, dataset, max_seq_len):
    # load template
    template_loader = jinja2.FileSystemLoader(searchpath="./dataset/templates")
    template_env = jinja2.Environment(loader=template_loader)
    entity_template = template_env.get_template("NER")
    relation_template = template_env.get_template("RC")
    eae_template = template_env.get_template("EE")
    sf_template = template_env.get_template("SF")
    pc_template = template_env.get_template("PC")
    sar_template = template_env.get_template("SAR")
    sc_template = template_env.get_template("SC")

    # render the template
    prompt_list = []
    for i,(text,qw,qtype) in enumerate(zip(text_list,qw_list,qtypes)):
        text=text[:max_text_len]
        if qtype==task_type_dict['ner']:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['t_type_set']])
            selected_class_defs = [class_defs[label] for label in candidates]
            question_describe = f'Based on the above classes, please determine which entity class the "{qw}" in the above text belongs to. Use the following format to answer: ```Answer: [ONLY the entity class name that you think most correct; not a complete sentence]```'
            prompt = entity_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            text = text_suitable(text, prompt, max_seq_len, max_text_len)
            prompt = entity_template.render(text=text, class_defs=selected_class_defs,question_describe=question_describe)
            prompt_list.append(prompt)
        elif qtype==task_type_dict['rc']:
            candidates = res_dict['r_type_set']
            selected_class_defs = [class_defs[label] for label in candidates]
            args1, args2 = qw.split(",")
            question_describe = f'Based on the above classes, please determine which relation class the relationship between "{args1}" and "{args2}" in the above text belongs to. Use the following format to answer: ```Answer: [ONLY the relation class name that you think most correct; not a complete sentence]```'
            prompt = relation_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
        elif qtype == task_type_dict['ee']:
            candidates = ["event"]
            selected_class_defs = [class_defs[label_name] for label_name in candidates]
            question_describe = f'Based on the above classes, provide the event attribute words in the above text based on the given trigger word "{qw}". Use the following format to answer: ```Answer: [ONLY the "required arg1 : arg word1; required arg2 : arg word2"; not a complete sentence]```'
            prompt = eae_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
        if qtype==task_type_dict['sf']:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['sf_type_set']])
            selected_class_defs = [class_defs[label] for label in candidates]
            question_describe = f'Based on the above classes, please determine which solt class the "{qw}" in the above text belongs to. Use the following format to answer: ```Answer: [ONLY the solt class name that you think most correct; not a complete sentence]```'
            prompt = sf_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
        if qtype==task_type_dict['pc']:
            candidates = res_dict['pc_type_set']
            choices = ', '.join(candidates)
            selected_class_defs = [class_defs["pc"]]
            question_describe = f'Based on the above classes, Please tell me whether the given text pertains to glass science text class, all answer options are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt = pc_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
        if qtype==task_type_dict['sar']:
            candidates = res_dict['sar_set']
            selected_class_defs = [class_defs[label] for label in candidates]
            question_describe = f'Based on the above classes, please determine which synthesis action class the "{qw}" in the above text belongs to. Use the following format to answer: ```Answer: [ONLY the synthesis action class name that you think most correct; not a complete sentence]```'
            prompt = sar_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
        if qtype==task_type_dict['sc']:
            candidates = res_dict['sc_type_set']
            choices = ', '.join(candidates)
            selected_class_defs = [class_defs["sc"]]
            question_describe = f'Based on the above classes, please determines whether the given text is the experimental fact text class, all answer options are in Options. Options: {choices}. Use the following format to answer: ```Answer: [ONLY the option that you think most correct; not a complete sentence]```'
            prompt = sc_template.render(text=text, class_defs=selected_class_defs, question_describe=question_describe)
            prompt_list.append(prompt)
    return prompt_list

