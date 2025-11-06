from query_llama import query_llama
from utils import get_class_name


def gen_description(label, generator, data, description):
    if len(description) == 0 or not data.use_gold_description:
        prompt = f'Please tell me what "{label}" means in the field of materials science. Use the following format to answer: ```Answer: [ONLY the "definition"]```'
        output_list, _, _ = query_llama(generator, [prompt], data, False)
        description = output_list[0]
        if "Answer:" in description:
            description = description.split("Answer:")[1].strip()
    class_description = '\n\t"""' + '\n\t' + description + '\n\t"""'
    return description, class_description


def gen_properties(label, description, generator, data):
    n = data.property_num
    if n == 0:
        n_str = ""
    else:
        n_str = str(n)+" "
    prompt = f'According to the definition of "{label}": {description}, Suppose "{label}" is a class in code, please tell me about the {n_str}most common properties of this class Use the following format to answer: ```Answer: [ONLY the "property name : definition"'
    output_list, _, _ = query_llama(generator, [prompt], data, False)
    res = output_list[0]
    property_dict = {}
    if "Answer:" in res:
        res = res.split("Answer:")[1].strip()
        res_list = res.split("\n")
        for line in res_list:
            if " : " in line:
                spans = line.split(" : ")
                name, definition = spans[-2], spans[-1]
                name = name.strip()
                definition = definition.strip()
                if " " in name:
                    name=name.split(" ")[1]
                name=name.replace("*", "").replace("`", "").replace("-", "")
                definition=definition.replace("*", "").replace("`", "").replace("-", "")
                property_dict[name]=definition
    class_properties = ""
    for key in property_dict.keys():
        class_properties += '\n\t\t' + "self." + key + "  # " + property_dict[key]
    return property_dict, class_properties


def ner_class(label, dataset, father, class_description, class_properties):
    class_name = get_class_name(label)
    class_def= f'''class {class_name}({father}):{class_description}
	def __init__(self, name: str):
		super().__init__(name=name){class_properties}'''
    return class_def


def rc_class(label, dataset, father, class_description, class_properties):
    if dataset == 2:
        label = label.replace("_of", "")
        class_name = get_class_name(label)
        class_def = f'''class {class_name}({father}):{class_description}
	def __init__(self, head_entity: Entity, tail_entity: Entity):
		super().__init__(head_entity = head_entity, tail_entity = tail_entity){class_properties}'''
    else:
        class_def = f'''class {label.capitalize()}({father}):{class_description}
	def __init__(self, head_entity: Entity, tail_entity: Entity):
		super().__init__(head_entity = head_entity, tail_entity = tail_entity){class_properties}'''
    return class_def


def ee_class(dataset, father, class_description, candidates, generator, data, event, gold_description_dict):
    args1, args2 = "", ""
    for i, arg in enumerate(candidates):
        if arg == "none":
            continue
        if class_description != "":
            description = gold_description_dict[arg]
            arg_description, _ = gen_description(arg, generator, data,description)
            if i == 0:
                args1 += arg + ": Entity,"
                args2 += "self." + arg + " = " + arg + "  # " + arg_description
            else:
                args1 += " " + arg + ": Entity,"
                args2 += "\n\t\t" + "self." + arg + " = " + arg + "  # " + arg_description
        else:
            if i == 0:
                args1 += arg + ": Entity,"
                args2 += "self." + arg + " = " + arg
            else:
                args1 += " " + arg + ": Entity,"
                args2 += "\n\t\t" + "self." + arg + " = " + arg
    class_def = f'''class {event}({father}):{class_description}
    def __init__(self, trigger: str, {args1}):
    	super().__init__(trigger=trigger)
    	{args2}'''
    return class_def


def sf_class(label, dataset, father, class_description, class_properties):
    class_name = get_class_name(label)
    class_def = f'''class {class_name}({father}):{class_description}
	def __init__(self, name: str):
		super().__init__(name=name){class_properties}'''
    return class_def


def pc_class(dataset, father, class_description, class_properties):
    class_def = f'''class GlassScienceText({father}):{class_description}
	def __init__(self, text: str):
		super().__init__(text=text){class_properties}'''
    return class_def


def sar_class(label, dataset, father, class_description, class_properties):
    class_name = get_class_name(label)
    class_def = f'''class {class_name}({father}):{class_description}
	def __init__(self, name: str):
		super().__init__(name=name){class_properties}'''
    return class_def


def sc_class(dataset, father, class_description, class_properties):
    class_def = f'''class ExperimentalFactText({father}):{class_description}
	def __init__(self, text: str):
		super().__init__(text=text){class_properties}'''
    return class_def
