import json
from build_class_def import ner_class, rc_class, ee_class, sf_class, pc_class, sar_class, sc_class, \
    gen_properties, gen_description
from utils import remove_duplicates_preserve_order, remove_label_bio


def gen_class(res_dict, qtypes, dataset, generator, data, gold_description_dict):

    label_class_dict_base = dict()
    label_class_dict_with_description = dict()
    label_class_dict_with_description_properties = dict()

    for qtype in list(set(qtypes)):
        # NER
        if qtype == 0:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['t_type_set']])
            for label in candidates:
                description = gold_description_dict[label]
                true_label = label
                description, class_description = gen_description(true_label, generator, data, description)
                property_dict, class_properties = gen_properties(true_label, description, generator, data)
                label_class_dict_base[label] = ner_class(true_label, dataset, "Entity", "", "")
                label_class_dict_with_description[label] = ner_class(true_label, dataset, "Entity", class_description, "")
                if dataset == 2:
                    label_class_dict_with_description_properties[label] = ner_class(true_label, dataset, "Entity", class_description, "")
                else:
                    label_class_dict_with_description_properties[label] = ner_class(true_label, dataset, "Entity", class_description, class_properties)

        # RC
        if qtype == 2:
            candidates = res_dict['r_type_set']
            for label in candidates:
                true_label = label
                description = gold_description_dict[true_label]
                if true_label not in label_class_dict_base.keys():
                    description, class_description = gen_description(true_label, generator, data, description)
                    property_dict, class_properties = gen_properties(true_label, description, generator, data)
                    label_class_dict_base[true_label] = rc_class(true_label, dataset, "Relation", "", "")
                    label_class_dict_with_description[true_label] = rc_class(true_label, dataset, "Relation", class_description, "")
                    label_class_dict_with_description_properties[true_label] = rc_class(true_label, dataset, "Relation", class_description, class_properties)

        # EE
        if qtype == 3:
            candidates = res_dict['e_role_set']
            for label in candidates:
                true_label = label
                description = gold_description_dict[true_label]
                if true_label not in label_class_dict_base.keys():
                    description, class_description = gen_description(true_label, generator, data, description)
            if dataset == 2:
                event = "MaterialsSynthesis"
            else:
                event = "Doping"
            description = gold_description_dict["event"]
            description, class_description = gen_description(event, generator, data, description)
            label_class_dict_base["event"] = ee_class(dataset, "Event", "", candidates, generator, data, event, gold_description_dict)
            class_def = ee_class(dataset, "Event", class_description, candidates, generator, data, event, gold_description_dict)
            label_class_dict_with_description["event"] = class_def
            label_class_dict_with_description_properties["event"] = class_def

        # SF
        if qtype == 6:
            candidates = remove_duplicates_preserve_order([remove_label_bio(label) for label in res_dict['sf_type_set']])
            for true_label in candidates:
                description = gold_description_dict[true_label]
                description, class_description = gen_description(true_label, generator, data, description)
                property_dict, class_properties = gen_properties(true_label, description, generator, data)
                label_class_dict_base[true_label] = sf_class(true_label, dataset, "Slot", "", "")
                label_class_dict_with_description[true_label] = sf_class(true_label, dataset, "Slot", class_description, class_properties)
                label_class_dict_with_description_properties[true_label] = sf_class(true_label, dataset, "Slot", class_description, class_properties)

        # PC (no candidates)
        if qtype == 1:
            description = gold_description_dict["pc"]
            description, class_description = gen_description("GlassScienceText", generator, data, description)
            property_dict, class_properties = gen_properties("GlassScienceText", description, generator, data)
            label_class_dict_base["pc"] = pc_class(dataset, "MaterialScienceText", "", "")
            label_class_dict_with_description["pc"] = pc_class(dataset, "MaterialScienceText", class_description, "")
            label_class_dict_with_description_properties["pc"] = pc_class(dataset, "MaterialScienceText", class_description, class_properties)

        # SAR
        if qtype == 4:
            candidates = res_dict['sar_set']
            for label in candidates:
                true_label = label
                description = gold_description_dict[true_label]
                description, class_description = gen_description(true_label, generator, data, description)
                property_dict, class_properties = gen_properties(true_label, description, generator, data)
                label_class_dict_base[true_label] = sar_class(true_label, dataset, "SynthesisAction", "", "")
                label_class_dict_with_description[true_label] = sar_class(true_label, dataset, "SynthesisAction", class_description, "")
                label_class_dict_with_description_properties[true_label] = sar_class(true_label, dataset, "SynthesisAction", class_description, class_properties)

        # SC (no candidates)
        if qtype == 5:
            description = gold_description_dict["sc"]
            description, class_description = gen_description("ExperimentalFactText", generator, data, description)
            property_dict, class_properties = gen_properties("ExperimentalFactText", description, generator, data)
            label_class_dict_base["sc"] = sc_class(dataset, "MaterialScienceText", "", "")
            label_class_dict_with_description["sc"] = sc_class(dataset, "MaterialScienceText", class_description, "")
            label_class_dict_with_description_properties["sc"] = sc_class(dataset, "MaterialScienceText", class_description, class_properties)

    # save
    with open(data.code_path+f"/{dataset}_class_base.json", "w") as json_file:
        json.dump(label_class_dict_base, json_file, indent=4)

    with open(data.code_path+f"/{dataset}_class_with_description.json", "w") as json_file:
        json.dump(label_class_dict_with_description, json_file, indent=4)

    with open(data.code_path+f"/{dataset}_class_with_description_properties.json", "w") as json_file:
        json.dump(label_class_dict_with_description_properties, json_file, indent=4)






