import re



def llama_text_parse(output_list, qtype, dataset):
    pred_list = []
    for pred, type in zip(output_list, qtype):
        pred = pred.replace("```", "")
        if type == 3:
            try:
                # print(pred)
                if "Answer:" in pred:  # 去掉Answer
                    pred = pred.split("Answer:")[1]
                if "\n" in pred:  # 多行
                    pred = pred.split("\n")[0]
                pred = pred.strip()

                if ";" in pred:  # 多个的情况
                    preds = pred.split(";")
                else:
                    if pred.split(":") == 2:  # 单个的情况
                        preds = [pred]
                    else:
                        continue
                new_preds = []
                # print(preds)
                for p in preds:
                    if ":" not in p:
                        continue
                    arg, wd = p.split(":")
                    arg = arg.strip()
                    wd = wd.strip()
                    if "," in wd:  # 多个词的情况
                        wd = wd.split(",")[0]
                    wd = wd.strip()
                    # 过滤一些
                    if arg.lower() == "trigger":
                        continue
                    if "none" in wd.lower():
                        continue
                    new_preds.append(wd + ":" + arg)
                pred = ",".join(new_preds)
                # print(pred)
                # print("...............")
            except:
                pred = "none:none"
        else:
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1]
            else:
                pred = "none"
        pred_list.append(pred.lower().strip())
    return pred_list

        
def llama_code_parse(output_list, qtype, dataset):
    pred_list = []
    for pred, type in zip(output_list, qtype):
        pred = pred.replace("```", "")
        if type == 3:
            try:
                # print(pred)
                if "Answer:" in pred:  # 去掉Answer
                    pred = pred.split("Answer:")[1]
                if "\n" in pred:  # 多行
                    pred = pred.split("\n")[0]
                pred = pred.strip()

                if ";" in pred:  # 多个的情况
                    preds = pred.split(";")
                else:
                    if pred.split(":") == 2:  # 单个的情况
                        preds = [pred]
                    else:
                        continue
                new_preds = []
                # print(preds)
                for p in preds:
                    if ":" not in p:
                        continue
                    arg, wd = p.split(":")
                    arg = arg.strip()
                    wd = wd.strip()
                    if "," in wd:  # 多个词的情况
                        wd = wd.split(",")[0]
                    wd = wd.strip()
                    # 过滤一些
                    if arg.lower() == "trigger":
                        continue
                    if "none" in wd.lower():
                        continue
                    new_preds.append(wd + ":" + arg)
                pred = ",".join(new_preds)
                # print(pred)
                # print("...............")
            except:
                pred = "none:none"
        else:
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1]
            else:
                pred = "none"
        pred_list.append(pred.lower().strip())
    return pred_list


def llama_code_parse2(output_list, qtypes, dataset):
    pred_list = []
    for pred, qtype in zip(output_list, qtypes):
        pred = pred.replace("```", "")
        if qtype==0:
            if "Part2:" in pred:
                pred = pred.split("Part2:")[1].replace("```", "")
        pred_list.append(pred.lower().strip())
    return pred_list