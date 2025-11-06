import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


def load_llama(data):
    generator = AutoModelForCausalLM.from_pretrained(
        data.ckpt_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    return generator


def text2dialog(batch_input_data):
    batch_dialog = [[{"role": "user", "content": item}] for item in batch_input_data]
    return batch_dialog


def gen_res(generator, batch_dialog, data):

    tokenizer = AutoTokenizer.from_pretrained(data.ckpt_dir)

    # 处理输入
    input_ids = tokenizer.apply_chat_template(
        batch_dialog[0],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(generator.device)

    # 终止 token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 生成输出，同时获取每个 token 的概率
    outputs = generator.generate(
        input_ids,
        max_new_tokens=data.max_gen_len,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        top_p=0.9,
        return_dict_in_generate=True,  # 让 model.generate 返回一个字典
        output_scores=True  # 输出每个 token 的 logits
    )

    # 获取生成的 token 序列
    generated_tokens = outputs.sequences[0][input_ids.shape[-1]:]

    # 拼接完整的回答
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 计算每个生成 token 的对数概率
    log_probs = []
    for step, logits in enumerate(outputs.scores):  # 遍历所有时间步的 logits
        token_id = generated_tokens[step].item()  # 获取当前时间步的 token id
        token_log_probs = F.log_softmax(logits, dim=-1)  # 计算 log softmax
        log_probs.append(token_log_probs[0, token_id].item())  # 取出该 token 的对数概率

    if data.print_dialog:
        print(batch_dialog[0])
        print(response_text)
        print(log_probs)
        print("\n==================================\n")
    average_log_prob = sum(log_probs[:-1]) / len(log_probs[:-1])
    return [response_text], [log_probs[-2]], [average_log_prob]


def query_llama(generator, input_list, data, need_tqdm):
    longest_length = len(max(input_list, key=len))
    # print("longest_length:",longest_length)
    output_list = []
    output_logprobs_list = []
    output_logprobs2_list = []
    batch_size = data.max_batch_size
    batch_num = len(input_list)//batch_size
    if data.print_dialog:
        for i in range(batch_num + 1):
            if i < batch_num:
                batch_input_data = input_list[i * batch_size:(i + 1) * batch_size]
            else:
                batch_input_data = input_list[i * batch_size:len(input_list)]
            if len(batch_input_data) == 0:
                continue
            batch_dialog = text2dialog(batch_input_data)
            batch_output, batch_logprobs, batch_logprobs2 = gen_res(generator, batch_dialog, data)
            output_list += batch_output
            output_logprobs_list += batch_logprobs
            output_logprobs2_list += batch_logprobs2

    else:
        if need_tqdm:
            for i in tqdm(range(batch_num+1)):
                if i<batch_num:
                    batch_input_data = input_list[i*batch_size:(i+1)*batch_size]
                else:
                    batch_input_data = input_list[i*batch_size:len(input_list)]
                if len(batch_input_data)==0:
                    continue
                batch_dialog = text2dialog(batch_input_data)
                batch_output, batch_logprobs, batch_logprobs2 = gen_res(generator, batch_dialog, data)
                output_list += batch_output
                output_logprobs_list += batch_logprobs
                output_logprobs2_list += batch_logprobs2
        else:
            for i in range(batch_num+1):
                if i<batch_num:
                    batch_input_data = input_list[i*batch_size:(i+1)*batch_size]
                else:
                    batch_input_data = input_list[i*batch_size:len(input_list)]
                if len(batch_input_data)==0:
                    continue
                batch_dialog = text2dialog(batch_input_data)
                batch_output, batch_logprobs, batch_logprobs2 = gen_res(generator, batch_dialog, data)
                output_list += batch_output
                output_logprobs_list += batch_logprobs
                output_logprobs2_list += batch_logprobs2

    return output_list, output_logprobs_list, output_logprobs2_list