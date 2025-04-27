import os
from openai import OpenAI
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from tqdm import tqdm
import numpy as np

# ff = open("verilog_stackoverflow_0405_test_set.jsonl", 'r', encoding="utf-8")
# content_file = ff.read().rstrip("\n").split("\n")
# ff.close()
# tot = 0

test_model = ['deepseek', 'doubao', 'zhipu',
              'gpt', 'claude', 'xunfei',
              'llama', 'baidu', 'mistral', 'qwen']
n = len(test_model)


def get_model_response(m, q, cur_samples=None):
    n_sample = 10
    cur_sample_num = 0 if not cur_samples else len(cur_samples)
    samples = [] if not cur_samples else cur_samples

    max_retry = 3
    delay = 1
    retries = 0

    if m == 'deepseek':
        c = OpenAI(api_key="enter your api key",
                           base_url="https://api.deepseek.com")
        model = 'deepseek-chat'
    elif m == 'doubao':
        c = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="enter your api key")
        model = 'doubao-1-5-pro-32k-250115'
    elif m == 'zhipu':
        from zhipuai import ZhipuAI
        c = ZhipuAI(api_key="enter your api key")
        model = 'glm-4-plus'
    elif m == 'gpt' or m == 'claude':
        c = OpenAI(base_url="https://xiaoai.plus/v1",
                # base_url="https://api.ai-gaochao.cn/v1/",
                api_key="enter your api key",
                   )
        model = 'gpt-4' if m == 'gpt' else 'claude-3-5-sonnet-20241022'
    elif m == 'xunfei':
        c = OpenAI(
            api_key="enter your api key",
            base_url='https://spark-api-open.xf-yun.com/v1')
        model = "generalv3"
    elif m == 'llama' or m == 'baidu':
        c = OpenAI(api_key="enter your api key",
                   base_url="https://qianfan.baidubce.com/v2")
        model = "ernie-4.5-8k-preview" if m == 'baidu' else "llama-4-maverick-17b-128e-instruct"
    elif m == 'mistral':
        from mistralai import Mistral
        c = Mistral(api_key="enter your api key")
        model = "codestral-mamba-latest"
    else:
        c = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="enter your api key")
        model = 'qwen-coder-plus'

    while cur_sample_num < n_sample:
        while retries < max_retry:
            try:
                if m != 'mistral':
                    response = c.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a verilog expert, please solve the user's problem"},
                            {"role": "user", "content": q},
                        ],
                        max_tokens=2048,
                        temperature=0.5,
                        stream=False,
                    )
                else:
                    response = c.chat.complete(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a verilog expert, please solve the user's problem"},
                            {"role": "user", "content": q},
                        ],
                        max_tokens=2048,
                        temperature=0.5,
                        stream=False,
                    )
            except Exception as e:
                retries += 1
                if retries == max_retry:
                    samples.append('')
                    retries = 0
                    cur_sample_num += 1
                    break
                time.sleep(delay)
                print(f"{m} get Error: {e} at {retries} tries. Retrying in {delay} seconds...")
            else:
                samples.append(response.choices[0].message.content)
                retries = 0
                cur_sample_num += 1
                break
    return samples


def get_samples_for_task(ll_json):
    sorted_idx = np.argsort(np.array(ll_json['score']))
    negative_idx = len(sorted_idx)
    positive_idx = 0
    reference_answers = []
    reference_scores = []
    for i in range(len(sorted_idx)):
        if 'keyw' in ll_json['answer'][i]:
            reference_answers.append(ll_json['answer'][i])
            reference_scores.append(ll_json['score'][i])
        else:
            positive_idx = i
            negative_idx = min(i, negative_idx)
    positive_example = ll_json['answer'][sorted_idx[positive_idx]]['body']
    negative_example = ll_json['answer'][sorted_idx[negative_idx]]['body']
    ll_json['positive_out'] = positive_example
    ll_json['negative_out'] = negative_example
    question = ll_json['question']
    with ThreadPoolExecutor(max_workers=n) as executor:
        for idx, future in enumerate(executor.map(get_model_response, test_model, [question] * n)):
            # print(ll_json.keys())
            # print(ll_json["Title"])
            # print(ll_json["Body"])
            # print(response.choices[0].message.content)
            ll_json[f'{test_model[idx]}_out'] = future
    return ll_json


# ff=open("output_with_keyw_full_set_new.jsonl",'r',encoding="utf-8")
ff=open("ultra_output_full_set_new.jsonl")  # 补齐没有成功生成的输出
json_list=[json.loads(ll) for ll in ff.read().rstrip("\n").split("\n")]
ff.close()
fout=open("ultra_output_full_set_new_0415.jsonl",'w',encoding="utf-8")
parallel_task = []
mistral_task = 0
for idx, ll_json in tqdm(enumerate(json_list)):
    for k in ll_json:
        if k.endswith("_out"):
            model_name = k[:-4]
            if ll_json[k] == "" or len([j for j in ll_json[k] if j != ""]) < 10:
                if model_name == "mistral":
                    mistral_task += 1
                    json_list[idx][k] = get_model_response(model_name, ll_json['question'])
                else:
                    parallel_task.append((idx, model_name))
print('mistral: ', mistral_task)
print('other: ', len(parallel_task))
with ThreadPoolExecutor(max_workers=len(parallel_task)) as executor:
    for idx, future in tqdm(enumerate(executor.map(get_model_response, [o[1] for o in parallel_task],
                                              [json_list[o[0]]['question'] for o in parallel_task]))):
        # print(ll_json.keys())
        # print(ll_json["Title"])
        # print(ll_json["Body"])
        # print(response.choices[0].message.content)
        json_list[parallel_task[idx][0]][f'{parallel_task[idx][1]}_out'] = future
# with ProcessPoolExecutor(max_workers=min(14, os.cpu_count() // 2)) as pe:
#     for ret in tqdm(pe.map(get_samples_for_task, [json.loads(ll) for ll in json_list])):
#         output_ls.append(ret)

for od in json_list:
    fout.write(json.dumps(od) + "\n")
fout.close()



        # reference_indexes = [idx for idx in range(len(ll_json['answer']))if 'keyw' in ll_json['answer'][idx]]
        # reference_answers = [ll_json['answer'][idx] for idx in reference_indexes]
        # reference_scores = [ll_json['score'][idx] for idx in reference_indexes]

        # fout.write(json.dumps(ll_json)+'\n')
# fout = open("output_test_set_100.jsonl", 'w', encoding="utf-8")
# test_set_size = 100
# process_bar_total = 100
# process_update = process_bar_total / test_set_size
#     for ll in content_file:
#         ll_json = json.loads(ll)
        # question = ''
        # answer = ''
        # for message in ll_json['messages']:
        #     if 'question' in message:
        #         question += message['question']
        #     elif 'answer' in message:
        #         answer += message['answer']
        # all_task = [executor.submit(get_model_response, c, m, ll_json['question']) for c, m in zip(client_ls, test_model)]
