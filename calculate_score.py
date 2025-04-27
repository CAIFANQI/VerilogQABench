# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
from tqdm import tqdm
import re
import math
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from openai import OpenAI
from nltk.translate.bleu_score import corpus_bleu
import time


def generate_jsonl(dict_list, f_name):
    with open(f_name, 'wb') as f:
        for i in dict_list:
            f.write(bytes(json.dumps(i) + '\n', encoding='utf-8'))


strong_accept_regex = re.compile(r'(success)|(fix)|(solve)|(answer)', flags=re.I | re.M | re.S | re.U)
weak_accept_regex = re.compile(r'(thank)|(thx)|(help)|(useful)', flags=re.I | re.M | re.S | re.U)
client = OpenAI(api_key="enter your api key", base_url="https://api.deepseek.com")
max_retry = 3
delay = 1
max_worker_num = min(16, os.cpu_count() // 2)


# combine regular match and bleu to obtain the weight
def get_score(question: str, answer: Dict) -> float:
    retries = 0
    if answer['accept']:
        return 1.0 * 2.0 * max(1, answer['score'])
    if strong_accept_regex.search(answer['body']):
        return 1.0 * 2.0 * answer['score']  # * 2.0 # len(messages)
    if weak_accept_regex.search(answer['body']):
        return 0.5 * 2.0 * answer['score']

    while retries < max_retry:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a verilog expert, please fix the user's question"},
                    {"role": "user", "content": question},
                ],
                max_tokens=2048,
                temperature=0.5,
                stream=False
            )
        except Exception as e:
            retries += 1
            if retries == max_retry:
                return 0.0
            # time.sleep(delay)
            print(f"Error: {e}. Retrying in {delay} seconds...")
        else:
            model_answer = response.choices[0].message.content
            r = corpus_bleu([[model_answer.split('\n')]], [answer['body'].split('\n')])
            # 0.3 is threshold, which can be tuned according the performance
            if r < 0.3:
                return r * answer['score'] / 2  # len(messages)
            else:
                return r * 2.0 * answer['score']  # s * len(messages)}


def calculate_answer_score(question: str, answers: List[Dict]) -> List:
    score_ls = []
    with ThreadPoolExecutor(max_workers=max_worker_num) as executor:
        # tasks = [executor.submit(get_score, question, ans) for ans in answers]
        for fu in executor.map(get_score, [question]*len(answers), answers):
            # s = fu.result()
            score_ls.append(fu)
    return score_ls


if __name__ == "__main__":
    # It seems that every chat only has one turn at most, so it is useless to check the length of chat messages
    # nxt format {chat ID: ls_idx}, which means which block the chat belongs to, and nxt append at last
    nxt = {}
    # score format {ls_idx: score}, store the score of corresponding block's answer
    score = {}
    # acc format {ls_idx: accepted_id}, store the accepted_id of corresponding block's answer
    acc = {}
    # list of messages block
    chat_ls: List[Dict] = []
    test_n = 59
    tot_count = 0
    # first add all questions related to verilog into the chat_ls
    for i in tqdm(range(test_n)):
        df = pd.read_parquet('./stackoverflow-posts-{:0>5d}-of-00058.parquet'.format(i))
        features = ['Id', 'PostTypeId', 'AcceptedAnswerId', 'ParentId', 'Body', 'Tags']
        for row in tqdm(df.itertuples(index=False)):
            isT = False
            # isC = False
            # isR = False
            tmp_dict = {k: getattr(row, k) for k in features}
            if tmp_dict['PostTypeId'] != 1:
                continue
            for tag in tmp_dict['Tags']:
                lower_tag = tag.lower()
                if lower_tag == "verilog" or lower_tag == "hdl":
                    isT = True
                    break
                # if lower_tag == 'c++' or lower_tag == 'cpp':
                #     isC = True
                # if re.match('risc-?v', lower_tag):
                #     isR = True
                # if (isT and isC) or isR:
            if isT:
                tot_count += 1
                if math.isnan(tmp_dict['ParentId']) and not math.isnan(tmp_dict['AcceptedAnswerId']):
                    nxt[tmp_dict['Id']] = len(chat_ls)
                    acc[int(tmp_dict['AcceptedAnswerId'])] = len(chat_ls)
                    chat_ls.append({'question': tmp_dict['Body'], 'answer': []})
        del df
    print(tot_count)

    # add the corresponding answer into the chat_ls and record the score
    for i in tqdm(range(test_n)):
        df = pd.read_parquet('./stackoverflow-posts-{:0>5d}-of-00058.parquet'.format(i))
        features = ['Id', 'PostTypeId', 'ParentId', 'Score', 'Body']
        for row in tqdm(df.itertuples(index=False)):
            tmp_dict = {k: getattr(row, k) for k in features}
            if tmp_dict['PostTypeId'] != 2:
                continue
            if not math.isnan(tmp_dict['ParentId']) and int(tmp_dict['ParentId']) in nxt:
                idx = nxt[int(tmp_dict['ParentId'])]
                # if idx != acc[tmp_dict['Id']]:
                #     print('here is a index bug')
                #     idx = acc[tmp_dict['Id']]
                chat_ls[idx]['answer'].append({'body': tmp_dict['Body'],
                                               'accept': tmp_dict['Id'] in acc and acc[tmp_dict['Id']] == idx,
                                               'score': tmp_dict['Score']})
        del df
        
    print(len(chat_ls))
    # all_task = [executor.submit(calculate_answer_score, chat, score[idx]) for idx, chat in enumerate(chat_ls)]
    target_ls = []
    # for future in as_completed(all_task):
    #    data = future.result()
    threshold = 1
    # positive_size = 20
    # negative_size = 80
    # test_set_size = positive_size + negative_size
    positive_size = 1
    # negative_size = 2
    total_size = 4
    # test_set_size = 100
    process_bar_total = 100
    process_update = process_bar_total / len(chat_ls)  # test_set_size
    current_num = 0
    with tqdm(total=process_bar_total) as pbar:
        for idx, chat in enumerate(chat_ls):
            # negative_num = 0
            if len(chat['answer']) < total_size:
                pbar.update(process_update)
                continue
            positive_num = 0
            for a in chat['answer']:
                if a['accept']:
                   # negative_num += 1
                # else:
                    positive_num += 1
            # if negative_num >= negative_size and positive_num >= positive_size:
            if positive_num >= positive_size:
                target_ls.append(chat)
                current_num += 1
            pbar.update(process_update)
            #if current_num >= test_set_size:
            #    break
            # data = calculate_answer_score(chat['question'], chat['answer'])
            # if data['score'] > threshold and positive_num < positive_size:
            #     positive_num += 1
            #     target_ls.append(data)
            #     pbar.update(process_update)
            #     print('positive example')
            # if data['score'] <= threshold and negative_num < negative_size:
            #     negative_num += 1
            #     target_ls.append(data)
            #     pbar.update(process_update)
            #     print('negative example')
            # if positive_num >= positive_size and negative_num >= negative_size:
            #     break
    print(current_num)
    
    # use original score as metric
    for idx, data in enumerate(target_ls):
        target_ls[idx]['score'] = [ans['score'] for ans in data['answer']

    # use a new metric to transform the original score into a metric, and you can use the original score as well
    '''
    with ProcessPoolExecutor(max_workers=8) as pe:
        # all_task = [pe.submit(calculate_answer_score, chat['question'], chat['answer']) for chat in target_ls]
        for idx, future in tqdm(enumerate(pe.map(calculate_answer_score, [chat['question'] for chat in target_ls], [chat['answer'] for chat in target_ls]))):
            # s_ls = future.result()
            target_ls[idx]['score'] = future
    file_name = f'verilog_stackoverflow_0407_full_set.jsonl'
    generate_jsonl(target_ls, file_name)
    '''



            # elif not math.isnan(tmp_dict['ParentId']):
            #     user_type = 'user' if tmp_dict['PostTypeId'] == 1 else 'answer'
            #     parent_idx = -1
            #     son_idx = -1
            #     if tmp_dict['ParentId'] in nxt:
            #         parent_idx = nxt[tmp_dict['ParentId']]
            #         chat_ls[parent_idx].append({user_type: tmp_dict['Body']})
            #         nxt.pop(tmp_dict['ParentId'])
            #         nxt[tmp_dict['Id']] = parent_idx
            #     if tmp_dict['Id'] in pre:
            #         son_idx = pre[tmp_dict['Id']]
            #         if parent_idx != -1:
            #             chat_ls[parent_idx].extend(chat_ls[son_idx])
            #             nxt.pop(tmp_dict['Id'])
            #             for k, v in nxt.items():
            #                 if v == son_idx:
            #                     nxt[k] = parent_idx
            #                     break
            #             chat_ls.pop(son_idx)
            #         else:
            #             chat_ls[son_idx].insert(0, {user_type: tmp_dict['Body']})
            #             pre.pop(tmp_dict['Id'])
            #             pre[tmp_dict['ParentId']] = son_idx
            #     if parent_idx == -1 and son_idx == -1:
            #         pre[tmp_dict['ParentId']] = len(chat_ls)
            #         nxt[tmp_dict['Id']] = len(chat_ls)
            #         chat_ls.append([{user_type: tmp_dict['Body']}])
            #     if parent_idx != -1 and tmp_dict['PostTypeId'] == 2 and tmp_dict['Id'] == acc[parent_idx]:
            #         score[parent_idx] = tmp_dict['Score']
