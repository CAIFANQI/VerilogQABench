import os
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict


ff=open("output_full_set.jsonl", 'r', encoding="utf-8")
json_list=[json.loads(i) for i in ff.read().rstrip("\n").split("\n")]
ff.close()

positive_template = '''
You are an expert who can extract key tokens based on questions and answers.
Can you extract some key tokens from the answers based on the following questions and corresponding correct answers?
The main criterion for extraction is that the statement containing the extracted token is likely to be the correct answer.
Question: {question}
Answer: {answer}
'''

negative_template = '''
You are an expert who can extract key tokens based on questions and answers.
Can you extract some key tokens from the answers based on the following questions and corresponding incorrect responses? 
The main criterion for extraction is that statements containing the extracted token are likely to have incorrect answers.
Question: {question}
Answer: {answer}
'''


def get_keyword(user_question):
    max_retry = 3
    retries = 0
    c = OpenAI(api_key="enter your api_key",
                       base_url="https://api.deepseek.com")

    while retries < max_retry:
        try:
            response = c.chat.completions.create(
                model='deepseek-chat',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant, please solve the user's question"},
                    {"role": "user", "content": user_question},
                ],
                max_tokens=2048,
                temperature=0.5,
                stream=False
            )
        except Exception as e:
            retries += 1
            if retries == max_retry:
                return ''
            print(f"{user_question} meets Error: {e} at {retries} times.")
        else:
            return response.choices[0].message.content


n = min(10, os.cpu_count() // 2)
positive_ls = []
negative_ls = []
idx_dict = defaultdict(list)
for index, ll_json in enumerate(json_list):
    min_score = ll_json['score'][0]
    min_index = 0
    for idx in range(len(ll_json['answer'])):
        if ll_json['answer'][idx]['accept']:
            positive_ls.append(positive_template.format(question=ll_json['question'],
                                                        answer=ll_json['answer'][idx]['body']))
            idx_dict[index].append(idx)
        else:
            if ll_json['score'][idx] < min_score:
                min_score = ll_json['score'][idx]
                min_index = idx
    negative_ls.append(negative_template.format(question=ll_json['question'],
                                                answer=ll_json['answer'][min_index]['body']))
    idx_dict[index].append(min_index)

k = len(json_list)
print(k)
positive_ls.extend(negative_ls)
print(len(positive_ls))
with ProcessPoolExecutor(max_workers=n) as executor:
    for idx, future in tqdm(enumerate(executor.map(get_keyword,positive_ls))):
        if idx < k:
            json_list[idx]['answer'][idx_dict[idx][0]]['keyw'] = future
        else:
            json_list[idx-k]['answer'][idx_dict[idx-k][1]]['keyw'] = future

with open("output_with_keyw_full_set.jsonl",'w',encoding="utf-8") as fout:
    for i in json_list:
        fout.write(json.dumps(i)+'\n')
