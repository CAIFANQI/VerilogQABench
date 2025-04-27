from nltk.translate.bleu_score import corpus_bleu
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm


ff=open("output_full_set.jsonl",'r',encoding="utf-8")
json_list=ff.read().rstrip("\n").split("\n")
ff.close()


fout=open("score_full_set.jsonl",'w',encoding="utf-8")
test_model = ['deepseek', 'doubao', 'qwen']
n = len(test_model)


def get_task_score(model_out, answers, scores):
    candidates = [model_out.split('\n')]
    max_score = max(scores)
    sims = [corpus_bleu([[ans['body'].split('\n')]], candidates, weights=(0.33, 0.33, 0.33, 0))
            for ans in answers]
    only_bleu = 0.0
    bleu_weight_with_score_max_norm = 0.0
    bleu_weight_with_score_norm, bleu_weight_without_score_norm = 0.0, 0.0
    for i in range(len(answers)):
        if answers[i]['accept']:
            only_bleu = sims[i]
        if scores[i] < 0.3:
            bleu_weight_with_score_norm += -sims[i]
            bleu_weight_with_score_max_norm += -sims[i]*(scores[i])/max_score
            bleu_weight_without_score_norm += -sims[i]*scores[i]
        else:
            bleu_weight_with_score_norm += sims[i]
            bleu_weight_with_score_max_norm += sims[i]*(scores[i])/max_score
            bleu_weight_without_score_norm += sims[i]*scores[i]
    return {'only_bleu': only_bleu,
            'bleu_weight_without_score_norm': bleu_weight_without_score_norm,
            "bleu_weight_with_score_norm": bleu_weight_with_score_norm,
            "bleu_weight_with_score_max_norm": bleu_weight_with_score_max_norm}

with ThreadPoolExecutor(max_workers=n) as executor:
    for ll in tqdm(json_list):
        ll_json = json.loads(ll)
        # print(ll_json.keys())
        # print("---------------------")
        # print(ll_json["Title"])
        # print(ll_json["Body"])
        # print("-----")
        # print(ll_json["Response"])
        # tot+=1
        # if tot>20:
        #     break
        score_dict = {}
        # score = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
        # all_task = [executor.submit(get_task_score,
        #                             ll_json[f'{m}_out'],
        #                             ll_json['answer'],
        #                             ll_json['score']) for m in test_model]
        for idx, future in enumerate(executor.map(get_task_score,
                                                  [ll_json[f'{m}_out'] for m in test_model],
                                                  [ll_json['answer']]*n,
                                                  [ll_json['score']]*n)):
            # score = future.result()
            score_dict[f'{test_model[idx]}_score'] = future
        fout.write(json.dumps(score_dict) + "\n")
