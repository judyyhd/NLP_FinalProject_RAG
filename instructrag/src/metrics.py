import re, json, string
from tqdm import tqdm
import numpy as np

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False

def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['answers'], item["rationale"]))

        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)

def exact_match(pred, gold):
    # Remove leading "Answer" if it appears at the start
    if pred.lower().startswith("answer"):
        # Remove "Answer", and also any colon, dash, or whitespace after it
        pred = pred[len("answer"):].lstrip(":：-–— \n")  

    return normalize_answer(pred) == normalize_answer(gold) 

def f1_score(pred, gold):  
    pred_tokens = normalize_answer(pred).split() 
    gold_tokens = normalize_answer(gold).split()  
    if len(pred_tokens) == 0 or len(gold_tokens) == 0: 
        return 0.0  
    common = set(pred_tokens) & set(gold_tokens)  
    if len(common) == 0:  
        return 0.0  
    precision = len(common) / len(pred_tokens) 
    recall = len(common) / len(gold_tokens)  
    return 2 * precision * recall / (precision + recall)  

def compute_hotpotqa_em_f1(data): 
    total_em, total_f1 = 0, 0  
    for d in data:  
        pred = d["rationale"] 
        golds = d["answers"]  
        em = max(exact_match(pred, g) for g in golds)  
        f1 = max(f1_score(pred, g) for g in golds)  
        total_em += em  
        total_f1 += f1  
    n = len(data)  
    return (total_em / n) * 100, (total_f1 / n) * 100


def get_metrics(data, save_dir=None, is_asqa=False):
    idx = 0
    num_accurate = 0
    print('Evaluating results...')
    if is_asqa:
        rationale_str_em, _ = compute_str_em(data)
    else:
        for d in tqdm(data):
            idx += 1
            is_accurate = exact_presence(d['answers'], d['rationale'])
            num_accurate += 1 if is_accurate else 0

    if is_asqa:
        print(f"Rationale EM: {rationale_str_em:.1f}%")
        eval_result = {"EM": rationale_str_em, "num_examples": idx}
    else:
        accuracy = num_accurate / idx * 100
        print(f"Accuracy: {accuracy:.1f}%")

        em, f1 = compute_hotpotqa_em_f1(data) 
        print(f"EM: {em:.1f}%")  
        print(f"F1: {f1:.1f}%")

        eval_result = {"accuracy": accuracy, "EM": em, "F1": f1, "num_examples": idx}
    
    with open(f"{save_dir}/metrics.json", "w") as f:
        f.write(json.dumps(eval_result) + "\n")   

    return eval_result