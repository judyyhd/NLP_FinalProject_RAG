import os
import sys
import argparse
import data_utils
import common_utils
from metrics import get_metrics

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        #load_in_8bit=True,   
        load_in_8bit=False,  
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Extract newly generated tokens (after prompt)
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return {"prompt": prompt, "generated": generated_text}  

def generate_rationale(args):
    data_path = f"dataset/{args.dataset_name}/hotpotqa_dev_500.json"
    train_data = common_utils.jload(data_path)[:args.max_instances]

    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path)
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_rationale_generation=True,
    )

    results = []
    for p, sample in zip(prompts, train_data):
        text = generate_text(model, tokenizer, p, args.max_tokens, args.temperature)
        results.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "rationale": text["generated"],
            "prompt": text["prompt"]
        })

    output_file = os.path.join(args.output_dir, "with_rationale/train.json")
    common_utils.jdump(results, output_file)
    print("Saved:", output_file)


def save_outputs(outputs, test_data, output_file, n_docs):
    output_data = []
    for i, output in enumerate(outputs):
        sample = test_data[i]

        ctxs = list(sample.get("ctxs", []))
        if any("score" in c for c in ctxs):
            ctxs = sorted(ctxs, key=lambda c: c.get("score", float("-inf")), reverse=True)
        ctxs = ctxs[:n_docs]

        output_data.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "qa_pairs": sample.get("qa_pairs", None),
            "rationale": output["generated"],
            "prompt": output["prompt"],
            "ctxs": ctxs,
        })

    common_utils.jdump(output_data, output_file)
    print("Saved:", output_file)
    return output_data


def eval_model(args):
    data_path = f"dataset/{args.dataset_name}/{args.devset_name}"
    # data_path = f"dataset/{args.dataset_name}/hotpotqa_dev_500.json"
    # data_path = f"dataset/{args.dataset_name}/test.json"
    test_data = common_utils.jload(data_path)[:args.max_instances]

    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path)
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        data_list=test_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
    )

    outputs = [
        generate_text(model, tokenizer, p, args.max_tokens, args.temperature)
        for p in prompts
    ]

    output_file = os.path.join(args.output_dir, "result.json")
    results = save_outputs(outputs, test_data, output_file, args.n_docs)

    get_metrics(results, args.output_dir, is_asqa=args.dataset_name == 'ASQA')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--devset_name', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--do_rationale_generation', action='store_true')
    parser.add_argument('--n_docs', type=int, default=5)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_instances', type=int, default=sys.maxsize)
    args = parser.parse_args()

    print("do rationale", args.do_rationale_generation)
    # print("rag_model", args.rag_model)
    eval_model(args)

    # if args.do_rationale_generation:
    #     generate_rationale(args)
    # else:
    #     eval_model(args)

