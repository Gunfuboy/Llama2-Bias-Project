from transformers import AutoTokenizer, pipeline
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler
import argparse
import json


def sample_model(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1028
    )
    return sequences[0]['generated_text']


if __name__ == "__main__":
    #Arguments
    parser = argparse.ArgumentParser(description='Get generated output')

    parser.add_argument(
        "--model_path",
        dest='model_path',
        help="model"
    )
    
    parser.add_argument(
        "--data_path",
        dest='data_path',
        help="data"
    )

    parser.add_argument(
        "--gen_version",
        dest='gen_version',
        help="version of the generation."
    )



    args = parser.parse_args()

    model = "/network/weights/llama.var/llama2/Llama-2-7b-chat-hf"


    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    data_path = args.data_path #"/home/mila/k/khaoula.chehbouni/home/NLP/final/prompts_data_v1.csv"
    df = pd.read_csv(data_path)


    gen_dic={}
    for i, row in df.iterrows():
        output=sample_model(row["new_prompt"])
        gen_dic[i] = output

    
    #Drop the dic
    filename = "/home/mila/k/khaoula.chehbouni/home/NLP/final/output_datav1_%s.json" % (args.gen_version)
    with open(filename, 'w') as fp:
        json.dump(gen_dic, fp)




