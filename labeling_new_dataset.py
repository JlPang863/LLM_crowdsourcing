import torch
import fire
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
from functools import partial
from torch.utils.data import DataLoader,Dataset
# from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from accelerate.utils import is_xpu_available
# from typing import Iterable, List, Optional, Tuple
from torch import Tensor
import regex as re

from datasets import load_dataset

### store the model 

B_INST, E_INST = "[INST]", "[/INST]"


class CustomDataset(Dataset):
    def __init__(self, dataset_name, dialogs, template):
        self.dataset_name = dataset_name
        self.dialogs = dialogs
        self.template = template
    def __getitem__(self, idx):
        # features: ['Tweet', 'Target', 'Stance'],

        dialog = self.dialogs[idx]
        return dialog

    
    def __len__(self):
        return len(self.dialogs)
    
    def map(self, function):
        self.dialogs = [function(item, self.template) for item in self.dialogs]
        return self
    


def read_dialogs_from_json(file: str,  data_size):
    dialogs = []
    # The number of dialogs
    # halueval
    # end_dialog_idx = 1010
    # start_dialog_idx = 10
    
    start_dialog_idx = 0
    # end_dialog_idx = 10000

    end_dialog_idx = data_size
    
    dialog_idx = 0  # Start counting from 0 to correctly match the first dialog as index 1
    with open(file, 'r') as f:
        for line in f:
            # Skip comment lines
            if not line.strip() or line.strip().startswith("//"):
                continue  # Skip the line if it's a comment or blank


            if start_dialog_idx <= dialog_idx <  end_dialog_idx:
                # This point can use pdb for debugging or directly process the data
                dialog_data = json.loads(line)
                user_query = dialog_data["user_query"]
                chatgpt_response = dialog_data['chatgpt_response']
                hallucination_label = dialog_data['hallucination']  # 'yes' or 'no'
                # hallucination_spans = dialog_data.get('hallucination_spans', [])  # Use get to handle missing fields

                # Construct the dialog dictionary
                dialog = [{
                    # "role": "user",
                    "content": user_query,
                    "chatgpt_response": chatgpt_response,
                    "hallucination_label": hallucination_label,
                    # "hallucination_spans": hallucination_spans
                }]
                
                dialogs.append(dialog)
            elif dialog_idx > end_dialog_idx:
                # Stop reading the file if the end of the target range is reached
                break
            
            
            dialog_idx += 1  # Increment dialog index only for non-comment lines

    return dialogs


def create_prompt_formats(dialog, template):
    """
    Format various fields of the dialog ('content', 'chatgpt_response')
    Then concatenate them using two newline characters: pre_prompt & post_prompt
    """
    pre_prompt = (
        "As a stance evaluator, your task is to determine the stance of the tweet towards the target mentioned. The stance should be classified as one of the following: 'AGAINST', 'NONE', or 'FAVOR'."
        "There are some examples:\n"
        "Example 1::\n"
        "### Tweet: saw @ChemBros at the Essential Festival in Brighton too #khole #SemST\n"
        "### Target: Atheism\n"
        "### Stance: NONE\n\n"
        "Example 2::\n"
        "### Tweet: Sorry, Hillary's new normal folk image doesn't take away from Behgnazi & her 0 foreign policy successes as Secretary of State.\n"
        "### Target: Hillary Clinton\n"
        "### Stance: AGAINST\n\n"
        "Example 3::\n"
        "### Tweet: @LilaGraceRose defunding PP won't mean fewer abortions. It will mean more dead women. #SemST\n"
        "### Target: Legalization of Abortion\n"
        "### Stance: FAVOR\n\n"
        "Now, please evaluate the following tweet:"
        )
    # typical question-answering type
    # dialog["Tweet"] = f"{B_INST}{pre_prompt}\n### Tweet: {dialog['Tweet']}\n### Target: {dialog['Target']}\n### Stance:{E_INST}"
    dialog["Tweet"] = f"{pre_prompt}\n### Tweet: {dialog['Tweet']}\n### Target: {dialog['Target']}\n#### Stance:"

    return dialog






def main(
    model_name: str = "llama",
    dataset_name: str = 'classified_data_final_w_worker_hash',
    # task_name: str = "classified_data_final_w_worker_hash",
    subset_name: str = None,
    prompt_template = 4, ### the prompt template
    data_size = 3000,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    token_gap: int=0,
    root_path: str='logs',
    gpu_id: int=None,
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    enable_llamaguard_content_safety: bool = False,
    target_token_idx: int = 0, 
    top_g: int=5,
    replace_token: bool= False,
    dataset_type: str = 'train', #train,validation, test
    **kwargs
):

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    '''load model & tokenizer'''
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # flash attention 2
    # torch.bfloat16
    # 8bit 4bit bitsandbytes
    # accelerate
    
    if model_name == 'llama':
        # model_full_name = "meta-llama/Llama-2-7b-chat-hf"
        # model_full_name = "meta-llama/Meta-Llama-3-8B"

        model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # model_full_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        system_prompt_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        user_prompt_start= "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
    elif model_name == 'mistral':
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        # model_full_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        
    elif model_name == 'gemma':
        model_full_name = 'google/gemma-2b-it'
        # model_full_name = 'google/gemma-7b-it'
        system_prompt_start = "<bos><start_of_turn>system"
        user_prompt_start= "<end_of_turn><bos><start_of_turn>user"
        assistant_prompt = "<end_of_turn><start_of_turn>model"
        
    elif model_name == 'opt':
        model_full_name = 'facebook/opt-6.7b'

    else:
        raise NotImplementedError
    
    print(f'Loading LLM model: {model_full_name}')
    device_map=f'cuda:{gpu_id}' if gpu_id is not None else 'auto'

    # import pdb;pdb.set_trace()
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        # torch_dtype=torch.bfloat16,
        # load_in_4bit=True,
        # attn_implementation="flash_attention_2", #accelerate
        # device_map="auto",
        # device_map="balanced",#"auto", "balanced", "balanced_low_0", "sequential"
        device_map=device_map,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # '''prompting'''
    # print("choose the prompt for model!!")
    # pre_prompt = prompting(model_name)

    pre_prompt = (
        '''You are a helpful assistant, and your task is to evaluate and analyze the given comment taken from various online platforms via questions such that we can clean the platforms.\n
        Here are the questions.
        ## Question 1: How toxic do you personally find this comment?\n
        * Extremely toxic, toxic score: 4\n
        * Very toxic, toxic score: 3\n
        * Moderately toxic, toxic score: 2\n
        * Slightly toxic, toxic score: 1\n
        * Not at all toxic, toxic score: 0\n\n
        ## Question 2: [If any but not at all] Please select all the reasons why below: [using tags: True or False]
        o It is threatening,\n
        o It is insulting\n
        o It is profane\n
        o It is an attack on a person or group based on their identity\n
        o It is sexual harassment\n\n
        
        ## Question 3: Is this a type of comment you would want to see? [Radio]\n
        o I would never want to see it, tag: 1\n
        o It depends on the context, tag: 2\n
        o This is fine for me to see, tag: 3\n\n
        '''
        )
    
    post_prompt = (
        '''
        Now, please evaluate the following comment and return the final answers in the json format:\n
        {
            "question 1": {
                "toxic score": <number>
            },
            "question 2": {
                "threatening": <true/false>,
                "insulting": <true/false>,
                "profane": <true/false>,
                "attack": <true/false>,
                "sexual harassment": <true/false>
            },
            "question 3": {
                "tag": <number>
            }
        }
        '''
        )


    '''preprocess dataset'''
    print("Preprocessing dataset...")
    ##########################################################################################
    # dataset_name = "krishnagarg09/SemEval2016Task6"
    # data = load_dataset(dataset_name)
    # # features: ['Tweet', 'Target', 'Stance'],
    # dialogs = data[dataset_type]
    if dataset_name == "classified_data_final_w_worker_hash":
        data_path = './raw_data/classified_data_final_w_worker_hash.json'
        
    # data_path = './raw_data/classified_data_final_w_worker_hash.json'
    # raw_dataset = load_dataset('json', data_files=data_path, split=None, cache_dir=None)

    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_data.append(data)

    comments = [raw_data[idx]['comment'] for idx in range(len(raw_data))]
    human_labels = []
    for idx in range(len(raw_data)):
        labels = []
        for item in raw_data[idx]['ratings']:
            labels.append(item['toxic_score'])
        human_labels.append(labels)
        
    inputs = []
    for comment in comments:
        # input = f"{pre_prompt}\n## comment: {comment}\n{post_prompt}"
        # input = f"{system_prompt_start}\n{pre_prompt}\n{user_prompt_start}\n{post_prompt}\n## comment: {comment}"# \n\n {assistant_prompt}
        input = f"{system_prompt_start}\n{pre_prompt}\n{user_prompt_start}\n{post_prompt}\n## comment: {comment}\n\n {assistant_prompt}"# \n\n {assistant_prompt}

        inputs.append(input)
        
    # inputs = inputs[:100]
    
    dataset = CustomDataset(dataset_name, inputs, template=prompt_template)
    # dataset = dataset.map(create_prompt_formats)
    data_loader = DataLoader(dataset, batch_size=50, shuffle=False) #, shuffle=True, seed=42 
    
    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')


    output_text_all = []
    output_labels_all = []
    true_labels = []
    for batch in tqdm(data_loader, desc="Generating inference info for answers"):
        dialogs = batch
        encodings = tokenizer(dialogs, padding=True, max_length=2048, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # for input_ids, attention_mask in tqdm(zip(encodings["input_ids"], encodings['attention_mask'])):
                # import pdb;pdb.set_trace()
                # input_ids = input_ids.unsqueeze(0)  # 添加批次维度
                # attention_mask = attention_mask.unsqueeze(0)  # 添加批次维度
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            # output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            # output_answer_text_batch = [tokenizer.decode(x[attention_mask.shape[1]:], skip_special_tokens=True) for x in outputs]
            output_labels = [None] * len(dialogs) 


            for idx, output_text in enumerate(output_text_batch):
                # print("########################################################################################\n")
                # print(output_text)   
                # print("\n########################################################################################")
                # import pdb;pdb.set_trace()

                #extract rating score
                # match = re.search(r"#### Stance: (\w+)", output_text)
                # stance = match.group(1) if match else "UNKNOWN"
                try:
                    retry_count = 3
                    while retry_count>0:

                        matches = json_pattern.findall(output_text)
                        if matches:
                            try:
                                # 解析并保存完整的 JSON 对象
                                json_obj = json.loads(matches[-1])
                                output_labels[idx] = json_obj
                                break  # 成功提取到 JSON，退出循环
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for input {dialogs[idx]}")
                                output_labels[idx] = None
                        else:
                            print(f"No JSON match for input {dialogs[idx]}, recalculating...")
                            
                        retry_count -= 1
                except Exception as exc:
                    print(f'{dialogs[idx]} generated an exception: {exc}')
                    output_labels[idx] = None        
                
                # output_labels.append(stance)
            output_labels_all.extend(output_labels)
            output_text_all.extend(output_text_batch)


    '''load parameters'''
    print('Storing parameters...')
    # if subset_name is not None: 
    #     path = os.path.join(root_path, model_name, f"{dataset_name}-{subset_name}", dataset_type)
    # else:
    #     path = os.path.join(root_path, model_name, dataset_name, dataset_type)

    path = f"{root_path}/{dataset_name}/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
            
    torch.save(output_text_all, path + f'/output_text_all.pt')
    torch.save(output_labels, path + f'/output_labels_all.pt')
    torch.save(true_labels, path + f'/true_labels.pt')

    # import pdb;pdb.set_trace()

    print('Finished generation!')
    
    
    
    

if __name__ == '__main__':
    fire.Fire(main)
    
    
    