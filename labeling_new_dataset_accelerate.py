import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import regex as re
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler

B_INST, E_INST = "[INST]", "[/INST]"

class CustomDataset(Dataset):
    def __init__(self, dataset_name, dialogs, template):
        self.dataset_name = dataset_name
        self.dialogs = dialogs
        self.template = template

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        return dialog, idx

    def __len__(self):
        return len(self.dialogs)

    def map(self, function):
        self.dialogs = [function(item, self.template) for item in self.dialogs]
        return self


def merge_and_cleanup_output_files(path: str, world_size: int):
    """
    只在 rank 0 的进程中合并文件并删除原始文件。

    :param path: 文件所在的根目录。
    :param world_size: 进程的数量 (对应生成的文件数)。
    """
    # 仅在 rank 0 进程中执行合并操作
    rank = dist.get_rank()
    
    if rank == 0:
        print("Rank 0 is performing the merge operation...")
        
        output_text_all = []
        output_labels_all = []
        true_labels = []

        for rank in range(world_size):
            output_text_file = os.path.join(path, f'output_text_all_{rank}.pt')
            output_labels_file = os.path.join(path, f'output_labels_all_{rank}.pt')
            true_labels_file = os.path.join(path, f'true_labels_{rank}.pt')

            if os.path.exists(output_text_file):
                output_text_all.extend(torch.load(output_text_file))
                os.remove(output_text_file)
                print(f"Deleted file: {output_text_file}")
            else:
                print(f"File not found: {output_text_file}")

            if os.path.exists(output_labels_file):
                output_labels_all.extend(torch.load(output_labels_file))
                os.remove(output_labels_file)
                print(f"Deleted file: {output_labels_file}")
            else:
                print(f"File not found: {output_labels_file}")

            if os.path.exists(true_labels_file):
                true_labels.extend(torch.load(true_labels_file))
                os.remove(true_labels_file)
                print(f"Deleted file: {true_labels_file}")
            else:
                print(f"File not found: {true_labels_file}")

        # 保存合并结果
        unique_text_all = {x[0].item() if isinstance(x[0], torch.Tensor) else x[0]: x[1] for x in output_text_all}
        output_text_all = [unique_text_all[key] for key in sorted(unique_text_all.keys())]

        # import pdb;pdb.set_trace()
        
        unique_label_all = {x[0].item() if isinstance(x[0], torch.Tensor) else x[0]: x[1] for x in output_labels_all}
        output_labels_all = [unique_label_all[key] for key in sorted(unique_label_all.keys())]
        
        print(f"Unavailable labeling counting size: {output_labels_all.count(None)}")
        
        torch.save(output_text_all, os.path.join(path, 'output_text_all.pt'))
        torch.save(output_labels_all, os.path.join(path, 'output_labels_all.pt'))
        torch.save(true_labels, os.path.join(path, 'true_labels.pt'))
        print(f"Merged files saved to {path}")
    else:
        print(f"Rank {rank} is skipping the merge operation...")



def main(
    model_name: str = "llama",
    dataset_name: str = 'classified_data_final_w_worker_hash',
    prompt_template=4,
    data_size=3000,
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens=256,
    min_new_tokens: int = 0,
    prompt_file: str = None,
    seed: int = 42,
    token_gap: int = 0,
    root_path: str = 'logs',
    gpu_id: int = None,
    safety_score_threshold: float = 0.5,
    do_sample: bool = True,
    use_cache: bool = True,
    top_p: float = 0.9,
    temperature: float = 1.0,
    top_k: int = 50,
    batch_size: int = 10,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
    enable_azure_content_safety: bool = False,
    enable_sensitive_topics: bool = False,
    enable_saleforce_content_safety: bool = True,
    use_fast_kernels: bool = False,
    enable_llamaguard_content_safety: bool = False,
    target_token_idx: int = 0,
    top_g: int = 5,
    replace_token: bool = False,
    dataset_type: str = 'train',
    **kwargs
):
    import torch.distributed as dist

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    print(f"Process rank: {rank}, Total number of processes (world_size): {world_size}")

    # Set the device for each process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Adjust the seed for each process
    # seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    '''load model & tokenizer'''
    if model_name == 'llama':
        model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        system_prompt_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        user_prompt_start = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        batch_size = 10
    elif model_name == 'mistral':
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        
    elif model_name == 'gemma':
        # model_full_name = 'google/gemma-2b-it'
        model_full_name = 'google/gemma-2-2b-it'
        system_prompt_start = "<bos><start_of_turn>system"
        user_prompt_start = "<end_of_turn><bos><start_of_turn>user"
        assistant_prompt = "<end_of_turn><start_of_turn>model"
        batch_size = 20
        
    elif model_name == 'opt':
        model_full_name = 'facebook/opt-6.7b'
    else:
        raise NotImplementedError

    print(f'Loading LLM model: {model_full_name}')
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        # device_map={local_rank: device},
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    '''prompting'''
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
    if dataset_name == "classified_data_final_w_worker_hash":
        data_path = './raw_data/classified_data_final_w_worker_hash.json'

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
        input_text = f"{system_prompt_start}\n{pre_prompt}\n{user_prompt_start}\n{post_prompt}\n## comment: {comment}\n\n {assistant_prompt}"
        inputs.append(input_text)

    print(f"Total Data Size: {len(inputs)}")
    # inputs = inputs[:200]
    dataset = CustomDataset(dataset_name, inputs, template=prompt_template)

    # Use DistributedSampler to split data among processes
    sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    output_text_all = []
    output_labels_all = []
    true_labels = []

    for batch in tqdm(data_loader, desc=f"Generating inference on rank {rank}"):
        dialogs, indices = batch
        encodings = tokenizer(dialogs, padding=True, max_length=2048, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            output_text_batch = [(dialog_idx, tokenizer.decode(x, skip_special_tokens=True)) for dialog_idx, x in zip(indices, outputs)]
            output_labels = [None] * len(output_text_batch)

            for idx, (dialog_idx, output_text) in enumerate(output_text_batch):
                try:
                    retry_count = 10
                    while retry_count>0:
                        matches = json_pattern.findall(output_text)
                        if matches:
                            try:
                                json_obj = json.loads(matches[-1])
                                output_labels[idx] = (dialog_idx, json_obj)
                                break
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for input index: {idx}")
                                output_labels[idx] =  (dialog_idx, None)
                        else:
                            print(f"No JSON match for input index: {idx}")
                            output_labels[idx] = (dialog_idx, None)
                            
                        if output_labels[idx][1] is None:
                            single_encodings = tokenizer(dialogs[idx], padding=True, max_length=2048, truncation=True, return_tensors="pt").to(device)
                            with torch.no_grad():
                                output_text = model.generate(
                                    **single_encodings,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    top_p=top_p,
                                    temperature=temperature,
                                    use_cache=use_cache,
                                    top_k=top_k,
                                    repetition_penalty=repetition_penalty,
                                    length_penalty=length_penalty,
                                    **kwargs
                                )[0]
                                output_text = tokenizer.decode(output_text, skip_special_tokens=True)
                                # print(f"retrying count {retry_count}-- output-text: {output_text}")
                                # print(f"retrying count {retry_count}")

                        retry_count -= 1
                        
                    print("########################################################################################\n")
                    print(output_text)   
                    print("\n########################################################################################")
                        
                except Exception as exc:
                    print(f'dialog index: {idx} generated an exception: {exc}')
                    output_labels[idx] = (dialog_idx, None)

            output_labels_all.extend(output_labels)
            output_text_all.extend(output_text_batch)

    '''load parameters'''
    print('Storing parameters...')
    path = f"{root_path}/{dataset_name}/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Save outputs per rank
    torch.save(output_text_all, os.path.join(path, f'output_text_all_{rank}.pt'))
    torch.save(output_labels_all, os.path.join(path, f'output_labels_all_{rank}.pt'))
    torch.save(true_labels, os.path.join(path, f'true_labels_{rank}.pt'))

    print('Finished generation!')

    ### Combine them
    dist.barrier()
    merge_and_cleanup_output_files(path, world_size)
    dist.barrier()
    print(f"Rank {dist.get_rank()} has finished.")
    
    #check the len
    # assert len(output_labels_all) == len(inputs)
    # print(f"len(output_labels_all): {len(output_labels_all)};; len(inputs): {len(inputs)}")
    
if __name__ == '__main__':
    fire.Fire(main)
    
    
# CUDA_VISIBLE_DEVICES=0,1,3,4,7 torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py  --model_name gemma

# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 torchrun --nproc_per_node=6 labeling_new_dataset_accelerate.py  --model_name gemma 

# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 torchrun --nproc_per_node=6 labeling_new_dataset_accelerate.py --model_name gemma > zzz_gemma_labeling.log 2>&1