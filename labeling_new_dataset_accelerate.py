import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import fire
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import regex as re
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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


def merge_and_cleanup_chunk_files(path: str, split_size: int):

    # 仅在 rank 0 进程中执行合并操作
    rank = dist.get_rank()
    
    if rank == 0:
        print("Rank 0 is performing the merge operation...")
        
        output_text_all = []
        output_labels_all = []
        true_labels = []

        for chunk_idx in range(split_size):
            output_text_file = os.path.join(path, f'chunk-{chunk_idx}', f'output_text_all.pt')
            output_labels_file = os.path.join(path, f'chunk-{chunk_idx}', f'output_labels_all.pt')
            true_labels_file = os.path.join(path, f'chunk-{chunk_idx}', f'true_labels.pt')

            if os.path.exists(output_text_file):
                output_text_all.extend(torch.load(output_text_file))
                # os.remove(output_text_file)
                # print(f"Deleted file: {output_text_file}")
            else:
                print(f"File not found: {output_text_file}")

            if os.path.exists(output_labels_file):
                output_labels_all.extend(torch.load(output_labels_file))
                # os.remove(output_labels_file)
                # print(f"Deleted file: {output_labels_file}")
            else:
                print(f"File not found: {output_labels_file}")

            if os.path.exists(true_labels_file):
                true_labels.extend(torch.load(true_labels_file))
                # os.remove(true_labels_file)
                # print(f"Deleted file: {true_labels_file}")
            else:
                print(f"File not found: {true_labels_file}")

        print(f"Unavailable labeling counting size: {output_labels_all.count(None)}")
        
        torch.save(output_text_all, os.path.join(path, 'output_text_all.pt'))
        torch.save(output_labels_all, os.path.join(path, 'output_labels_all.pt'))
        torch.save(true_labels, os.path.join(path, 'true_labels.pt'))
        print(f"Merged files saved to {path}")
    else:
        print(f"Rank {rank} is skipping the merge operation...")


def merge_and_cleanup_output_files(path: str, world_size: int):

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



def data_processing(task_name, system_prompt, user_prompt, assistant_prompt):
    ## SemEva2016 dataset
    if task_name == 'SemEval2016':

        pre_prompt = (
            "As a stance evaluator, your task is to answer the following questions for the comments taken from various online platforms. Some of these comments may contain toxic content.\n"
            # "There are some examples:\n"
            # "Example 1::\n"
            # "## Comment: Just a matter of time before pick up on this story.\n"
            # "### Question: How toxic do you personally find this comment?\n"
            # "* Extremely toxic, toxic score: 4\n"
            # "* Very toxic, toxic score: 3\n"
            # "* Moderately toxic, toxic score: 2\n"
            # "* Slightly toxic, toxic score: 1\n"
            # "* Not at all toxic, toxic score: 0\n"
            # "### Toxic score: 0\n\n"
            # "Example 2::\n"
            # "## Comment: this is QUINN you DUMBASS 😭😭😭\n"
            # "### Question: How toxic do you personally find this comment?\n"
            # "* Extremely toxic, toxic score: 4\n"
            # "* Very toxic, toxic score: 3\n"
            # "* Moderately toxic, toxic score: 2\n"
            # "* Slightly toxic, toxic score: 1\n"
            # "* Not at all toxic, toxic score: 0\n\n"
            # "### Toxic score: 3\n\n"
            "Now, please evaluate the following comment by return the toxic score.\n"
        )
        inputs = []
        for comment in comments:
            input = f"{system_prompt}\n{pre_prompt}\n{user_prompt}\n{post_prompt}\n## comment: {comment}\n\n {assistant_prompt}"# \n\n {assistant_prompt}
            inputs.append(input)
    ############################################################################################################################################
    # ###classified_data_final_w_worker_hash.json dataset
    
    elif task_name == "classified_data_final_w_worker_hash":
        data_path = './raw_data/classified_data_final_w_worker_hash.json'

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
        

        # data = {
        #     'comments': comments
        # }
        # data = Dataset.from_dict(data)
        
        # data = data.map(prompt_template)
        
        
        pre_prompt = (
            f'''As a helpful system evaluator, your task is to answer the following questions for the given comment taken from various online platforms. Some of these comments may contain toxic content.\n
            ## Question 1: How toxic do you personally find this comment?\n
            * Extremely toxic, toxic score: 4\n
            * Very toxic, toxic score: 3\n
            * Moderately toxic, toxic score: 2\n
            * Slightly toxic, toxic score: 1\n
            * Not at all toxic, toxic score: 0\n\n
            
            ## Question 2: [If any but not at all] Please select all the reasons why below: [Checkmark]
            o It is threatening (ex. “If you comment on my feed again, I will personally beat you up”)\n
            o It is insulting (ex. “You are an absolutely disgusting hypocrite”)\n
            o It is profane (ex. “Why the f*** not you heathen?”)\n
            o It is an attack on a person or group based on their identity (ex. “Women are stupid and belong in the kitchen”)\n
            o It is sexual harassment (ex. “All I want from you are nudes”)\n\n
            
            ## Question 2: Is this a type of comment you would want to see? [Radio]\n
            o I would never want to see it, tag: 1\n
            o It depends on the context, tag: 2\n
            o This is fine for me to see, tag: 3\n\n
            '''
            )
        
        post_prompt = (
            '''
            Now, please evaluate the following comment and summarily return the final answers for the above questions in the following json format:
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
        inputs = []
        for comment in comments:
            input = f"{system_prompt}\n{pre_prompt}\n{user_prompt}\n{post_prompt}\n## comment: {comment}\n\n {assistant_prompt}"# \n\n {assistant_prompt}
            inputs.append(input)        
    
    ##################################################################################################


    elif task_name == 'text_for_labeling':
        data_path = 'raw_data/text_for_labeling.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        comments = raw_data
        pre_prompt = (
            f'''As a helpful system judgement assistant, your task is to answer the following questions for the given comment.\n
            You should label a comment as hateful if it expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group.
            You should label a comment as offensive if it contains hurtful, derogatory or obscene comments.
            You should label a comment as toxic if it is a rude,disrespectful, or unreasonable comment that is likely to make readers want to leave a discussion.\n\n
            ## Question 1: Do you think this comment is hateful?\n
            o Yes\n
            o No\n\n
            ## Question 2: Do you think this comment is offensive?\n
            o Yes \n
            o No\n\n
            ## Question 3: Do you think this comment is toxic?\n
            o Yes\n
            o No\n\n
            '''
            )
        
        post_prompt = (
                '''
                Now, please evaluate the following comment and summarily return the final answers for the above questions in the following json format:
            {
                "question 1": {
                    "hateful": <yes/no>
                },
                "question 2": {
                    "offensive": <yes/no>,
                },
                "question 3": {
                    "toxic": <yes/no>,
                }
            }
                '''
                )
        inputs = []
        for comment in comments:
            input = f"{system_prompt}\n{pre_prompt}\n{user_prompt}\n{post_prompt}\n## comment: {comment}\n\n {assistant_prompt}"# \n\n {assistant_prompt}
            inputs.append(input)
            
    ##################################################################################################
    
    elif task_name == "multipref": #####
        pre_prompt = ('''Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, harmless, truthfulness, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
            After providing your explanation, output your final verdict by strictly based on a 5-point Likert scale options: \n
            o A-is-clearly-better\n
            o A-is-slightly-better\n 
            o Tie\n
            o B-is-slightly-better\n
            o B-is-clearly-better\n
            ''')
            
        post_prompt = (
                '''
                Now, please evaluate the responses and summarily return the final verdict in the following json format:
            {
                "final verdict": <A-is-clearly-better / A-is-slightly-better / Tie / B-is-slightly-better / B-is-clearly-better>
                "short explanation": <your short explanation within one sentences>
            }
                '''
                )
        
        raw_dataset = load_dataset("allenai/multipref")['train']
        # raw_dataset = load_dataset("allenai/multipref")['train'].select(range(100))
        questions, answer_a, answer_b = [], [], []

        for sample in raw_dataset:
            questions.append(sample['text'])
            answer_a.append(sample['completion_a'])
            answer_b.append(sample['completion_b'])
        
        # new_dataset = DatasetDict({
        #     "train": Dataset.from_dict({
        #         "question": questions,
        #         "answer_a": answer_a,
        #         "answer_b": answer_b
        #     })
        # })
        
        inputs = []
        for ques, ans_a, ans_b in zip(questions, answer_a, answer_b):
            input = f"{system_prompt}\n{pre_prompt}\n{user_prompt}\n{post_prompt}\n### Question: {ques}\n\n ### Response A: {ans_a}\n\n ### Response B: {ans_b}\n\n{assistant_prompt}"
            inputs.append(input)  
              
    else:
        print("Task/dataset not found!")    
        
        
    return inputs




def main(
    model_name: str = "llama",
    dataset_name: str = 'classified_data_final_w_worker_hash',
    prompt_template=4,
    data_size=3000,
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens=128,
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
    **kwargs
):

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
    if 'llama' in model_name.lower():
        model_full_name = "meta-llama/Llama-2-7b-chat-hf"
        system_prompt = "<s>[INST] <<SYS>>\n"
        user_prompt = "<</SYS>>\n\n"
        assistant_prompt = " [/INST]"


        # model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        # user_prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        # assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        batch_size = 5
        
    elif 'mistral' in model_name.lower():
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        system_prompt = "<s>[INST]system"
        user_prompt = "[/INST][INST]user"
        assistant_prompt = "[/INST][INST]assistant"
        batch_size = 5
        
    elif 'gemma' in  model_name.lower():
        if 'gemma-9b' in model_name.lower():
            model_full_name = 'google/gemma-2-9b-it' ##3
            batch_size = 2
            
        elif 'gemma-2b' in model_name.lower():
            model_full_name = 'google/gemma-2-2b-it' ## 10
            batch_size = 10
        else:
            raise NotImplementedError
        
        system_prompt = "<bos><start_of_turn>system"
        user_prompt = "<end_of_turn><bos><start_of_turn>user"
        assistant_prompt = "<end_of_turn><start_of_turn>model"
        
    elif "opt" in model_name.lower():
        model_full_name = 'facebook/opt-6.7b'
        system_prompt = "</s>"
        user_prompt = ""
        assistant_prompt = ""
        batch_size = 5        
        
        
    elif "phi" in model_name.lower():
        model_full_name = "microsoft/Phi-3.5-mini-instruct"
        # model_full_name = "microsoft/Phi-3.5-MoE-instruct"
        system_prompt = "<|system|>"
        user_prompt = "<|end|>\n <|user|>"
        assistant_prompt = "<|end|>\n <|assistant|>"
        batch_size = 5
        
        
    else:
        raise NotImplementedError

    print(f'Loading LLM model: {model_full_name}')
    model_name = os.path.basename(model_full_name)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        torch_dtype=torch.bfloat16,
        # quantization_config = bnb_config,
        # device_map={local_rank: device},
    # )
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token


    ## processing inputs, e.g., chat template
    print(f"dataset_name: {dataset_name}")
    inputs = data_processing(dataset_name, system_prompt, user_prompt, assistant_prompt)
    print(f"Total Data Size: {len(inputs)}")
    # import pdb;pdb.set_trace()
    
    chunk_size = 1000 # the batch_size
    split_size = len(inputs) // chunk_size + (1 if len(inputs) % chunk_size != 0 else 0)
    
    
    for chunk_idx in tqdm(range(split_size), desc=f'{model_name} labeling'):
        chunk_end =  min((chunk_idx+1) * chunk_size, len(inputs)) # the end index of batch
        chunk_inputs = inputs[chunk_idx* chunk_size:chunk_end] ##data range
            
        dataset = CustomDataset(dataset_name, chunk_inputs, template=prompt_template)

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
                        #     print("########################################################################################\n")
                        #     print(output_text)   
                        #     print("\n########################################################################################")
                            
                        # print("########################################################################################\n")
                        # print(output_text)   
                        # print("\n########################################################################################")
                            
                    except Exception as exc:
                        print(f'dialog index: {idx} generated an exception: {exc}')
                        output_labels[idx] = (dialog_idx, None)

                output_labels_all.extend(output_labels)
                # output_text_all.extend(output_text_batch)
                
            del  output_text_batch

            torch.cuda.empty_cache()


        path = f"{root_path}/{dataset_name}/{model_name}/chunk-{chunk_idx}"
        if not os.path.exists(path):
            os.makedirs(path)

        # Save outputs per rank
        torch.save(output_text_all, os.path.join(path, f'output_text_all_{rank}.pt'))
        torch.save(output_labels_all, os.path.join(path, f'output_labels_all_{rank}.pt'))
        torch.save(true_labels, os.path.join(path, f'true_labels_{rank}.pt'))

        print('Finished generation!')
        
        '''load parameters'''
        print('Storing parameters...')
         
        ### Combine them
        dist.barrier()
        merge_and_cleanup_output_files(path, world_size)
        dist.barrier()
        print(f"Rank {dist.get_rank()} has finished.")
        
        torch.cuda.empty_cache()
    
    ##merge chunk files
    merge_and_cleanup_chunk_files(f"{root_path}/{dataset_name}/{model_name}/", split_size)
    
    
if __name__ == '__main__':
    fire.Fire(main)
    
    
# CUDA_VISIBLE_DEVICES=0,1,3,4,7 torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py  --model_name gemma

# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 torchrun --nproc_per_node=6 labeling_new_dataset_accelerate.py  --model_name gemma 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 nohup torchrun --nproc_per_node=6 labeling_new_dataset_accelerate.py --model_name gemma  --dataset_name text_for_labeling > zzz_gemma_labeling.log &

# CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 nohup torchrun --nproc_per_node=6 labeling_new_dataset_accelerate.py --model_name mistral --dataset_name text_for_labeling > zzz_mistral_labeling.log &

# CUDA_VISIBLE_DEVICES=1,2,3,4,6 nohup torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py --model_name llama > zzz_llama_labeling.log &

# CUDA_VISIBLE_DEVICES=1,2,3,4,6 torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py --model_name llama


# CUDA_VISIBLE_DEVICES=0,1,2,3,4 nohup   torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py --model_name phi  --dataset_name classified_data_final_w_worker_hash > zzz_phi_labeling.log &

# CUDA_VISIBLE_DEVICES=1,2,4,6,7 nohup  torchrun --nproc_per_node=5 labeling_new_dataset_accelerate.py --model_name gemma  --dataset_name text_for_labeling > zzz_gemma_labeling.log &

# CUDA_VISIBLE_DEVICES=0,3,6,7  torchrun --nproc_per_node=4 labeling_new_dataset_accelerate.py --model_name opt  --dataset_name text_for_labeling 

# CUDA_VISIBLE_DEVICES=1,2,6,7 nohup torchrun --nproc_per_node=4 labeling_new_dataset_accelerate.py --model_name gemma  --dataset_name classified_data_final_w_worker_hash > zzz_gemma_labeling.log &