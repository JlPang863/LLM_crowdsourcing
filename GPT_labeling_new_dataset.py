
from openai import OpenAI
import regex as re
import torch
import fire
from datasets import load_dataset, Dataset
import os
import json



'''API key'''

# openai.api_key = 'your-api-key'
# client = openai.OpenAI(api_key=api_key)

# export OPENAI_API_KEY=your_actual_api_key
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key) # API key from Docta.ai

# echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
# source ~/.bashrc



def prompt_template(conversation):
    
    content = (
        f'''As a stance evaluator, your task is to answer the following questions for the comments taken from various online platforms. Some of these comments may contain toxic content.\n
        "Now, please evaluate the following comment
        {conversation} and answer these questions:\n
        ## Question 1: How toxic do you personally find this comment?\n
        * Extremely toxic, toxic score: 4\n
        * Very toxic, toxic score: 3\n
        * Moderately toxic, toxic score: 2\n
        * Slightly toxic, toxic score: 1\n
        * Not at all toxic, toxic score: 0\n\n
        
        ## Question 2:[If any but not at all] Please select all the reasons why below: [Checkmark]
        o It is threatening (ex. “If you comment on my feed again, I will personally beat you up”)\n
        o It is insulting (ex. “You are an absolutely disgusting hypocrite”)\n
        o It is profane (ex. “Why the f*** not you heathen?”)\n
        o It is an attack on a person or group based on their identity (ex. “Women are stupid and belong in the kitchen”)\n
        o It is sexual harassment (ex. “All I want from you are nudes”)\n
        o Other, please specify\n
        
        ## Question 2: Is this a type of comment you would want to see? [Radio]\n
        o I would never want to see it\n
        o It depends on the context\n
        o This is fine for me to see\n
        '''
    )
    return content



def main(
    model_name='gpt-3.5-turbo', # 'gpt-3.5-turbo',
    task_name = 'text_for_labeling',
    ):
    ############################################################################################################################################
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
    
    ############################################################################################################################################
    # ###classified_data_final_w_worker_hash.json dataset
    
    if task_name == "classified_data_final_w_worker_hash":
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
            input = f"{pre_prompt}\n## comment: {comment}\n{post_prompt}"
            inputs.append(input)

    ##################################################################################################


    if task_name == 'text_for_labeling':
        data_path = 'raw_data/text_for_labeling.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
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
        human_labels = []
        comments = raw_data
        for comment in comments:
            input = f"{pre_prompt}\n## comment: {comment}\n{post_prompt}"
            inputs.append(input)
            
    ##################################################################################################

    def fetch_content(input, idx):
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input}],
            stream=True,
        )
        GPT_content = ''
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                GPT_content += chunk.choices[0].delta.content
        return idx, GPT_content


    total_output_labels = []

    
    path = f"./{task_name}/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
            
            
    import concurrent.futures
    from tqdm import tqdm
    
    print("Start GPT labeling...")
    
    # import pdb;pdb.set_trace()
    # 使用ThreadPoolExecutor并行调用API
    
    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    # inputs = inputs[:100]
    # batch_size = 10 # the batch_size 
     
    # inputs = inputs[:10000]
    batch_size = 1000 # the batch_size
    split_size = len(inputs) // batch_size + (1 if len(inputs) % batch_size != 0 else 0)
    for batch_idx in tqdm(range(split_size), desc=f'{model_name} labeling'):
        batch_end =  min((batch_idx+1) * batch_size, len(inputs)) # the end index of batch
        batch_inputs = inputs[batch_idx* batch_size:batch_end] ##data range
        
        contents = [None] * len(batch_inputs)
        output_labels = [None] * len(batch_inputs) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_input = {executor.submit(fetch_content, input, idx): idx for idx, input in enumerate(batch_inputs)}
            for future in concurrent.futures.as_completed(future_to_input):
                idx = future_to_input[future]
                # try:
                #     idx, GPT_content = future.result()
                #     contents[idx] = GPT_content

                #     # match = re.search(r"\'toxic score\': (\d+)", GPT_content)
                #     # output_labels[idx] = int(match.group(1)) if match else -1
                #     matches = json_pattern.findall(GPT_content)
                #     if matches:
                #         try:
                #             # 解析并保存完整的 JSON 对象
                #             json_obj = json.loads(matches[0])
                #             output_labels[idx] = json_obj
                #         except json.JSONDecodeError:
                #             output_labels[idx] = None
                #     else:
                #         output_labels[idx] = None
                        
                # except Exception as exc:
                #     print(f'{inputs[idx]} generated an exception: {exc}')
                try:
                    retry_count = 3
                    while retry_count>0:
                        idx, GPT_content = future.result()
                        contents[idx] = GPT_content

                        matches = json_pattern.findall(GPT_content)
                        if matches:
                            try:
                                # 解析并保存完整的 JSON 对象
                                json_obj = json.loads(matches[0])
                                output_labels[idx] = json_obj
                                break  # 成功提取到 JSON，退出循环
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for input {inputs[idx]}")
                                output_labels[idx] = None
                        else:
                            print(f"No JSON match for input {inputs[idx]}, recalculating...")
                            future = executor.submit(fetch_content, inputs[idx], idx)  # 重新提交任务
                            
                        retry_count -= 1
                        
                except Exception as exc:
                    print(f'{inputs[idx]} generated an exception: {exc}')
                    output_labels[idx] = None
        # 输出结果
        # for content in contents:
        #     # print(content)
        #     print('#'*80)
        #     print(matches)
        # import pdb;pdb.set_trace()
        # for output_label in output_labels:
        #     print('#'*80)
        #     print(output_label)


        print(f'### batch_idx {batch_idx}\'s error output with None: {output_labels.count(None)}')
        
            
        # import pdb;pdb.set_trace()
        # torch.save(contents, path + f"output_contents_{idx}.pt")
        torch.save(output_labels, path + f"output_labels_{batch_idx}.pt")
        total_output_labels.extend(output_labels)
        
    # import pdb;pdb.set_trace()

    torch.save(human_labels, path + "human_labels.pt")
    torch.save(total_output_labels, path + f"total_output_labels.pt")

    # assert len(human_labels) == len(total_output_labels)

    print(f'total data size: {len(total_output_labels)}')
    print(f'unlabeled data size: {total_output_labels.count(-1)}')
    
    print("Finish GPT labeling!!!")
    ##################################################################################################

    '''
    single example labeling
    '''
    # contents = []
    # output_labels = []
    # for idx, input in enumerate(inputs[:10]):
        
    #     stream = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "user", "content": input}],
    #         stream=True,
    #         )
    #     GPT_content = ''
    #     for chunk in stream:
    #         # CHUNK form: ChatCompletionChunk(id='chatcmpl-9mpk2ZphIgvKsrBWcNkLzvLbMjE9a', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1721425666, model='gpt-4-0613', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None)
    #         if chunk.choices[0].delta.content is not None:

    #             print(chunk.choices[0].delta.content, end="")
    #             GPT_content += chunk.choices[0].delta.content


    #     print('\n' + '='*100 +'\n')
    #     print(f"Input idx {idx}: {GPT_content}")
    #     print('\n' + '='*100 +'\n')
    #     match = re.search(r"### Toxic score: (\d+)", GPT_content)
    #     label = int(match.group(1)) if match else -1
        
    #     import pdb;pdb.set_trace()

        
    #     contents.append(GPT_content)
    #     output_labels.append(label)

    # path = f".//toxicity/"
    # import pdb;pdb.set_trace()

    # if not os.path.exists(path):
    #     os.makedirs(path)
        
    # torch.save(contents, path + "output_contents.pt")
    # torch.save(output_labels, path + "output_labels.pt")
    # torch.save(human_labels, path + "human_labels.pt")


    # print("Finishing GPT labeling!!!")

# nohup python3 GPT_labeling_new_dataset.py > zzz_gpt_labeling.log &

if __name__ == '__main__':
    fire.Fire(main)
    
    
