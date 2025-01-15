export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=4

##### GPT API labeling #####
# model_name="gpt-3.5-turbo" ###gpt-4o-mini gpt-3.5-turbo
# task_name='multipref' #multipref classified_data_final_w_worker_hash SemEval2016
# python GPT_labeling_new_dataset.py --model_name $model_name --task_name $task_name


######### Huggingface model labeling ####
# dataset_name='multipref' #text_for_labeling
# model_name_list=("mistral" "phi" "gemma-9b" "gemma-2b")

model_name_list=("mistral" "phi" "gemma-2b")
dataset_name_list=("multipref" "text_for_labeling")

idx_list=("1" "2")

for idx in ${idx_list[@]}; do

    root_path="./more_results_${idx}"

    for dataset_name in ${dataset_name_list[@]}; do

        for model_name in ${model_name_list[@]}; do
            echo "*** Current idx: ${idx} ***"
            echo "*** Current task name: ${dataset_name} ***"
            echo "*** Current model_name: ${model_name} ***"

            accelerate launch \
                --main_process_port 29506 \
                --num_processes $num_gpus \
                labeling_new_dataset_accelerate.py \
                --model_name $model_name \
                --dataset_name $dataset_name \
                --root_path $root_path
        done
    done

done


# --mixed_precision bf16 \
# --config_file fsdp_configs/fsdp_config.yaml \   CUDA error: device-side assert triggered


# bash labeling.sh > zzz_multipref_labeling.log 2>&1
# bash labeling.sh > zzz_multipref_labeling_gemma-9b.log 2>&1
# bash labeling.sh > zzz_labeling_all.log 2>&1
