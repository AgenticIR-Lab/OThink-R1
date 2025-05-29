cd ../..


DATASETMODE=FUSE
mode=MATH-SPLIT
modelsize=7B
epochs=4
modelAccumulation=4
modelBatch=2
GPUNUM=4


for beta1 in 0.001; do
  for beta2 in 0.0001; do

    python training.py \
      train.trainerConifg.save_model_prefix="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.wandb_project="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.data_id="../OThinkR1Construct/OThinkR1Data/${DATASETMODE}/${modelsize}/${mode}" \
      train.trainerConifg.model_id="/Your/R1-base-model/Path" \
      train.trainerConifg.accumulate_steps=${modelAccumulation} \
      train.trainerConifg.per_device_batch=${modelBatch} \
      train.trainerConifg.lr=3e-5 \
      train.trainerConifg.train_epochs=${epochs} \
      +train.trainerConifg.r1_base_id="/Your/R1-base-model/Path" \
      +train.trainerConifg.qwen_base_id="/Your/Qwen-base-model/Path" \
      +train.trainerConifg.beta1=${beta1} \
      +train.trainerConifg.beta2=${beta2} \
    
    python training.py \
      train.trainerConifg.save_model_prefix="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.wandb_project="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.data_id="../OThinkR1Construct/OThinkR1Data/${DATASETMODE}/${modelsize}/${mode}" \
      train.trainerConifg.model_id="/Your/R1-base-model/Path" \
      train.trainerConifg.accumulate_steps=${modelAccumulation} \
      train.trainerConifg.per_device_batch=${modelBatch} \
      train.trainerConifg.lr=4e-5 \
      train.trainerConifg.train_epochs=${epochs} \
      +train.trainerConifg.r1_base_id="/Your/R1-base-model/Path" \
      +train.trainerConifg.qwen_base_id="/Your/Qwen-base-model/Path" \
      +train.trainerConifg.beta1=${beta1} \
      +train.trainerConifg.beta2=${beta2} \
    
    python training.py \
      train.trainerConifg.save_model_prefix="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.wandb_project="${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}" \
      train.trainerConifg.data_id="../OThinkR1Construct/OThinkR1Data/${DATASETMODE}/${modelsize}/${mode}" \
      train.trainerConifg.model_id="/Your/R1-base-model/Path" \
      train.trainerConifg.accumulate_steps=${modelAccumulation} \
      train.trainerConifg.per_device_batch=${modelBatch} \
      train.trainerConifg.lr=5e-5 \
      train.trainerConifg.train_epochs=${epochs} \
      +train.trainerConifg.r1_base_id="/Your/R1-base-model/Path" \
      +train.trainerConifg.qwen_base_id="/Your/Qwen-base-model/Path" \
      +train.trainerConifg.beta1=${beta1} \
      +train.trainerConifg.beta2=${beta2} \

    TestDataset=GSM8K
    for lr in {3..5}; do
        for model_path in ./save_models/${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}/SFT_R1_lr_${lr}e-05_accumulation_${modelAccumulation}_batch_${modelBatch}/*; do
            python eval.py \
                model=DeepSeek-R1-Distill-Qwen-${modelsize}-Fix \
                model.path=${model_path} \
                +model.mode="FUSE-${mode}-beta1-${beta1}-beta2-${beta2}" \
                data=${TestDataset} \
                data.datasets.${TestDataset}.splits.test.slice=\"[:100%]\" \
                model.inference.tensor_parallel_size=${GPUNUM} \
                model.inference.gpu_memory_utilization=0.9 \
                +model.inference.repetition_penalty=1.0 \
                model.inference.temperature=0.9 \
                model.inference.top_p=0.95 \
                model.inference.max_tokens=16384
        done
    done


    TestDataset=ASDIV
    for lr in {3..5}; do
        for model_path in ./save_models/${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}/SFT_R1_lr_${lr}e-05_accumulation_${modelAccumulation}_batch_${modelBatch}/*; do
            python eval.py \
                model=DeepSeek-R1-Distill-Qwen-${modelsize}-Fix \
                model.path=${model_path} \
                +model.mode="FUSE-${mode}-beta1-${beta1}-beta2-${beta2}" \
                data=${TestDataset} \
                data.datasets.${TestDataset}.splits.test.slice=\"[:100%]\" \
                model.inference.tensor_parallel_size=${GPUNUM} \
                model.inference.gpu_memory_utilization=0.9 \
                model.inference.temperature=0.9 \
                +model.inference.repetition_penalty=1.0 \
                model.inference.top_p=0.95 \
                model.inference.max_tokens=16384
        done
    done

    TestDataset=MATH
    for lr in {3..5}; do
        # 获取所有模型路径
        for model_path in ./save_models/${DATASETMODE}-${modelsize}-${mode}-ACCU-${modelAccumulation}-BATCH-${modelBatch}-beta1-${beta1}-beta2-${beta2}/SFT_R1_lr_${lr}e-05_accumulation_${modelAccumulation}_batch_${modelBatch}/*; do
            python validation.py \
                model=DeepSeek-R1-Distill-Qwen-${modelsize}-Fix \
                model.path=${model_path} \
                +model.mode="FUSE-${mode}-beta1-${beta1}-beta2-${beta2}" \
                data=${TestDataset} \
                model.inference.tensor_parallel_size=${GPUNUM} \
                model.inference.gpu_memory_utilization=0.9 \
                +model.inference.repetition_penalty=1.0 \
                model.inference.temperature=0.9 \
                model.inference.top_p=0.95 \
                model.inference.max_tokens=16384
        done
    done

  
  done
done



