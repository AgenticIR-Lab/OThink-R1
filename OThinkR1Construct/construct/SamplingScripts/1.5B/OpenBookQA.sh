cd ../..

ConstructData=OpenBookQA
modelSize=1.5B
GPUNUM=4

python LRMall.py \
    model=DeepSeek-R1-Distill-Qwen-${modelSize}-Fix \
    model.inference.tensor_parallel_size=${GPUNUM} \
    model.inference.gpu_memory_utilization=0.9 \
    model.inference.temperature=0.9 \
    model.inference.top_p=0.95 \
    model.inference.max_tokens=16384 \
    data=${ConstructData} \
    data.datasets.${ConstructData}.splits.train.slice=\"[:100%]\" \

python LLMall.py \
    model=Qwen2.5-${modelSize}-Instruct \
    model.inference.tensor_parallel_size=${GPUNUM} \
    model.inference.gpu_memory_utilization=0.9 \
    model.inference.temperature=0.9 \
    model.inference.top_p=0.95 \
    model.inference.max_tokens=16384 \
    data=${ConstructData} \
    data.datasets.${ConstructData}.splits.train.slice=\"[:100%]\" \