

cd ../..

dataname=GSM8K
modelsize=7B

python give_class_part1.py \
    --data_path ../OThinkR1Parts/${dataname}/${modelsize}/part1 \
    --output_file ../OThinkR1LLMJudge/${dataname}/${modelsize}/class.log

