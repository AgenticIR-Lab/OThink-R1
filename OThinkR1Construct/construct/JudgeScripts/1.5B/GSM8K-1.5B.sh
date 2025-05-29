

cd ../..

dataname=GSM8K
modelsize=1.5B

python give_class_part1.py \
    --data_path ../OThinkR1Parts/${dataname}/${modelsize}/part1 \
    --output_file ../OThinkR1LLMJudge/${dataname}/${modelsize}/class.log

