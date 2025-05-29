# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

# pip install hydra-core omegaconf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
import torch
import os
import numpy as np
import random
import wandb
import hydra
import torch.nn.functional as F
from transformers import TrainerCallback


os.environ["HF_HUB_OFFLINE"] = "1"


class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.report_to == "wandb":
            import wandb
            logs["train/kl_loss_r1"] = sum(trainer._metrics["train/kl_loss_r1"])/len(trainer._metrics["train/kl_loss_r1"])
            logs["train/kl_loss_qwen"] = sum(trainer._metrics["train/kl_loss_qwen"])/len(trainer._metrics["train/kl_loss_qwen"])
            wandb.log(logs, step=state.global_step)

class OThinkR1Trainer(SFTTrainer):
    def __init__(self, base_model1, base_model2, beta1,beta2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_qwen = base_model1 # qwen
        self.base_model_r1   = base_model2

        self.beta1 = beta1
        self.beta2 = beta2

    def kl_divergence(self, logits, base_logits):
        # Calculate the KL divergence
        p = F.softmax(logits, dim=-1)
        q = F.softmax(base_logits, dim=-1)
        kl_loss = F.kl_div(p.log(), q, reduction='batchmean')

        return kl_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Calculate the original loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Calculate the output of the base model
        with torch.no_grad():
            base_outputs_r1   = self.base_model_r1(**inputs)
            base_outputs_qwen = self.base_model_qwen(**inputs) 

        # Calculate the KL divergence
        kl_loss_r1          = self.beta1 * self.kl_divergence(outputs.logits, base_outputs_r1.logits)
        kl_loss_qwen        = self.beta2 * self.kl_divergence(outputs.logits, base_outputs_qwen.logits)
        # Combine the KL divergence to the original loss
        total_loss = loss +  kl_loss_r1 + kl_loss_qwen

        if self.accelerator.is_main_process:   
            self._store_metric("train/kl_loss_r1", kl_loss_r1.detach())
            self._store_metric("train/kl_loss_qwen", kl_loss_qwen.detach())
            self._store_metric("train/total_kl", (kl_loss_r1 + kl_loss_qwen).detach())

        # Record the token accuracy
        if "labels" in inputs and not self.use_liger:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Acuring the precision
            predictions = shift_logits.argmax(dim=-1)

            mask = shift_labels != -100

            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            accuracy = (correct_tokens.sum() / total_tokens.sum()).item() if total_tokens.sum() > 0 else 0.0
            self._metrics["mean_token_accuracy"].append(accuracy)

        return (total_loss, outputs) if return_outputs else total_loss
    def _store_metric(self, name, value):
        scalar_value = value.item() if isinstance(value, torch.Tensor) else value
        
        gathered_values = self.accelerator.gather_for_metrics(
            torch.tensor(scalar_value).to(self.accelerator.device)
        )
        avg_value = gathered_values.mean().item()

        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(avg_value)

        if self.args.report_to == "wandb":
            import wandb
            wandb.log({name: avg_value}, step=self.state.global_step)




@hydra.main(version_base=None, config_path="./config", config_name="SFT")
def main(config: DictConfig) -> None:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True)) # 强制解析
    print(f"config = \n{config}\ntype = {type(config)}")

    if config.logging.wandb.enable:
        wandb.login(key=config.logging.wandb.key)
        wandb.init(project=config.train.trainerConifg.wandb_project)

    model_id = config.train.trainerConifg.model_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    r1_base = AutoModelForCausalLM.from_pretrained(
        config.train.trainerConifg.r1_base_id,
        torch_dtype="auto",
        device_map="auto",
    )
    qwen_base = AutoModelForCausalLM.from_pretrained(
        config.train.trainerConifg.qwen_base_id,
        torch_dtype="auto",
        device_map="auto",
    )


    tokenizer = AutoTokenizer.from_pretrained(model_id)


    data_id = config.train.trainerConifg.data_id
    dataset = load_dataset(data_id, split="train")


    
    accumulate_steps = config.train.trainerConifg.accumulate_steps
    per_device_batch = config.train.trainerConifg.per_device_batch 
    lr               = config.train.trainerConifg.lr
    train_epochs     = config.train.trainerConifg.train_epochs
    model_save_steps = config.train.trainerConifg.model_save_steps
    
    output_dir = f"./save_models/{config.train.trainerConifg.save_model_prefix}/SFT_R1_lr_{lr}_accumulation_{accumulate_steps}_batch_{per_device_batch}"
    save_model_dir = os.path.dirname(output_dir)
    os.makedirs(save_model_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        gradient_accumulation_steps=accumulate_steps,
        per_device_train_batch_size=per_device_batch,
        learning_rate=lr,
        num_train_epochs=train_epochs,
        report_to="wandb",
        logging_steps=5,
        save_strategy="epoch",
        save_only_model=True,
        bf16=True,
        max_seq_length = 3000,
        seed = 0,
        data_seed = 0
    )




    SFT_trainer = OThinkR1Trainer(
        base_model1 = qwen_base,
        base_model2 = r1_base,
        beta1       = config.train.trainerConifg.beta1,
        beta2       = config.train.trainerConifg.beta2,
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[CustomWandbCallback]
    )


    SFT_trainer.train()


if __name__ == "__main__":
    main()