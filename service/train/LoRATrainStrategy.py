from service.train.TrainStrategy import TrainStrategy
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from typing import Any, Dict, Optional
import torch
from tqdm import tqdm
import json
import os


class LoRATrainStrategy(TrainStrategy):
    """LoRA训练策略实现"""

    def __init__(self, model_path: str, tokenizer, model, lora_config: Dict[str, Any]):
        super().__init__(model_path, tokenizer, model)
        self.lora_config = lora_config
        self.max_length = 512

    def prepare_dataset(self, dataset_path: str, max_length: int) -> Dataset:
        """准备LoRA训练数据集"""
        print(f"加载数据集: {dataset_path}")
        self.max_length = max_length

        data = []
        error_count = 0

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_index, line in enumerate(tqdm(f, desc="加载数据")):
                try:
                    raw_data = json.loads(line)
                    formatted = self._format_conversion(raw_data)

                    if formatted and len(formatted["text"]) > 10:
                        data.append(formatted)
                    else:
                        print(f"跳过无效对话：第{line_index + 1}行")

                except Exception as e:
                    error_count += 1
                    print(f"数据转换错误：第{line_index + 1}行，错误信息：{str(e)}")
                    if error_count > 10:
                        raise RuntimeError("发现过多错误，请先修正数据格式")

        print(f"成功加载{len(data)}条有效数据(跳过{error_count}行数据)")
        return Dataset.from_list(data)

    def _format_conversion(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """格式转换"""
        messages = []
        for msg in example.get("conversations", []):
            role = msg.get("role", msg.get("assistant", "unknown")).lower()
            content = msg.get("content", "")

            if role not in ["user", "assistant"]:
                if len(messages) == 0:
                    role = "user"
                else:
                    role = "assistant" if messages[-1]["role"] == "user" else "user"

            if len(content.strip()) < 1:
                continue

            messages.append({"role": role, "content": content})

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Template error: {e}")
            return None

        labels = []
        for msg in messages:
            content_ids = self.tokenizer.encode(msg["content"], add_special_tokens=False)
            if msg["role"] == "assistant":
                labels.extend(content_ids + [self.tokenizer.eos_token_id])
            else:
                labels.extend([-100] * (len(content_ids) + 1))

        return {
            "text": text,
            "labels": labels[:self.max_length]
        }

    def _preprocess_function(self, examples):
        """数据预处理"""
        tokenized = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = torch.full(
            (len(examples["text"]), self.max_length),
            -100,
            dtype=torch.long
        )

        for i, lbl in enumerate(examples["labels"]):
            labels[i, :len(lbl)] = torch.LongTensor(lbl[:self.max_length])

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    def create_trainer(self, train_dataset: Dataset, config: Dict[str, Any]) -> Trainer:
        """创建LoRA训练器"""
        print("创建LoRA训练器...")

        # 配置LoRA
        peft_config = LoraConfig(
            r=self.lora_config.get('r', 8),
            lora_alpha=self.lora_config.get('lora_alpha', 32),
            target_modules=self.lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
            lora_dropout=self.lora_config.get('lora_dropout', 0.05),
            bias=self.lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]
        )

        # 应用LoRA
        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()

        # 预处理数据集
        processed_dataset = train_dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=32,
            remove_columns=["text", "labels"]
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )

        # 训练参数
        training_args = TrainingArguments(
            per_device_train_batch_size=config.get('per_device_train_batch_size', 7),
            learning_rate=config.get('learning_rate', 5e-5),
            num_train_epochs=config.get('num_train_epochs', 5),
            warmup_steps=config.get('warmup_steps', 10),
            report_to="none",
            output_dir=config.get('output_dir', './output'),
            save_safetensors=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator
        )

        print("LoRA训练器创建完成")
        return trainer

    def train(self, trainer: Trainer, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """执行训练"""
        print("开始LoRA训练...")

        if resume_from_checkpoint:
            print(f"从检查点恢复训练: {resume_from_checkpoint}")
            trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer_stats = trainer.train()

        print("LoRA训练完成")
        return trainer_stats

    def save_model(self, trainer: Trainer, output_dir: str):
        """保存模型"""
        print(f"保存LoRA模型到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, safe_serialization=True, save_embedding_checkpoint=True)
        print("LoRA模型保存完成")
