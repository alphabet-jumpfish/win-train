from service.train.TrainStrategy import TrainStrategy
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
from typing import Any, Dict, Optional
import torch
import os


class TRLTrainStrategy(TrainStrategy):
    """TRL训练策略实现"""

    def prepare_dataset(self, dataset_path: str, max_length: int) -> Dataset:
        """准备TRL训练数据集"""
        print(f"加载数据集: {dataset_path}")

        # 加载原始数据
        raw_ds = load_dataset(
            path="json",
            data_files={"train": dataset_path},
            split="train"
        )

        # 转换为对话格式
        conversations = []
        for item in raw_ds:
            if 'conversations' in item:
                conversations.append(item['conversations'])

        # 使用tokenizer的chat_template格式化
        chat_inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False
        )

        # 创建Dataset
        train_dataset = Dataset.from_dict({"text": chat_inputs})
        print(f"数据集准备完成，共 {len(train_dataset)} 条数据")

        return train_dataset

    def create_trainer(self, train_dataset: Dataset, config: Dict[str, Any]) -> SFTTrainer:
        """创建TRL训练器"""
        print("创建TRL训练器...")

        # 创建训练配置
        train_args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            max_steps=config.get('max_steps', 1000),
            learning_rate=config.get('learning_rate', 2e-4),
            warmup_steps=config.get('warmup_steps', 10),
            logging_steps=config.get('logging_steps', 20),
            optim=config.get('optim', 'adamw_torch'),
            weight_decay=config.get('weight_decay', 0.01),
            lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
            seed=config.get('seed', 929),
            report_to="none",
            output_dir=config.get('output_dir', './output'),
            bf16=False,
            fp16=False,
            save_strategy="no",  # 禁用自动保存checkpoint
            save_steps=None
        )

        # 创建数据整理器
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # 创建训练器
        trainer = SFTTrainer(
            model=self.model,
            data_collator=collator,
            train_dataset=train_dataset,
            args=train_args
        )

        print("TRL训练器创建完成")
        return trainer

    def train(self, trainer: SFTTrainer, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """执行训练"""
        print("开始TRL训练...")

        if resume_from_checkpoint:
            print(f"从检查点恢复训练: {resume_from_checkpoint}")
            trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer_stats = trainer.train()

        print("TRL训练完成")
        return trainer_stats

    def save_model(self, trainer: SFTTrainer, output_dir: str):
        """保存模型"""
        print(f"保存模型到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{output_dir}/model.pth")
        print("模型保存完成")
