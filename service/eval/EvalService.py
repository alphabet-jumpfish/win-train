from typing import Dict, Any, List, Optional

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import time
from evalscope.run import run_task
from evalscope.summarizer import Summarizer


class EvalService:
    """评估服务 - 集成evalscope评测"""

    def __init__(self):
        pass

    def evaluate_model(self, model_path: str, dataset_path: str, metrics: List[str], batch_size: int = 8,
                       lora_adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型性能
        Args:
            model_path: 模型路径
            dataset_path: 数据集路径
            metrics: 评估指标列表
            batch_size: 批次大小
        Returns:
            评估结果
        """
        print(f"开始评估模型: {model_path}")
        start_time = time.time()

        try:
            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.bfloat16
            )
            # 如果提供了LoRA适配器路径，加载适配器
            if lora_adapter_path:
                print(f"加载LoRA适配器: {lora_adapter_path}")
                model = PeftModel.from_pretrained(model, lora_adapter_path)

            # 加载数据集
            dataset = load_dataset("json", data_files=dataset_path, split="train")
            total_samples = len(dataset)

            print(f"数据集加载完成，共 {total_samples} 条数据")

            # 计算基础指标
            results = self._calculate_metrics(model, tokenizer, dataset, metrics, batch_size)

            eval_time = time.time() - start_time
            print(f"评估完成，耗时: {eval_time:.2f}秒")

            return {
                "metrics": results,
                "total_samples": total_samples,
                "eval_time": eval_time
            }

        except Exception as e:
            print(f"评估失败: {e}")
            raise

    def _calculate_metrics(self, model, tokenizer, dataset, metrics: List[str], batch_size: int) -> Dict[str, float]:
        """计算评估指标"""
        results = {}

        total_loss = 0.0
        total_samples = 0

        model.eval()
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                # 处理批次中的每个样本
                for j in range(len(batch['conversations']) if 'conversations' in batch else 0):
                    messages = batch['conversations'][j]
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)

                    outputs = model(**inputs, labels=inputs.input_ids)
                    total_loss += outputs.loss.item()
                    total_samples += 1

        if 'loss' in metrics:
            results['loss'] = total_loss / total_samples if total_samples > 0 else 0.0

        if 'perplexity' in metrics:
            results['perplexity'] = torch.exp(torch.tensor(results['loss'])).item()

        return results
