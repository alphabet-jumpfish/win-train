from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class TrainStrategy(ABC):
    """训练策略抽象基类"""

    def __init__(self, model_path: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model

    @abstractmethod
    def prepare_dataset(self, dataset_path: str, max_length: int) -> Any:
        """准备数据集"""
        pass

    @abstractmethod
    def create_trainer(self, train_dataset: Any, config: Dict[str, Any]) -> Any:
        """创建训练器"""
        pass

    @abstractmethod
    def train(self, trainer: Any, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """执行训练"""
        pass

    @abstractmethod
    def save_model(self, trainer: Any, output_dir: str):
        """保存模型"""
        pass
