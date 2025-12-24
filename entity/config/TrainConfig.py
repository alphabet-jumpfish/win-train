from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class TrainConfig(BaseModel):
    """训练配置模型"""
    strategy: Literal['trl', 'lora'] = Field(default='trl', description="训练策略: trl 或 lora")
    per_device_train_batch_size: int = Field(default=2, description="每个设备的batch size")
    gradient_accumulation_steps: int = Field(default=4, description="梯度累积步数")
    learning_rate: float = Field(default=2e-4, description="学习率")
    max_steps: int = Field(default=1000, description="训练步数")
    num_train_epochs: int = Field(default=5, description="训练轮数")
    warmup_steps: int = Field(default=10, description="预热步数")
    logging_steps: int = Field(default=20, description="日志记录步数")
    optim: str = Field(default='adamw_torch', description="优化器")
    weight_decay: float = Field(default=0.01, description="权重衰减")
    lr_scheduler_type: str = Field(default='linear', description="学习率调度器")
    seed: int = Field(default=929, description="随机种子")
    max_length: int = Field(default=512, description="最大序列长度")
    output_dir: Optional[str] = Field(default=None, description="输出目录")


class LoraConfig(BaseModel):
    """LoRA配置模型"""
    r: int = Field(default=8, description="LoRA秩")
    lora_alpha: int = Field(default=32, description="LoRA缩放因子")
    target_modules: List[str] = Field(default=['q_proj', 'k_proj', 'v_proj', 'o_proj'], description="目标模块")
    lora_dropout: float = Field(default=0.05, description="Dropout概率")
    bias: str = Field(default='none', description="偏置")


class InferenceConfig(BaseModel):
    """推理配置模型"""
    max_new_tokens: int = Field(default=512, description="最大生成token数")
    temperature: float = Field(default=0.7, description="温度")
    top_p: float = Field(default=0.8, description="top_p采样")
    top_k: int = Field(default=20, description="top_k采样")
    enable_thinking: bool = Field(default=False, description="是否启用thinking模式")
    stream: bool = Field(default=False, description="是否流式输出")
