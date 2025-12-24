from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class EvalRequest(BaseModel):
    """评估请求模型"""
    model_path: str = Field(..., description="模型路径")
    dataset_path: str = Field(..., description="数据集路径")
    metrics: List[str] = Field(default=['accuracy', 'loss'], description="评估指标")
    batch_size: int = Field(default=8, description="批次大小")
    lora_adapter_path: str = Field(..., description="lora地址")

class EvalResult(BaseModel):
    """评估结果模型"""
    metrics: Dict[str, float] = Field(..., description="评估指标结果")
    total_samples: int = Field(..., description="总样本数")
    eval_time: float = Field(..., description="评估耗时(秒)")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="额外信息")
