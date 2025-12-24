from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class Message(BaseModel):
    """对话消息模型"""
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class Conversation(BaseModel):
    """对话模型"""
    conversations: List[Message] = Field(..., description="对话列表")


class DatasetSplitConfig(BaseModel):
    """数据集划分配置"""
    train_ratio: float = Field(default=0.8, description="训练集比例")
    val_ratio: float = Field(default=0.1, description="验证集比例")
    test_ratio: float = Field(default=0.1, description="测试集比例")
    shuffle: bool = Field(default=True, description="是否打乱数据")
    seed: int = Field(default=42, description="随机种子")


class DataProcessRequest(BaseModel):
    """数据处理请求模型"""
    input_file: str = Field(..., description="输入文件路径")
    output_dir: str = Field(..., description="输出目录路径")
    format_type: str = Field(default='conversations', description="数据格式类型")
    split_config: Optional[DatasetSplitConfig] = Field(default=None, description="数据集划分配置")


class DataValidationResult(BaseModel):
    """数据验证结果"""
    is_valid: bool = Field(..., description="是否有效")
    total_samples: int = Field(..., description="总样本数")
    valid_samples: int = Field(..., description="有效样本数")
    invalid_samples: int = Field(..., description="无效样本数")
    error_messages: List[str] = Field(default=[], description="错误信息列表")
