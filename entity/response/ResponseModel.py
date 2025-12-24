from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(default=None, description="响应数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")


class TrainTaskResponse(BaseModel):
    """训练任务响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态: pending, running, completed, failed")
    message: str = Field(..., description="状态消息")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")


class TrainProgressResponse(BaseModel):
    """训练进度响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: float = Field(..., description="进度百分比 0-100")
    current_step: int = Field(default=0, description="当前步数")
    total_steps: int = Field(default=0, description="总步数")
    loss: Optional[float] = Field(default=None, description="当前损失")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="其他指标")
