from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainTask(BaseModel):
    """训练任务模型"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    strategy: str = Field(..., description="训练策略")
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: str = Field(..., description="输出目录")
    config: Dict[str, Any] = Field(..., description="训练配置")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
    started_at: Optional[str] = Field(default=None, description="开始时间")
    completed_at: Optional[str] = Field(default=None, description="完成时间")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    progress: float = Field(default=0.0, description="进度 0-100")
    current_step: int = Field(default=0, description="当前步数")
    total_steps: int = Field(default=0, description="总步数")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="训练指标")
