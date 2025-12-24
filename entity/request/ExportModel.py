from pydantic import BaseModel, Field
from typing import Literal, Optional


class ExportRequest(BaseModel):
    """模型导出请求"""
    model_path: str = Field(..., description="模型路径")
    export_format: Literal['onnx', 'torchscript', 'safetensors'] = Field(default='onnx', description="导出格式")
    output_path: str = Field(..., description="输出路径")
    opset_version: int = Field(default=14, description="ONNX opset版本")


class ExportResponse(BaseModel):
    """模型导出响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    export_path: Optional[str] = Field(default=None, description="导出文件路径")
    file_size: Optional[int] = Field(default=None, description="文件大小(字节)")
