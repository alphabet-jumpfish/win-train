from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    model_config = ConfigDict(protected_namespaces=())

    model_path: Optional[str] = Field(default=None, description="模型路径（可选，用于动态加载）")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: int = Field(default=512, description="最大生成token数")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.8, description="top_p采样")
    top_k: int = Field(default=20, description="top_k采样")
    stream: bool = Field(default=False, description="是否流式输出")
    enable_thinking: bool = Field(default=False, description="是否启用thinking模式")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    content: str = Field(..., description="生成的内容")
    thinking_content: Optional[str] = Field(default=None, description="思考过程内容")
    finish_reason: str = Field(default="stop", description="结束原因")


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    model_config = ConfigDict(protected_namespaces=())

    model_path: Optional[str] = Field(default=None, description="模型路径（可选，用于动态加载）")
    prompts: List[str] = Field(..., description="提示词列表")
    max_tokens: int = Field(default=512, description="最大生成token数")
    temperature: float = Field(default=0.7, description="温度参数")
