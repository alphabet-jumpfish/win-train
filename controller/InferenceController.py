from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from entity.request.InferenceModel import ChatRequest, ChatResponse, BatchInferenceRequest
from entity.response.ResponseModel import BaseResponse
from service.inference.InferenceService import InferenceService
from typing import Optional

router = APIRouter(prefix="/api/inference", tags=["模型推理"])

# 全局推理服务实例（需要在main.py中初始化）
inference_service: Optional[InferenceService] = None


def init_inference_service(model_path: str, lora_adapter_path: Optional[str] = None):
    """初始化推理服务"""
    global inference_service
    inference_service = InferenceService(model_path, lora_adapter_path)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天推理（非流式）"""
    global inference_service

    # 如果提供了model_path，动态初始化服务
    if request.model_path:
        inference_service = InferenceService(request.model_path)

    if not inference_service:
        raise HTTPException(status_code=503, detail="推理服务未初始化")

    try:
        messages = [msg.dict() for msg in request.messages]
        config = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'enable_thinking': request.enable_thinking
        }

        result = inference_service.generate(messages, config)

        return ChatResponse(
            content=result['content'],
            thinking_content=result.get('thinking_content'),
            finish_reason="stop"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """聊天推理（流式输出SSE）"""
    global inference_service

    # 如果提供了model_path，动态初始化服务
    if request.model_path:
        inference_service = InferenceService(request.model_path)

    if not inference_service:
        raise HTTPException(status_code=503, detail="推理服务未初始化")

    try:
        messages = [msg.dict() for msg in request.messages]
        config = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'enable_thinking': request.enable_thinking
        }

        return StreamingResponse(
            inference_service.generate_stream(messages, config),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BaseResponse)
async def batch_inference(request: BatchInferenceRequest):
    """批量推理"""
    global inference_service

    # 如果提供了model_path，动态初始化服务
    if request.model_path:
        inference_service = InferenceService(request.model_path)

    if not inference_service:
        raise HTTPException(status_code=503, detail="推理服务未初始化")

    try:
        config = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature
        }

        results = inference_service.batch_generate(request.prompts, config)

        return BaseResponse(
            success=True,
            message=f"批量推理完成，共 {len(results)} 条",
            data={"results": results}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
