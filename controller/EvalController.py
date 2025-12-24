from fastapi import APIRouter, HTTPException
from entity.request.EvalModel import EvalRequest, EvalResult
from entity.response.ResponseModel import BaseResponse
from service.eval.EvalService import EvalService

router = APIRouter(prefix="/api/eval", tags=["模型评估"])
eval_service = EvalService()


@router.post("/evaluate", response_model=BaseResponse)
async def evaluate_model(request: EvalRequest):
    """评估模型性能"""
    try:
        result = eval_service.evaluate_model(
            model_path=request.model_path,
            dataset_path=request.dataset_path,
            metrics=request.metrics,
            batch_size=request.batch_size
        )

        eval_result = EvalResult(
            metrics=result['metrics'],
            total_samples=result['total_samples'],
            eval_time=result['eval_time']
        )

        return BaseResponse(
            success=True,
            message="模型评估完成",
            data=eval_result.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
