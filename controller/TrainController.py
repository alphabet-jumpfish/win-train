from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from entity.response.ResponseModel import BaseResponse, TrainTaskResponse, TrainProgressResponse
from entity.config.TrainConfig import TrainConfig
from service.train.TrainService import TrainService
from typing import Dict, Any
from pydantic import BaseModel


class TrainRequest(BaseModel):
    """训练请求模型"""
    strategy: str
    dataset_path: str
    output_dir: str
    config: Dict[str, Any]


router = APIRouter(prefix="/api/train", tags=["模型训练"])
train_service = TrainService()


@router.post("/start", response_model=TrainTaskResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    request: TrainRequest
):
    """启动训练任务（异步）"""
    try:
        # 创建训练任务
        task_id = train_service.create_task(
            request.strategy,
            request.dataset_path,
            request.output_dir,
            request.config
        )

        # 在后台启动训练
        background_tasks.add_task(train_service.start_train_async, task_id)

        return TrainTaskResponse(
            task_id=task_id,
            status="pending",
            message="训练任务已创建，正在后台执行"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=TrainProgressResponse)
async def get_task_status(task_id: str):
    """查询训练任务状态"""
    try:
        task = train_service.get_task_status(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        return TrainProgressResponse(
            task_id=task.task_id,
            status=task.status.value,
            progress=task.progress,
            current_step=task.current_step,
            total_steps=task.total_steps,
            metrics=task.metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=BaseResponse)
async def get_all_tasks():
    """获取所有训练任务"""
    try:
        tasks = train_service.get_all_tasks()
        tasks_list = [task.dict() for task in tasks.values()]

        return BaseResponse(
            success=True,
            message=f"共 {len(tasks_list)} 个任务",
            data=tasks_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
