from fastapi import APIRouter, HTTPException
from entity.system_model.DatasetModel import DataProcessRequest, DataValidationResult
from entity.response.ResponseModel import BaseResponse
from service.data.DataService import DataService

router = APIRouter(prefix="/api/data", tags=["数据处理"])
data_service = DataService()


@router.post("/validate", response_model=BaseResponse)
async def validate_data(file_path: str):
    """验证数据格式"""
    try:
        data = data_service.load_raw_data(file_path)
        result = data_service.validate_conversations(data)

        return BaseResponse(
            success=result.is_valid,
            message="数据验证完成",
            data=result.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=BaseResponse)
async def process_data(request: DataProcessRequest):
    """处理数据集"""
    try:
        # 加载原始数据
        data = data_service.load_raw_data(request.input_file)

        # 标准化数据
        normalized_data = data_service.normalize_conversations(data)

        # 如果需要划分数据集
        if request.split_config:
            split_data = data_service.split_dataset(normalized_data, request.split_config)

            # 保存划分后的数据集
            data_service.save_dataset(split_data['train'], f"{request.output_dir}/train.json")
            data_service.save_dataset(split_data['val'], f"{request.output_dir}/val.json")
            data_service.save_dataset(split_data['test'], f"{request.output_dir}/test.json")

            message = f"数据处理完成，训练集: {len(split_data['train'])}, 验证集: {len(split_data['val'])}, 测试集: {len(split_data['test'])}"
        else:
            # 保存全部数据
            data_service.save_dataset(normalized_data, f"{request.output_dir}/processed.json")
            message = f"数据处理完成，共 {len(normalized_data)} 条数据"

        return BaseResponse(
            success=True,
            message=message,
            data={"output_dir": request.output_dir}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
