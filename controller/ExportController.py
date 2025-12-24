from fastapi import APIRouter, HTTPException
from entity.request.ExportModel import ExportRequest, ExportResponse
from service.export.ExportService import ExportService

router = APIRouter(prefix="/api/export", tags=["模型导出"])
export_service = ExportService()


@router.post("/onnx", response_model=ExportResponse)
async def export_to_onnx(request: ExportRequest):
    """导出模型为ONNX格式"""
    try:
        result = export_service.export_to_onnx(
            model_path=request.model_path,
            output_path=request.output_path,
            opset_version=request.opset_version
        )

        return ExportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
