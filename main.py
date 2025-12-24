from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
from util.WinConstant import Constant
from controller import DataController, TrainController, InferenceController, EvalController, ExportController

# 创建FastAPI应用
app = FastAPI(
    title="模型训练管理系统",
    description="基于FastAPI的模型训练、推理、评估和导出管理系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(DataController.router)
app.include_router(TrainController.router)
app.include_router(InferenceController.router)
app.include_router(EvalController.router)
app.include_router(ExportController.router)


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    print("=" * 50)
    print("模型训练管理系统启动中...")
    print("=" * 50)

    # 加载配置
    try:
        with open(Constant.CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"配置文件加载成功: {Constant.CONFIG_PATH}")

        # 初始化推理服务（可选）
        if config.get('inference', {}).get('auto_load', False):
            model_path = config['model']['base_model_path']
            lora_path = config['model'].get('lora_output_path')
            InferenceController.init_inference_service(model_path, lora_path)
            print(f"推理服务已初始化: {model_path}")

    except Exception as e:
        print(f"启动初始化失败: {e}")

    print("系统启动完成！")
    print("=" * 50)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "模型训练管理系统",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # 加载配置
    with open(Constant.CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    host = config.get('api', {}).get('host', '127.0.0.1')
    port = config.get('api', {}).get('port', 8801)

    print(f"启动服务: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
