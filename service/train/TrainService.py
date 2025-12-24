from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from service.train.TrainStrategy import TrainStrategy
from service.train.TRLTrainStrategy import TRLTrainStrategy
from service.train.LoRATrainStrategy import LoRATrainStrategy
from entity.task.TaskModel import TrainTask, TaskStatus
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor


class TrainService:
    """训练服务 - 使用策略模式管理不同的训练方式"""

    def __init__(self):
        self.tasks: Dict[str, TrainTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_task(self, strategy: str, dataset_path: str, output_dir: str, config: Dict[str, Any]) -> str:
        """创建训练任务"""
        task_id = str(uuid.uuid4())

        task = TrainTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            strategy=strategy,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config
        )

        self.tasks[task_id] = task
        print(f"创建训练任务: {task_id}, 策略: {strategy}")
        return task_id

    def _get_strategy(self, strategy_name: str, model_path: str, config: Dict[str, Any]) -> TrainStrategy:
        """根据策略名称获取训练策略实例"""
        print(f"加载模型: {model_path}")

        # 自动选择合适的数据类型
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
            use_cache=False
        )

        tokenizer.pad_token = tokenizer.eos_token

        if strategy_name.lower() == 'trl':
            return TRLTrainStrategy(model_path, tokenizer, model)
        elif strategy_name.lower() == 'lora':
            lora_config = config.get('lora_config', {})
            return LoRATrainStrategy(model_path, tokenizer, model, lora_config)
        else:
            raise ValueError(f"不支持的训练策略: {strategy_name}")

    def _execute_train(self, task_id: str):
        """执行训练任务"""
        task = self.tasks[task_id]

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            print(f"开始执行训练任务: {task_id}")

            # 获取训练策略
            print(f"正在加载模型: {task.config.get('model_path')}")
            strategy = self._get_strategy(
                task.strategy,
                task.config.get('model_path'),
                task.config
            )
            print(f"模型加载完成")

            # 准备数据集
            print(f"正在准备数据集: {task.dataset_path}")
            dataset = strategy.prepare_dataset(
                task.dataset_path,
                task.config.get('max_length', 512)
            )
            print(f"数据集准备完成")

            # 创建训练器
            print(f"正在创建训练器")
            trainer = strategy.create_trainer(dataset, task.config)
            print(f"训练器创建完成")

            # 执行训练
            print(f"开始训练")
            resume_checkpoint = task.config.get('resume_from_checkpoint')
            train_stats = strategy.train(trainer, resume_checkpoint)
            print(f"训练完成")

            # 保存模型
            print(f"正在保存模型到: {task.output_dir}")
            strategy.save_model(trainer, task.output_dir)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            # 将TrainOutput对象转换为字典
            if hasattr(train_stats, 'metrics'):
                task.metrics = train_stats.metrics
            elif isinstance(train_stats, dict):
                task.metrics = train_stats
            else:
                task.metrics = {}
            print(f"训练任务完成: {task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            print(f"训练任务失败: {task_id}, 错误: {e}")
            import traceback
            traceback.print_exc()
            print(f"训练任务失败: {task_id}, 错误: {e}")

    async def start_train_async(self, task_id: str):
        """异步启动训练任务"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._execute_train, task_id)

    def get_task_status(self, task_id: str) -> Optional[TrainTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TrainTask]:
        """获取所有任务"""
        return self.tasks

