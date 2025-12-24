import requests
import json
import os
import time
from typing import Dict, Any
import sys

# 设置UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



class TrainAPITest:
    """训练API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.test_output_dir = os.path.join(os.path.dirname(__file__), "output")

        # 确保输出目录存在
        os.makedirs(self.test_output_dir, exist_ok=True)

        # 测试数据路径
        self.train_data_path = os.path.join(self.test_data_dir, "train_sample.json")

    def test_trl_training(self, model_path: str) -> Dict[str, Any]:
        """测试TRL训练策略"""
        print("=" * 50)
        print("开始测试 TRL 训练策略")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            "strategy": "trl",
            "dataset_path": self.train_data_path,
            "output_dir": os.path.join(self.test_output_dir, "trl_output"),
            "config": {
                "model_path": model_path,
                "per_device_train_batch_size": 1,
                "learning_rate": 2e-4,
                "max_steps": 10,
                "warmup_steps": 2,
                "logging_steps": 5
            }
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        # 发送训练请求
        try:
            print(f"\n发送训练请求到: {self.base_url}/api/train/start")
            response = requests.post(
                f"{self.base_url}/api/train/start",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n训练任务已创建:")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")

            return result

        except Exception as e:
            print(f"\n❌ TRL训练测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_lora_training(self, model_path: str) -> Dict[str, Any]:
        """测试LoRA训练策略"""
        print("=" * 50)
        print("开始测试 LoRA 训练策略")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            "strategy": "lora",
            "dataset_path": self.train_data_path,
            "output_dir": os.path.join(self.test_output_dir, "lora_output"),
            "config": {
                "model_path": model_path,
                "per_device_train_batch_size": 1,
                "learning_rate": 5e-5,
                "num_train_epochs": 1,
                "warmup_steps": 2,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "lora_dropout": 0.05
                }
            }
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        # 发送训练请求
        try:
            print(f"\n发送训练请求到: {self.base_url}/api/train/start")
            response = requests.post(
                f"{self.base_url}/api/train/start",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n训练任务已创建:")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")

            return result

        except Exception as e:
            print(f"\n❌ LoRA训练测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询训练任务状态"""
        try:
            response = requests.get(f"{self.base_url}/api/train/status/{task_id}")
            response.raise_for_status()
            result = response.json()

            print(f"\n任务状态:")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Progress: {result.get('progress')}%")
            print(f"Current Step: {result.get('current_step')}/{result.get('total_steps')}")

            return result

        except Exception as e:
            print(f"\n❌ 查询任务状态失败: {e}")
            return {"error": str(e)}

    def run_all_tests(self, model_path: str):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("开始运行训练API测试")
        print("=" * 60)

        # 测试TRL训练
        print("\n\n【测试1】TRL训练策略")
        trl_result = self.test_trl_training(model_path)

        if "task_id" in trl_result:
            print("\n✅ TRL训练任务创建成功")
            time.sleep(2)
            self.check_task_status(trl_result["task_id"])
        else:
            print("\n❌ TRL训练任务创建失败")

        print("\n" + "-" * 60)

        # 测试LoRA训练
        print("\n\n【测试2】LoRA训练策略")
        lora_result = self.test_lora_training(model_path)
        
        if "task_id" in lora_result:
            print("\n✅ LoRA训练任务创建成功")
            time.sleep(2)
            self.check_task_status(lora_result["task_id"])
        else:
            print("\n❌ LoRA训练任务创建失败")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


if __name__ == "__main__":
    # 从配置文件读取模型路径
    import yaml
    import sys
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            model_path = config['model']['base_model_path']
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        print("请确保 config.yaml 文件存在且配置正确")
        sys.exit(1)

    # 创建测试实例并运行
    tester = TrainAPITest()
    tester.run_all_tests(model_path)
