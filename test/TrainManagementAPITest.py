import requests
import json
import os
import time
import sys

# 设置UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class TrainManagementAPITest:
    """训练管理API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.test_output_dir = os.path.join(os.path.dirname(__file__), "output")

        # 确保输出目录存在
        os.makedirs(self.test_output_dir, exist_ok=True)

        # 测试数据路径
        self.train_data_path = os.path.join(self.test_data_dir, "train_sample.json")

    def test_start_trl_training(self, model_path: str):
        """测试启动TRL训练任务API"""
        print("=" * 50)
        print("测试启动TRL训练任务API")
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

            if result.get('task_id'):
                print("\n✅ TRL训练任务创建成功")
            else:
                print("\n❌ TRL训练任务创建失败")

            return result

        except Exception as e:
            print(f"\n❌ TRL训练任务创建失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_start_lora_training(self, model_path: str):
        """测试启动LoRA训练任务API"""
        print("\n" + "=" * 50)
        print("测试启动LoRA训练任务API")
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

            if result.get('task_id'):
                print("\n✅ LoRA训练任务创建成功")
            else:
                print("\n❌ LoRA训练任务创建失败")

            return result

        except Exception as e:
            print(f"\n❌ LoRA训练任务创建失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_get_task_status(self, task_id: str):
        """测试查询训练任务状态API"""
        print("\n" + "=" * 50)
        print("测试查询训练任务状态API")
        print("=" * 50)

        try:
            print(f"\n发送状态查询请求到: {self.base_url}/api/train/status/{task_id}")
            response = requests.get(f"{self.base_url}/api/train/status/{task_id}")
            response.raise_for_status()
            result = response.json()

            print(f"\n任务状态:")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Status: {result.get('status')}")
            print(f"Progress: {result.get('progress')}%")
            print(f"Current Step: {result.get('current_step')}/{result.get('total_steps')}")

            if result.get('metrics'):
                print(f"Metrics: {json.dumps(result.get('metrics'), indent=2, ensure_ascii=False)}")

            print("\n✅ 任务状态查询成功")
            return result

        except Exception as e:
            print(f"\n❌ 任务状态查询失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_get_all_tasks(self):
        """测试获取所有训练任务API"""
        print("\n" + "=" * 50)
        print("测试获取所有训练任务API")
        print("=" * 50)

        try:
            print(f"\n发送请求到: {self.base_url}/api/train/tasks")
            response = requests.get(f"{self.base_url}/api/train/tasks")
            response.raise_for_status()
            result = response.json()

            print(f"\n所有任务:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get('success'):
                print("\n✅ 获取所有任务成功")
            else:
                print("\n❌ 获取所有任务失败")

            return result

        except Exception as e:
            print(f"\n❌ 获取所有任务失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def run_all_tests(self, model_path: str):
        """运行所有训练管理API测试"""
        print("\n" + "=" * 60)
        print("开始运行训练管理API测试")
        print("=" * 60)

        # 测试1: 启动TRL训练
        print("\n\n【测试1】启动TRL训练任务API")
        trl_result = self.test_start_trl_training(model_path)
        trl_task_id = trl_result.get('task_id')

        print("\n" + "-" * 60)

        # 测试2: 启动LoRA训练
        print("\n\n【测试2】启动LoRA训练任务API")
        lora_result = self.test_start_lora_training(model_path)

        print("\n" + "-" * 60)

        # 等待一段时间让训练开始
        time.sleep(3)

        # 测试3: 查询任务状态
        print("\n\n【测试3】查询训练任务状态API")
        if trl_task_id:
            self.test_get_task_status(trl_task_id)

        print("\n" + "-" * 60)

        # 测试4: 获取所有任务
        print("\n\n【测试4】获取所有训练任务API")
        all_tasks_result = self.test_get_all_tasks()

        print("\n" + "=" * 60)
        print("训练管理API测试完成")
        print("=" * 60)

        return {
            "trl_training": trl_result,
            "lora_training": lora_result,
            "all_tasks": all_tasks_result
        }


if __name__ == "__main__":
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            model_path = config['model']['base_model_path']
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        sys.exit(1)

    tester = TrainManagementAPITest()
    tester.run_all_tests(model_path)
