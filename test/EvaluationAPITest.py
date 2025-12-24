import requests
import json
import os
import sys

# 设置UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class EvaluationAPITest:
    """评估服务API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "data")

    def test_evaluate_model(self, model_path: str):
        """测试模型评估API"""
        print("=" * 50)
        print("测试模型评估API")
        print("=" * 50)

        # 准备请求数据
        test_data_path = os.path.join(self.test_data_dir, "train_sample.json")
        request_data = {
            "model_path": model_path,
            "dataset_path": test_data_path,
            "metrics": ["loss", "perplexity"],
            "batch_size": 2
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送评估请求到: {self.base_url}/api/eval/evaluate")
            response = requests.post(
                f"{self.base_url}/api/eval/evaluate",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n评估结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("success"):
                print("\n✅ 模型评估测试成功")
            else:
                print("\n❌ 模型评估测试失败")

            return result

        except Exception as e:
            print(f"\n❌ 模型评估测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def run_all_tests(self, model_path: str):
        """运行所有评估服务API测试"""
        print("\n" + "=" * 60)
        print("开始运行评估服务API测试")
        print("=" * 60)

        # 测试1: 模型评估
        print("\n\n【测试1】模型评估API")
        eval_result = self.test_evaluate_model(model_path)

        print("\n" + "=" * 60)
        print("评估服务API测试完成")
        print("=" * 60)

        return {"evaluate": eval_result}


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

    tester = EvaluationAPITest()
    tester.run_all_tests(model_path)
