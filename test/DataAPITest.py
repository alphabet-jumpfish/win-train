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


class DataAPITest:
    """数据处理API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.test_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

        # 确保输出目录存在
        os.makedirs(self.test_output_dir, exist_ok=True)

        # 测试数据路径
        self.train_data_path = os.path.join(self.test_data_dir, "train_sample.json")

    def test_validate_data(self):
        """测试数据验证API"""
        print("=" * 50)
        print("测试数据验证API")
        print("=" * 50)

        try:
            print(f"\n发送验证请求到: {self.base_url}/api/data/validate")
            print(f"数据文件: {self.train_data_path}")

            response = requests.post(
                f"{self.base_url}/api/data/validate",
                params={"file_path": self.train_data_path},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n验证结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("success"):
                print("\n✅ 数据验证测试成功")
            else:
                print("\n❌ 数据验证测试失败")

            return result

        except Exception as e:
            print(f"\n❌ 数据验证测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_process_data(self):
        """测试数据处理API"""
        print("\n" + "=" * 50)
        print("测试数据处理API")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            "input_file": self.train_data_path,
            "output_dir": os.path.join(self.test_output_dir, "processed_data"),
            "split_config": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "shuffle": True
            }
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送处理请求到: {self.base_url}/api/data/process")
            response = requests.post(
                f"{self.base_url}/api/data/process",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n处理结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("success"):
                print("\n✅ 数据处理测试成功")
            else:
                print("\n❌ 数据处理测试失败")

            return result

        except Exception as e:
            print(f"\n❌ 数据处理测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def run_all_tests(self):
        """运行所有数据处理API测试"""
        print("\n" + "=" * 60)
        print("开始运行数据处理API测试")
        print("=" * 60)

        # 测试1: 数据验证
        print("\n\n【测试1】数据验证API")
        validate_result = self.test_validate_data()

        print("\n" + "-" * 60)

        # 测试2: 数据处理
        print("\n\n【测试2】数据处理API")
        process_result = self.test_process_data()

        print("\n" + "=" * 60)
        print("数据处理API测试完成")
        print("=" * 60)

        return {
            "validate": validate_result,
            "process": process_result
        }


if __name__ == "__main__":
    tester = DataAPITest()
    tester.run_all_tests()
