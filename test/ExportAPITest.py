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


class ExportAPITest:
    """导出服务API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.test_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_export_onnx(self, model_path: str):
        """测试导出ONNX模型API"""
        print("=" * 50)
        print("测试导出ONNX模型API")
        print("=" * 50)

        # 准备请求数据
        output_path = os.path.join(self.test_output_dir, "exported_model.onnx")
        request_data = {
            "model_path": model_path,
            "output_path": output_path,
            "export_format": "onnx",
            "opset_version": 14
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送导出请求到: {self.base_url}/api/export/onnx")
            response = requests.post(
                f"{self.base_url}/api/export/onnx",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n导出结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("success"):
                print("\n✅ ONNX导出测试成功")
            else:
                print("\n❌ ONNX导出测试失败")

            return result

        except Exception as e:
            print(f"\n❌ ONNX导出测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def run_all_tests(self, model_path: str):
        """运行所有导出服务API测试"""
        print("\n" + "=" * 60)
        print("开始运行导出服务API测试")
        print("=" * 60)

        # 测试1: 导出ONNX模型
        print("\n\n【测试1】导出ONNX模型API")
        export_result = self.test_export_onnx(model_path)

        print("\n" + "=" * 60)
        print("导出服务API测试完成")
        print("=" * 60)

        return {"export_onnx": export_result}


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

    tester = ExportAPITest()
    tester.run_all_tests(model_path)
