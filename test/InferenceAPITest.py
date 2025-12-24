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


class InferenceAPITest:
    """推理服务API测试类"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url

    def test_chat_inference(self, model_path: str):
        """测试普通推理API"""
        print("=" * 50)
        print("测试普通推理API")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            # "model_path": model_path,
            "messages": [
                {"role": "user", "content": "你好"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送推理请求到: {self.base_url}/api/inference/chat")
            response = requests.post(
                f"{self.base_url}/api/inference/chat",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n推理结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("content"):
                print("\n✅ 普通推理测试成功")
            else:
                print("\n❌ 普通推理测试失败")

            return result

        except Exception as e:
            print(f"\n❌ 普通推理测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def test_stream_inference(self, model_path: str):
        """测试流式推理API"""
        print("\n" + "=" * 50)
        print("测试流式推理API")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            "model_path": model_path,
            "messages": [
                {"role": "user", "content": "介绍一下Python编程语言"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送流式推理请求到: {self.base_url}/api/inference/chat/stream")
            response = requests.post(
                f"{self.base_url}/api/inference/chat/stream",
                json=request_data,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            print(f"\n流式推理结果:")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str != '[DONE]':
                            print(data_str, end='', flush=True)
                            full_response += data_str

            print("\n\n✅ 流式推理测试成功")
            return {"success": True, "response": full_response}

        except Exception as e:
            print(f"\n❌ 流式推理测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
            return {"error": str(e)}

    def test_batch_inference(self, model_path: str):
        """测试批量推理API"""
        print("\n" + "=" * 50)
        print("测试批量推理API")
        print("=" * 50)

        # 准备请求数据
        request_data = {
            "model_path": model_path,
            "prompts": [
                "1+1等于几？",
                "Python是什么？"
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }

        print(f"\n请求数据:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        try:
            print(f"\n发送批量推理请求到: {self.base_url}/api/inference/batch")
            response = requests.post(
                f"{self.base_url}/api/inference/batch",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()

            print(f"\n批量推理结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result.get("success"):
                print("\n✅ 批量推理测试成功")
            else:
                print("\n❌ 批量推理测试失败")

            return result

        except Exception as e:
            print(f"\n❌ 批量推理测试失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return {"error": str(e)}

    def run_all_tests(self, model_path: str):
        """运行所有推理服务API测试"""
        print("\n" + "=" * 60)
        print("开始运行推理服务API测试")
        print("=" * 60)

        # 测试1: 普通推理
        print("\n\n【测试1】普通推理API")
        chat_result = self.test_chat_inference(
            model_path
        )

        print("\n" + "-" * 60)

        # 测试2: 流式推理
        # print("\n\n【测试2】流式推理API")
        # stream_result = self.test_stream_inference(model_path)
        #
        # print("\n" + "-" * 60)
        #
        # # 测试3: 批量推理
        # print("\n\n【测试3】批量推理API")
        # batch_result = self.test_batch_inference(model_path)

        print("\n" + "=" * 60)
        print("推理服务API测试完成")
        print("=" * 60)

        return {
            "chat": chat_result
            # ,
            # "stream": stream_result,
            # "batch": batch_result
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

    tester = InferenceAPITest()
    tester.run_all_tests(model_path)
