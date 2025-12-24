import sys
import os
import yaml

# 设置UTF-8编码，避免Windows控制台编码问题
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 导入所有测试类
from DataAPITest import DataAPITest
from TrainManagementAPITest import TrainManagementAPITest
from InferenceAPITest import InferenceAPITest
from EvaluationAPITest import EvaluationAPITest
from ExportAPITest import ExportAPITest


class AllAPITest:
    """所有API测试的主运行器"""

    def __init__(self, base_url: str = "http://127.0.0.1:8801"):
        self.base_url = base_url
        self.data_tester = DataAPITest(base_url)
        self.train_tester = TrainManagementAPITest(base_url)
        self.inference_tester = InferenceAPITest(base_url)
        self.eval_tester = EvaluationAPITest(base_url)
        self.export_tester = ExportAPITest(base_url)

    def run_all_tests(self, model_path: str):
        """运行所有API测试"""
        print("\n" + "=" * 80)
        print("开始运行所有API测试")
        print("=" * 80)

        results = {}

        # 测试1: 数据处理API
        print("\n\n" + "#" * 80)
        print("# 第一部分：数据处理API测试")
        print("#" * 80)
        try:
            results['data'] = self.data_tester.run_all_tests()
        except Exception as e:
            print(f"数据处理API测试失败: {e}")
            results['data'] = {"error": str(e)}

        print("\n\n" + "#" * 80)
        print("# 第二部分：训练管理API测试")
        print("#" * 80)
        try:
            results['train'] = self.train_tester.run_all_tests(model_path)
        except Exception as e:
            print(f"训练管理API测试失败: {e}")
            results['train'] = {"error": str(e)}

        print("\n\n" + "#" * 80)
        print("# 第三部分：推理服务API测试")
        print("#" * 80)
        try:
            results['inference'] = self.inference_tester.run_all_tests(model_path)
        except Exception as e:
            print(f"推理服务API测试失败: {e}")
            results['inference'] = {"error": str(e)}

        print("\n\n" + "#" * 80)
        print("# 第四部分：评估服务API测试")
        print("#" * 80)
        try:
            results['evaluation'] = self.eval_tester.run_all_tests(model_path)
        except Exception as e:
            print(f"评估服务API测试失败: {e}")
            results['evaluation'] = {"error": str(e)}

        print("\n\n" + "#" * 80)
        print("# 第五部分：导出服务API测试")
        print("#" * 80)
        try:
            results['export'] = self.export_tester.run_all_tests(model_path)
        except Exception as e:
            print(f"导出服务API测试失败: {e}")
            results['export'] = {"error": str(e)}

        # 打印测试总结
        print("\n\n" + "=" * 80)
        print("所有API测试完成")
        print("=" * 80)
        
        print("\n测试结果总结:")
        print("-" * 80)
        
        for category, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                print(f"❌ {category}: 失败 - {result['error']}")
            else:
                print(f"✅ {category}: 成功")
        
        print("=" * 80)
        
        return results


if __name__ == "__main__":
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            model_path = config['model']['base_model_path']
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        print("请确保 config.yaml 文件存在且配置正确")
        sys.exit(1)

    # 创建测试实例并运行所有测试
    tester = AllAPITest()
    tester.run_all_tests(model_path)
