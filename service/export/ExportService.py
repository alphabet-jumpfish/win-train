import torch
import os
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class ExportService:
    """模型导出服务 - 支持ONNX等格式"""

    def __init__(self):
        pass

    def export_to_onnx(self, model_path: str, output_path: str, opset_version: int = 14) -> Dict[str, Any]:
        """
        导出模型为ONNX格式
        Args:
            model_path: 模型路径
            output_path: 输出路径
            opset_version: ONNX opset版本
        Returns:
            导出结果
        """
        print(f"开始导出模型为ONNX格式: {model_path}")

        try:
            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32
            )

            model.eval()

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 准备示例输入
            dummy_input = tokenizer("Hello, world!", return_tensors="pt")

            print("正在导出ONNX模型...")

            # 导出为ONNX
            torch.onnx.export(
                model,
                (dummy_input['input_ids'],),
                output_path,
                opset_version=opset_version,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                }
            )

            file_size = os.path.getsize(output_path)
            print(f"ONNX模型导出成功: {output_path}, 文件大小: {file_size} 字节")

            return {
                "success": True,
                "message": "ONNX模型导出成功",
                "export_path": output_path,
                "file_size": file_size
            }

        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            print(f"ONNX导出失败: {error_msg}")
            return {
                "success": False,
                "message": f"导出失败: {error_msg}",
                "export_path": None,
                "file_size": None
            }
