from typing import List, Dict, Any, Optional, AsyncGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch
import json


class InferenceService:
    """推理服务 - 支持流式输出和批量推理"""

    def __init__(self, model_path: str, lora_adapter_path: Optional[str] = None):
        """
        初始化推理服务
        Args:
            model_path: 基础模型路径
            lora_adapter_path: LoRA适配器路径（可选）
        """
        print(f"加载模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            use_cache=True
        )

        # 如果提供了LoRA适配器路径，加载适配器
        if lora_adapter_path:
            print(f"加载LoRA适配器: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

        print("模型加载完成")

    def generate(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        普通推理（非流式）
        Args:
            messages: 对话消息列表
            config: 推理配置
        Returns:
            推理结果
        """
        enable_thinking = config.get('enable_thinking', False)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=config.get('max_new_tokens', 512),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.8),
            top_k=config.get('top_k', 20)
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 如果启用thinking模式，解析思考内容
        thinking_content = None
        if enable_thinking:
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        return {
            'content': content,
            'thinking_content': thinking_content
        }

    async def generate_stream(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        流式推理（SSE）
        Args:
            messages: 对话消息列表
            config: 推理配置
        Yields:
            生成的文本片段
        """
        enable_thinking = config.get('enable_thinking', False)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 使用生成器逐token生成
        max_new_tokens = config.get('max_new_tokens', 512)

        for i in range(max_new_tokens):
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=1,
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.8),
                top_k=config.get('top_k', 20),
                do_sample=True
            )

            new_token_id = outputs[0, -1].item()
            new_token = self.tokenizer.decode([new_token_id], skip_special_tokens=True)

            # 流式返回token
            yield f"data: {json.dumps({'token': new_token})}\n\n"

            # 检查是否结束
            if new_token_id == self.tokenizer.eos_token_id:
                break

            # 更新输入
            model_inputs = {"input_ids": outputs, "attention_mask": torch.ones_like(outputs)}

    def batch_generate(self, prompts: List[str], config: Dict[str, Any]) -> List[str]:
        """
        批量推理
        Args:
            prompts: 提示词列表
            config: 推理配置
        Returns:
            生成结果列表
        """
        print(f"批量推理，共 {len(prompts)} 条数据")

        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            result = self.generate(messages, config)
            results.append(result['content'])

        print("批量推理完成")
        return results
