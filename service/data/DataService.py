import json
import os
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from entity.system_model.DatasetModel import Conversation, DatasetSplitConfig, DataValidationResult
import random


class DataService:
    """数据处理服务"""

    def __init__(self):
        pass

    def load_raw_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载原始数据"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            print(f"成功加载 {len(data)} 条原始数据")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise

    def validate_conversations(self, data: List[Dict[str, Any]]) -> DataValidationResult:
        """验证对话数据格式"""
        total_samples = len(data)
        valid_samples = 0
        invalid_samples = 0
        error_messages = []

        for idx, item in enumerate(data):
            try:
                if 'conversations' not in item:
                    error_messages.append(f"第{idx+1}行: 缺少conversations字段")
                    invalid_samples += 1
                    continue

                conversations = item['conversations']
                if not isinstance(conversations, list) or len(conversations) == 0:
                    error_messages.append(f"第{idx+1}行: conversations必须是非空列表")
                    invalid_samples += 1
                    continue

                # 验证每条消息
                valid = True
                for msg in conversations:
                    if 'role' not in msg or 'content' not in msg:
                        error_messages.append(f"第{idx+1}行: 消息缺少role或content字段")
                        valid = False
                        break
                    if msg['role'] not in ['user', 'assistant']:
                        error_messages.append(f"第{idx+1}行: role必须是user或assistant")
                        valid = False
                        break

                if valid:
                    valid_samples += 1
                else:
                    invalid_samples += 1

            except Exception as e:
                error_messages.append(f"第{idx+1}行: {str(e)}")
                invalid_samples += 1

        return DataValidationResult(
            is_valid=(invalid_samples == 0),
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            error_messages=error_messages[:10]  # 只返回前10条错误
        )

    def normalize_conversations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化对话数据格式"""
        normalized_data = []

        for item in data:
            try:
                conversations = item.get('conversations', [])
                normalized_convs = []

                for msg in conversations:
                    # 自动修正键名
                    role = msg.get('role', msg.get('assistant', 'user'))
                    content = msg.get('content', '')

                    # 修正角色值
                    if role not in ['user', 'assistant']:
                        role = 'assistant' if len(normalized_convs) > 0 and normalized_convs[-1]['role'] == 'user' else 'user'

                    if content.strip():
                        normalized_convs.append({'role': role, 'content': content})

                if len(normalized_convs) > 0:
                    normalized_data.append({'conversations': normalized_convs})

            except Exception as e:
                print(f"标准化数据时出错: {e}")
                continue

        print(f"标准化完成，有效数据: {len(normalized_data)} 条")
        return normalized_data

    def split_dataset(self, data: List[Dict[str, Any]], config: DatasetSplitConfig) -> Dict[str, List[Dict[str, Any]]]:
        """划分数据集为训练集、验证集、测试集"""
        total = len(data)

        # 打乱数据
        if config.shuffle:
            random.seed(config.seed)
            data = data.copy()
            random.shuffle(data)

        # 计算划分点
        train_size = int(total * config.train_ratio)
        val_size = int(total * config.val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        print(f"数据集划分完成: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def save_dataset(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据集到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"数据集已保存到: {output_path}")
