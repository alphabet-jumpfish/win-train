# Win-Train 模型训练管理系统

基于 FastAPI 的大语言模型训练、推理、评估和导出管理系统。支持 TRL 和 LoRA 两种训练策略，提供完整的模型训练生命周期管理。

## 📋 目录

- [系统架构](#系统架构)
- [模型训练完整流程](#模型训练完整流程)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
                                                                                                                                                                                                                                                                                                            
---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 应用层                           │
│  (Controller: 数据/训练/推理/评估/导出)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     业务服务层 (Service)                     │
│  数据处理 | 训练管理 | 推理服务 | 评估服务 | 导出服务       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  实体层 (Entity)                             │
│  配置模型 | 数据模型 | 请求/响应模型 | 任务模型             │
└─────────────────────────────────────────────────────────────┘
```

**核心特性：**
- ✅ 策略模式训练：支持 TRL 和 LoRA 两种训练方式
- ✅ 异步任务管理：后台训练，实时进度查询
- ✅ 流式推理：SSE 流式输出，支持实时响应
- ✅ 批量推理：支持批量数据推理
- ✅ 模型评估：集成 evalscope 评测服务
- ✅ 模型导出：支持 ONNX 格式导出

---

## 🔄 模型训练完整流程

本系统完整实现了从数据处理到模型部署的全流程管理：

```
数据清洗/标注 → 特征工程 → 模型构建 → 训练(前向/反向传播) → 验证/调参 → 测试 → 部署
```

### 第一步：数据清洗/标注

**功能说明：**
- 验证数据格式的正确性
- 标准化对话数据（修正角色、内容格式）
- 数据集划分（训练集/验证集/测试集）

**对应 API：**
- `POST /api/data/validate` - 验证数据格式
- `POST /api/data/process` - 处理和划分数据集

**示例：**
```bash
# 验证数据格式
curl -X POST "http://127.0.0.1:8801/api/data/validate?file_path=/path/to/data.json"

# 处理数据集
curl -X POST "http://127.0.0.1:8801/api/data/process" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "/path/to/raw_data.json",
    "output_dir": "/path/to/output",
    "split_config": {
      "train_ratio": 0.8,
      "val_ratio": 0.1,
      "test_ratio": 0.1,
      "shuffle": true
    }
  }'
```

---

### 第二步：特征工程

**功能说明：**
- 自动进行对话格式转换
- Tokenization（分词）
- 应用 chat_template 格式化
- 数据预处理和标准化

**实现位置：**
- `service/data/DataService.py` - 数据标准化
- `service/train/TRLTrainStrategy.py` - TRL 数据准备
- `service/train/LoRATrainStrategy.py` - LoRA 数据准备

**说明：** 特征工程在训练过程中自动完成，无需单独调用 API。

---

### 第三步：模型构建

**功能说明：**
- 选择训练策略（TRL 或 LoRA）
- 加载预训练模型
- 配置训练参数（学习率、batch size 等）
- 应用 LoRA 适配器（如果使用 LoRA 策略）

**对应 API：**
- `POST /api/train/start` - 启动训练任务（包含模型构建）

**示例：**
```bash
# 使用 TRL 策略训练
curl -X POST "http://127.0.0.1:8801/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "trl",
    "dataset_path": "/path/to/train.json",
    "output_dir": "/path/to/output",
    "config": {
      "model_path": "/path/to/base_model",
      "per_device_train_batch_size": 2,
      "learning_rate": 2e-4,
      "max_steps": 1000
    }
  }'
```

---

### 第四步：训练（前向/反向传播）

**功能说明：**
- 异步后台训练任务
- 前向传播：计算模型输出
- 计算损失函数
- 反向传播：计算梯度
- 参数更新：优化器更新权重
- 实时监控训练进度

**对应 API：**
- `POST /api/train/start` - 启动训练（异步）
- `GET /api/train/status/{task_id}` - 查询训练进度
- `GET /api/train/tasks` - 获取所有训练任务

**示例：**
```bash
# 启动训练后会返回 task_id
# 查询训练进度
curl -X GET "http://127.0.0.1:8801/api/train/status/{task_id}"

# 响应示例
{
  "task_id": "xxx-xxx-xxx",
  "status": "running",
  "progress": 45.5,
  "current_step": 455,
  "total_steps": 1000,
  "loss": 0.234
}
```

---

### 第五步：验证/调参

**功能说明：**
- 在验证集上评估模型性能
- 计算评估指标（loss, perplexity 等）
- 诊断过拟合/欠拟合问题
- 根据验证结果调整超参数

**对应 API：**
- `POST /api/eval/evaluate` - 评估模型性能

**示例：**
```bash
curl -X POST "http://127.0.0.1:8801/api/eval/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/trained_model",
    "dataset_path": "/path/to/val.json",
    "metrics": ["loss", "perplexity"],
    "batch_size": 8
  }'
```

---

### 第六步：测试

**功能说明：**
- 在测试集上进行最终评估
- 验证模型的泛化能力
- 生成评估报告

**对应 API：**
- `POST /api/eval/evaluate` - 使用测试集评估

**示例：**
```bash
curl -X POST "http://127.0.0.1:8801/api/eval/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/final_model",
    "dataset_path": "/path/to/test.json",
    "metrics": ["loss", "perplexity", "accuracy"],
    "batch_size": 8
  }'
```

---

### 第七步：部署

**功能说明：**
- 模型推理服务（普通/流式/批量）
- 模型导出（ONNX 格式）
- 生产环境部署

**对应 API：**
- `POST /api/inference/chat` - 普通推理
- `POST /api/inference/chat/stream` - 流式推理（SSE）
- `POST /api/inference/batch` - 批量推理
- `POST /api/export/onnx` - 导出 ONNX 模型

**示例：**
```bash
# 普通推理
curl -X POST "http://127.0.0.1:8801/api/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，介绍一下自己"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'

# 导出 ONNX 模型
curl -X POST "http://127.0.0.1:8801/api/export/onnx" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/system_model",
    "output_path": "/path/to/output.onnx",
    "export_format": "onnx",
    "opset_version": 14
  }'
```

---


## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置文件

修改 `config.yaml` 中的路径配置：

```yaml
model:
  base_model_path: 'D:/path/to/your/base_model'
  save_path: 'D:/path/to/Win-Train/system_model'
  checkpoint_path: 'D:/path/to/Win-Train/checkpoints'

dataset:
  train_dataset_path: 'D:/path/to/Win-Train/data/train.json'
  val_dataset_path: 'D:/path/to/Win-Train/data/val.json'
  test_dataset_path: 'D:/path/to/Win-Train/data/test.json'
```

### 3. 启动服务

```bash
python main.py
```

服务启动后访问：
- API 文档：http://127.0.0.1:8801/docs
- 健康检查：http://127.0.0.1:8801/health

---


## 📚 API 文档

### 数据处理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/data/validate` | 验证数据格式 |
| POST | `/api/data/process` | 处理和划分数据集 |

### 训练管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train/start` | 启动训练任务（异步） |
| GET | `/api/train/status/{task_id}` | 查询训练进度 |
| GET | `/api/train/tasks` | 获取所有训练任务 |

### 推理服务 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/inference/chat` | 普通推理 |
| POST | `/api/inference/chat/stream` | 流式推理（SSE） |
| POST | `/api/inference/batch` | 批量推理 |

### 评估服务 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/eval/evaluate` | 评估模型性能 |

### 导出服务 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/export/onnx` | 导出 ONNX 格式模型 |

---


## 📁 项目结构

```
Win-Train/
├── config.yaml              # 配置文件
├── main.py                  # FastAPI主应用
├── requirements.txt         # 依赖包
├── README.md               # 项目说明
│
├── entity/                 # 实体层（数据模型）
│   ├── config/            # 配置模型
│   │   └── TrainConfig.py
│   ├── system_model/             # 数据模型
│   │   └── DatasetModel.py
│   ├── request/           # 请求模型
│   │   ├── InferenceModel.py
│   │   ├── EvalModel.py
│   │   └── ExportModel.py
│   ├── response/          # 响应模型
│   │   └── ResponseModel.py
│   └── task/              # 任务模型
│       └── TaskModel.py
│
├── service/               # 服务层（业务逻辑）
│   ├── data/             # 数据处理服务
│   │   └── DataService.py
│   ├── train/            # 训练服务
│   │   ├── TrainStrategy.py
│   │   ├── TRLTrainStrategy.py
│   │   ├── LoRATrainStrategy.py
│   │   └── TrainService.py
│   ├── inference/        # 推理服务
│   │   └── InferenceService.py
│   ├── eval/             # 评估服务
│   │   └── EvalService.py
│   └── export/           # 导出服务
│       └── ExportService.py
│
├── controller/            # 控制器层（API接口）
│   ├── DataController.py
│   ├── TrainController.py
│   ├── InferenceController.py
│   ├── EvalController.py
│   └── ExportController.py
│
├── util/                  # 工具类
│   ├── WinConfigUtil.py
│   └── WinConstant.py
│
└── model/                 # 模型存储目录
    └── (用于保存训练模型和下载模型)
```

---


## ⚙️ 配置说明

### 训练策略选择

系统支持两种训练策略：

**1. TRL 策略**
- 使用 SFTTrainer 进行监督微调
- 适合全参数微调场景
- 训练速度较快

**2. LoRA 策略**
- 使用 PEFT 的 LoRA 进行参数高效微调
- 只训练少量参数，节省显存
- 适合资源受限场景

### 数据格式要求

训练数据必须采用 conversations 格式：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
  ]
}
```

每行一个 JSON 对象（JSONL 格式）。

---


## 💡 完整使用示例

### 示例 1：使用 TRL 策略训练模型

```bash
# 1. 处理数据
curl -X POST "http://127.0.0.1:8801/api/data/process" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "/path/to/raw_data.json",
    "output_dir": "/path/to/processed",
    "split_config": {
      "train_ratio": 0.8,
      "val_ratio": 0.1,
      "test_ratio": 0.1
    }
  }'

# 2. 启动训练
curl -X POST "http://127.0.0.1:8801/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "trl",
    "dataset_path": "/path/to/processed/train.json",
    "output_dir": "/path/to/output",
    "config": {
      "model_path": "/path/to/base_model",
      "per_device_train_batch_size": 2,
      "learning_rate": 2e-4,
      "max_steps": 1000
    }
  }'
```


# 3. 查询训练进度
curl -X GET "http://127.0.0.1:8801/api/train/status/{task_id}"

# 4. 评估模型
curl -X POST "http://127.0.0.1:8801/api/eval/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/output",
    "dataset_path": "/path/to/processed/test.json",
    "metrics": ["loss", "perplexity"],
    "batch_size": 8
  }'

# 5. 推理测试
curl -X POST "http://127.0.0.1:8801/api/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "max_tokens": 512
  }'
```


### 示例 2：使用 LoRA 策略训练模型

```bash
# 启动 LoRA 训练
curl -X POST "http://127.0.0.1:8801/api/train/start" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "lora",
    "dataset_path": "/path/to/train.json",
    "output_dir": "/path/to/lora_output",
    "config": {
      "model_path": "/path/to/base_model",
      "per_device_train_batch_size": 7,
      "learning_rate": 5e-5,
      "num_train_epochs": 5,
      "lora_config": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05
      }
    }
  }'
```

---


## ⚠️ 注意事项

1. **路径配置**：所有路径必须使用绝对路径，Windows 系统建议使用正斜杠 `/`
2. **显存要求**：TRL 策略需要更多显存，LoRA 策略更节省资源
3. **数据格式**：训练数据必须严格遵循 conversations 格式
4. **异步训练**：训练任务在后台执行，通过 task_id 查询进度
5. **模型保存**：训练完成后模型自动保存到指定的 output_dir

---


## 🛠️ 技术栈

- **Web框架**: FastAPI 0.104.1
- **模型框架**: Transformers 4.57.3, ModelScope 1.32.0
- **训练框架**: TRL 0.26.0, PEFT (LoRA)
- **深度学习**: PyTorch 2.0+
- **数据处理**: Datasets, Pandas
- **模型评测**: EvalScope
- **模型导出**: ONNX 1.19.1

---

## 📝 开发建议

1. **从简单开始**：先用小数据集跑通完整流程
2. **监控训练**：实时查看训练进度和 loss 变化
3. **验证集评估**：及时发现过拟合问题
4. **参数调优**：根据验证集表现调整超参数
5. **测试集评估**：最后在测试集上验证模型性能

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！


---

## 🧪 测试说明

### 自动化测试

项目提供了完整的训练API自动化测试，包含TRL和LoRA两种训练策略的测试。

**测试文件位置：**
- 测试类：`test/TrainAPITest.py`
- 测试数据：`test/data/train_sample.json`
- 测试输出：`test/output/`

### 运行测试

**前提条件：**
1. 确保服务已启动：`python main.py`
2. 确保 `config.yaml` 中配置了正确的模型路径

**执行测试：**
```bash
cd test
python TrainAPITest.py
```


### 测试内容

**测试1：TRL训练策略**
- 策略：TRL (SFTTrainer)
- 数据集：5条样本数据
- 训练步数：10步（快速测试）
- 输出目录：`test/output/trl_output/`

**测试2：LoRA训练策略**
- 策略：LoRA (参数高效微调)
- 数据集：5条样本数据
- 训练轮数：1轮（快速测试）
- 输出目录：`test/output/lora_output/`

### 测试输出示例

```
============================================================
开始运行训练API测试
============================================================

【测试1】TRL训练策略
==================================================
开始测试 TRL 训练策略
==================================================

请求数据:
{
  "strategy": "trl",
  "dataset_path": "test/data/train_sample.json",
  "output_dir": "test/output/trl_output",
  "config": {
    "model_path": "/path/to/base_model",
    "per_device_train_batch_size": 1,
    "learning_rate": 0.0002,
    "max_steps": 10
  }
}

训练任务已创建:
Task ID: xxx-xxx-xxx
Status: pending
Message: 训练任务已创建，正在后台执行

✅ TRL训练任务创建成功
```


### 测试数据格式

测试数据采用标准的 conversations 格式（JSONL）：

```json
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！我是AI助手，很高兴为您服务。"}]}
{"conversations": [{"role": "user", "content": "介绍一下Python"}, {"role": "assistant", "content": "Python是一种高级编程语言，以其简洁易读的语法而闻名。"}]}
```

### 自定义测试

您可以修改 `test/data/train_sample.json` 来使用自己的测试数据，或者直接调用测试类的方法：

```python
from test.TrainAPITest import TrainAPITest

# 创建测试实例
tester = TrainAPITest(base_url="http://127.0.0.1:8801")

# 测试TRL训练
trl_result = tester.test_trl_training(model_path="/path/to/model")

# 测试LoRA训练
lora_result = tester.test_lora_training(model_path="/path/to/model")

# 查询任务状态
status = tester.check_task_status(task_id="xxx-xxx-xxx")
```

---

