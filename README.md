# GPU高利用率Whisper转录系统

一个优化的语音转录应用，集成DeepSeek AI总结功能，采用模块化架构设计。

## 🌟 主要特性

- 🎯 **GPU高利用率优化** - 最大化GPU使用效率，提升转录速度2-3倍
- 🤖 **AI智能总结** - 集成DeepSeek R1模型，自动生成文本摘要
- 🔗 **在线视频支持** - 提供便捷的视频下载工具链接
- 💾 **灵活文件保存** - 支持自定义路径和文件名保存
- ⚡ **预览与完整模式** - 快速预览前3分钟或完整转录
- 🎮 **实时GPU监控** - 动态显示GPU使用情况
- 🔧 **模块化架构** - 代码结构清晰，易于维护和扩展

## 🏗️ 架构设计

### 模块结构
```
whisper-medium/
├── main.py                 # 主程序入口
├── README.md              # 项目说明文档
├── config/                # 配置模块
│   ├── __init__.py
│   └── config.py          # API密钥和模型设置
├── src/                   # 源代码模块
│   ├── __init__.py
│   ├── utils.py           # 工具函数，系统检测和通用功能
│   ├── whisper_model.py   # Whisper模型管理和转录核心
│   ├── ai_summary.py      # DeepSeek AI总结功能
│   ├── file_operations.py # 文件保存和对话框处理
│   └── ui_components.py   # Gradio界面组件构建
├── video/                 # 测试视频文件
├── ffmpeg/                # FFmpeg工具
└── docs/                  # 文档目录（预留）
```

### 模块职责

| 模块 | 主要功能 | 核心类/函数 |
|------|----------|-------------|
| **config/config.py** | 配置管理 | `DEEPSEEK_API_KEY`, `OPTIMIZED_MODELS` |
| **src/utils.py** | 系统工具 | `check_ffmpeg()`, `get_gpu_info()`, `monitor_gpu_usage()` |
| **src/whisper_model.py** | 转录核心 | `OptimizedWhisperModel`, `transcribe_high_utilization()` |
| **src/ai_summary.py** | AI总结 | `summarize_with_deepseek()` |
| **src/file_operations.py** | 文件操作 | `save_transcript_with_dialog()`, `save_summary_with_dialog()` |
| **src/ui_components.py** | UI构建 | `create_system_status_html()`, `create_api_status_components()` |
| **main.py** | 主程序 | `main()`, 事件绑定和应用启动 |

## 🚀 快速开始

### 环境要求
- **Python**: 3.8+
- **PyTorch**: 支持CUDA的版本
- **GPU**: 推荐RTX 3060及以上
- **内存**: 建议8GB+

### 安装依赖
```bash
# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers gradio requests

# 或使用requirements.txt一键安装
pip install -r requirements.txt
```

### 配置API Key
在 `config/config.py` 中设置DeepSeek API密钥：
```python
DEEPSEEK_API_KEY = "sk-your-api-key-here"
```

如果不配置，也可以在界面上动态输入。

### 启动应用
```bash
python main.py
```

应用将在 `http://localhost:7862` 启动。

## 📋 使用指南

### 基本操作流程

#### 📁 本地文件转录
1. **上传文件** - 支持音频和视频格式
2. **选择模型** - Small/Medium/Large三种GPU优化模型
3. **选择语言** - 中文/英语/自动检测
4. **开始转录** - 预览模式或完整转录
5. **保存结果** - 自定义保存路径和文件名
6. **AI总结** - 一键生成智能摘要（可选）

#### 🔗 在线视频处理
1. **访问下载工具** - 点击界面提供的kukutool.com链接
2. **下载视频** - 在工具网站解析并下载所需视频
3. **上传转录** - 将下载的视频文件上传到本应用
4. **获得结果** - 完成转录和AI总结

### 模型选择建议
| 模型 | 适用场景 | 处理速度 | 准确率 | 显存需求 |
|------|----------|----------|--------|----------|
| **Small** | 快速预览 | 最快 | 良好 | 2GB |
| **Medium** | 平衡选择 | 中等 | 很好 | 4GB |
| **Large-v3** | 高精度 | 较慢 | 最佳 | 8GB |

### 性能优化特性
- ✅ **torch.compile**: 模型编译优化，提升15-30%性能
- ✅ **自动混合精度(AMP)**: 内存节省50%，速度提升20%
- ✅ **大批处理**: 批次大小优化，提升吞吐量
- ✅ **并行预处理**: CPU-GPU流水线并行
- ✅ **内存管理**: 智能缓存清理机制

## 🔧 配置说明

### API配置
```python
# config/config.py
DEEPSEEK_API_KEY = "your-key"        # DeepSeek API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"  # 使用的模型
```

### 模型配置
```python
OPTIMIZED_MODELS = {
    "Small (GPU优化)": {
        "name": "openai/whisper-small",
        "batch_size": 16
    },
    "Medium (高利用率)": {
        "name": "openai/whisper-medium", 
        "batch_size": 8
    },
    "Large-v3 (最大化GPU)": {
        "name": "openai/whisper-large-v3",
        "batch_size": 4
    }
}
```

## 📊 性能基准

### 测试环境：RTX 4070
| 模型 | 15分钟视频处理时间 | GPU利用率 | 内存使用 |
|------|-------------------|-----------|----------|
| Small | ~2分钟 | 85%+ | 3GB |
| Medium | ~3分钟 | 80%+ | 5GB |
| Large | ~5分钟 | 75%+ | 8GB |

### 与标准配置对比
- **处理速度**: 提升2-3倍
- **GPU利用率**: 从40%提升到80%+
- **内存效率**: 提升50%

## 🛠️ 开发说明

### 架构优势
1. **模块化设计** - 职责分离，易于维护
2. **配置集中** - 统一管理所有配置项
3. **错误处理** - 完善的异常处理机制
4. **扩展性** - 易于添加新功能模块

### 自定义开发
```python
# 添加新的转录后处理
from src.whisper_model import transcribe_high_utilization

def custom_post_process(transcript):
    # 自定义处理逻辑
    return processed_transcript

# 扩展AI功能
from src.ai_summary import summarize_with_deepseek

def custom_ai_analysis(text):
    # 自定义AI分析
    return analysis_result
```

### 故障排除
- **模型加载失败**: 检查网络连接和磁盘空间
- **GPU内存不足**: 降低批处理大小或使用小模型
- **转录结果为空**: 检查音频质量和语言设置
- **API调用失败**: 验证API密钥和网络连接
- **在线视频处理**: 使用kukutool.com等第三方工具下载视频后上传

## 📝 更新日志

### v2.1.0 - 界面优化
- 🔗 添加视频下载工具链接提示
- 📱 简化用户界面，专注核心功能
- 🛡️ 优化用户体验和操作流程
- 🧹 清理不必要的依赖项

### v2.0.0 - 模块化重构
- ✨ 完全重构为模块化架构
- 🚀 提升代码可维护性和扩展性
- 🔧 优化配置管理
- 📦 拆分功能模块，职责清晰

### v1.0.0 - 初始版本
- 🎯 GPU高利用率转录
- 🤖 DeepSeek AI总结集成
- 💾 文件保存功能
- 🎮 GPU监控

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题请提交 Issue 或联系开发者。