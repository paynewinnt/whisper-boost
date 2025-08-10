"""
UI组件模块 - 处理Gradio界面组件
"""
import gradio as gr
from config.config import DEEPSEEK_API_KEY, OPTIMIZED_MODELS
from src.utils import get_gpu_info

def create_system_status_html():
    """创建系统状态HTML"""
    _, gpu_name, gpu_memory = get_gpu_info()
    
    return f"""
    <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3>🎮 GPU优化状态</h3>
        <p><strong>设备:</strong> {gpu_name} ({gpu_memory})</p>
        <p><strong>优化:</strong> ✅ torch.compile + AMP + 并行处理</p>
        <p><strong>目标:</strong> 最大化GPU利用率，最小化处理时间</p>
    </div>
    """

def create_api_status_components():
    """创建API状态组件"""
    if DEEPSEEK_API_KEY:
        # 如果代码中已配置API Key，显示状态
        api_status = gr.HTML(
            value="""
            <div style="padding: 10px; background-color: #d4edda; border-radius: 5px; margin: 10px 0;">
                <strong>🔑 DeepSeek API状态:</strong> ✅ 已在代码中配置
            </div>
            """
        )
        api_key_input = gr.Textbox(visible=False)  # 隐藏输入框
    else:
        # 如果代码中未配置，显示输入框
        api_status = gr.HTML(
            value="""
            <div style="padding: 10px; background-color: #fff3cd; border-radius: 5px; margin: 10px 0;">
                <strong>🔑 DeepSeek API状态:</strong> ⚠️ 请在下方输入API Key
            </div>
            """
        )
        api_key_input = gr.Textbox(
            label="🔑 DeepSeek API Key",
            type="password",
            placeholder="请输入您的DeepSeek API Key",
            info="用于AI文本总结功能"
        )
    
    return api_status, api_key_input

def create_performance_info():
    """创建性能说明内容"""
    return """
    ### 🎯 高利用率优化特性:
    
    **GPU优化技术:**
    - ✅ **torch.compile**: 模型编译优化，提升15-30%性能
    - ✅ **自动混合精度 (AMP)**: 内存节省50%，速度提升20%  
    - ✅ **大批处理**: 批次大小优化到16-32，提升吞吐量
    - ✅ **Flash Attention**: 内存高效的注意力机制
    - ✅ **并行预处理**: CPU-GPU流水线并行
    
    **支持的输入方式:**
    - 📁 **本地文件**: 上传音频/视频文件
    - 🔗 **在线视频**: 使用kukutool.com等工具下载后上传
    
    **预期性能 (RTX 4070):**
    - **Small模型**: 15分钟视频 → ~2分钟处理
    - **Medium模型**: 15分钟视频 → ~3分钟处理  
    - **Large模型**: 15分钟视频 → ~5分钟处理
    
    **GPU利用率目标:**
    - 目标利用率: >80%
    - 内存利用率: 70-90%
    - 处理速度: 比标准配置快2-3倍
    """

