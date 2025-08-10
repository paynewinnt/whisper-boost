"""
主程序 - GPU高利用率Whisper转录应用
重构后的模块化版本
"""
import gradio as gr
import torch

# 导入自定义模块
from config.config import OPTIMIZED_MODELS, APP_TITLE, APP_PORT, MAX_FILE_SIZE, DEEPSEEK_API_KEY
from src.utils import get_gpu_info, monitor_gpu_usage
from src.whisper_model import transcribe_high_utilization, clear_all_cache
from src.ai_summary import summarize_with_deepseek
from src.file_operations import save_transcript_with_dialog, save_summary_with_dialog
from src.ui_components import create_system_status_html, create_api_status_components, create_performance_info

def main():
    """主函数"""
    # 系统信息
    gpu_available, gpu_name, gpu_memory = get_gpu_info()
    print(f"GPU: {gpu_name} ({gpu_memory})")
    print(f"Device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    
    # 创建界面
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# ⚡ GPU高利用率Whisper转录")
        
        # 系统状态
        gr.HTML(create_system_status_html())
        
        with gr.Row():
            with gr.Column():
                # 视频下载提示
                gr.HTML("""
                <div style="padding: 10px; background-color: #f0f8ff; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #007acc;">
                    <p style="margin: 0; color: #333;">
                        💡 <strong>需要下载在线视频？</strong> 
                        <a href="https://dy.kukutool.com/" target="_blank" style="color: #007acc; text-decoration: none;">
                            点击这里访问视频解析工具 →
                        </a>
                    </p>
                    <small style="color: #666;">支持抖音、B站、小红书等平台，下载后直接上传即可</small>
                </div>
                """)
                
                # 文件上传
                audio_input = gr.File(
                    label="📁 上传音视频文件",
                    file_types=["audio", "video"]
                )
                
                # 优化模型选择
                model_input = gr.Dropdown(
                    choices=list(OPTIMIZED_MODELS.keys()),
                    value="Medium (高利用率)",
                    label="🤖 选择GPU优化模型",
                    info="所有模型都针对高利用率优化"
                )
                
                # 语言
                language_input = gr.Dropdown(
                    choices=[("中文", "chinese"), ("英语", "english"), ("自动", "auto")],
                    value="chinese",
                    label="🌐 语言"
                )
                
                # 操作按钮
                with gr.Row():
                    preview_btn = gr.Button("⚡ 高速预览", variant="secondary", size="lg")
                    full_btn = gr.Button("🎯 GPU全力转录", variant="primary", size="lg")
                
                # 管理按钮
                with gr.Row():
                    clear_cache_btn = gr.Button("🗑️ 清理缓存", variant="secondary")
                    gpu_monitor = gr.Textbox(
                        label="🎮 GPU状态",
                        value=monitor_gpu_usage(),
                        interactive=False,
                        max_lines=1
                    )
            
            with gr.Column():
                # 转录结果
                transcript_output = gr.Textbox(
                    label="📝 转录结果 + 性能统计",
                    lines=25,
                    show_copy_button=True,
                    placeholder="GPU优化转录结果和详细性能统计..."
                )
                
                # 保存和总结功能
                with gr.Row():
                    save_btn = gr.Button("💾 保存转录文本", variant="secondary")
                    summary_btn = gr.Button("🤖 AI总结", variant="primary")
                
                save_status = gr.Textbox(
                    label="保存状态",
                    interactive=False,
                    max_lines=2,
                    visible=False
                )
                
                # API Key配置区域
                api_status, api_key_input = create_api_status_components()
                
                # 总结结果显示
                summary_output = gr.Textbox(
                    label="🤖 AI总结结果",
                    lines=15,
                    show_copy_button=True,
                    placeholder="AI总结将在这里显示...",
                    visible=False
                )
                
                # 保存AI总结按钮
                save_summary_btn = gr.Button("💾 保存AI总结", variant="secondary", visible=False)
                save_summary_status = gr.Textbox(
                    label="AI总结保存状态",
                    interactive=False,
                    max_lines=2,
                    visible=False
                )
        
        # 性能说明
        with gr.Accordion("⚡ GPU优化说明", open=False):
            gr.Markdown(create_performance_info())
        
        # 事件绑定
        def setup_event_handlers():
            """设置事件处理器"""
            
            # 转录事件
            preview_btn.click(
                fn=lambda *args: transcribe_high_utilization(*args, preview_mode=True),
                inputs=[audio_input, model_input, language_input],
                outputs=[transcript_output, gr.State(), gpu_monitor]
            )
            
            full_btn.click(
                fn=lambda *args: transcribe_high_utilization(*args, preview_mode=False),
                inputs=[audio_input, model_input, language_input],
                outputs=[transcript_output, gr.State(), gpu_monitor]
            )
            
            # 缓存清理
            clear_cache_btn.click(
                fn=clear_all_cache,
                outputs=[gpu_monitor]
            )
            
            # 保存转录文本
            save_btn.click(
                fn=save_transcript_with_dialog,
                inputs=[transcript_output],
                outputs=[save_status]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[save_status]
            )
            
            # AI总结相关事件
            def start_summary():
                return gr.update(value="🤖 正在生成AI总结，请稍候...", visible=True), gr.update(visible=False)
            
            def show_summary_button(summary_result):
                if summary_result and not summary_result.startswith("❌") and "正在生成AI总结" not in summary_result:
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)
            
            summary_btn.click(
                fn=start_summary,
                outputs=[summary_output, save_summary_btn]
            ).then(
                fn=summarize_with_deepseek,
                inputs=[transcript_output, api_key_input],
                outputs=[summary_output]
            ).then(
                fn=show_summary_button,
                inputs=[summary_output],
                outputs=[save_summary_btn]
            )
            
            # 保存AI总结
            save_summary_btn.click(
                fn=save_summary_with_dialog,
                inputs=[summary_output],
                outputs=[save_summary_status]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[save_summary_status]
            )
        
        setup_event_handlers()
    
    return demo

if __name__ == "__main__":
    print("启动GPU高利用率优化服务...")
    
    # GPU预热和优化
    if torch.cuda.is_available():
        print("GPU预热和编译优化...")
        try:
            dummy = torch.randn(1, 1000, device="cuda:0")
            _ = dummy @ dummy.T
            del dummy
            torch.cuda.empty_cache()
            print("GPU预热完成")
        except Exception as e:
            print(f"GPU预热跳过: {e}")
    
    # 启动应用
    demo = main()
    demo.launch(
        server_name="0.0.0.0",
        server_port=APP_PORT,
        share=True,
        show_error=True,
        max_file_size=MAX_FILE_SIZE
    )