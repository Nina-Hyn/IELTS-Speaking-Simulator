import os
import sys
import time
import json
import queue
import threading
import subprocess
import uuid
import cv2
import torch
import whisper
from datetime import datetime
from typing import Tuple

# -------------------------- 1. 全局配置（请根据你的环境修改） --------------------------
CONFIG = {
    "VIDEO_PATH": "original.mp4",          # 你的原视频文件路径（必填）
    "WAV_MONITOR_DIR": "./wav_inputs",     # 监控WAV文件的文件夹（自动创建）
    "OUTPUT_FRAME_DIR": "./output_frames", # 生成帧的输出目录（自动创建）
    "PROCESSING_TMP_DIR": "./processing",  # 处理中WAV的临时目录（自动创建）
    "LOG_FILE": "processing_log.json",     # 处理日志文件
    "MUSETALK_VAE_PATH": "ft-mse-vae.pt",  # MuseTalk的VAE权重路径（必填）
    "MUSETALK_UNET_PATH": "musetalk_unet.pt", # MuseTalk的UNet权重路径（必填）
    "DELAY_SEC": 0.3,                      # 音画同步延迟（默认0.3秒，可微调）
    "NUM_WORKERS": max(1, os.cpu_count() // 2) # 处理线程数（默认CPU核心数的一半）
}

# -------------------------- 2. 工具函数 --------------------------
def init_dirs() -> None:
    """初始化所有必要目录"""
    for dir_path in [
        CONFIG["WAV_MONITOR_DIR"],
        CONFIG["OUTPUT_FRAME_DIR"],
        CONFIG["PROCESSING_TMP_DIR"]
    ]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 目录初始化完成：\n- WAV监控：{CONFIG['WAV_MONITOR_DIR']}\n- 输出帧：{CONFIG['OUTPUT_FRAME_DIR']}\n- 临时处理：{CONFIG['PROCESSING_TMP_DIR']}")

def load_processing_log() -> dict:
    """加载处理日志（避免重复处理已完成的WAV）"""
    if os.path.exists(CONFIG["LOG_FILE"]):
        try:
            with open(CONFIG["LOG_FILE"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  日志加载失败，重新创建：{str(e)}")
    return {}

def save_processing_log(log: dict) -> None:
    """保存处理日志"""
    try:
        with open(CONFIG["LOG_FILE"], "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"⚠️  日志保存失败：{str(e)}")

def check_file_ready(file_path: str) -> bool:
    """检查文件是否已完全写入（避免读取未上传完的WAV）"""
    try:
        with open(file_path, "rb") as f:
            f.seek(-1, os.SEEK_END)  # 尝试读取文件最后一个字节
        return True
    except Exception:
        return False

# -------------------------- 3. MuseTalk核心处理模块 --------------------------
class MuseTalkProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 初始化模型（设备：{self.device}）...")
        self.musetalk = self._load_musetalk()
        self.whisper_model = whisper.load_model("tiny", device=self.device)
        print("✅ 模型初始化完成")

    def _load_musetalk(self):
        """加载MuseTalk模型（需提前下载权重）"""
        try:
            from muse_talk import MuseTalk  # 导入MuseTalk（需提前安装）
            return MuseTalk(
                vae_path=CONFIG["MUSETALK_VAE_PATH"],
                unet_path=CONFIG["MUSETALK_UNET_PATH"],
                device=self.device
            )
        except ImportError:
            print("❌ 未安装muse-talk，请先执行：pip install muse-talk")
            sys.exit(1)
        except FileNotFoundError:
            print(f"❌ MuseTalk权重文件未找到，请检查路径：\n- VAE: {CONFIG['MUSETALK_VAE_PATH']}\n- UNET: {CONFIG['MUSETALK_UNET_PATH']}")
            sys.exit(1)

    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """提取音频的Mel频谱特征（供MuseTalk使用）"""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)  # 统一音频长度
        return whisper.log_mel_spectrogram(audio).to(self.device)

    def process_video_frame(self, video_path: str, audio_path: str, output_dir: str) -> None:
        """处理视频，生成嘴形同步的帧"""
        # 1. 提取音频特征
        mel = self.extract_audio_features(audio_path)
        
        # 2. 打开原视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ 无法打开原视频：{video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay_frames = max(1, int(CONFIG["DELAY_SEC"] * fps))  # 延迟帧数（对齐音画）
        
        # 3. 逐帧处理
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频处理完毕
            
            # 输出帧路径（按索引命名，便于前端按顺序读取）
            frame_save_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            
            # 前N帧输出原视频（延迟对齐）
            if frame_idx < delay_frames:
                cv2.imwrite(frame_save_path, frame)
                frame_idx += 1
                continue
            
            # MuseTalk生成嘴形同步帧
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                generated_frame = self.musetalk.generate(
                    source_frame=rgb_frame,
                    ref_frame=rgb_frame,  # 用当前帧做参考（保持身份一致）
                    audio_mel=mel,
                    frame_idx=frame_idx - delay_frames  # 对齐音频时间
                )
                # 转换回BGR格式并保存
                cv2.imwrite(frame_save_path, cv2.cvtColor(generated_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                # 出错时保存原帧（避免中断）
                cv2.imwrite(frame_save_path, frame)
                print(f"⚠️  处理帧 {frame_idx} 出错：{str(e)}")
            
            frame_idx += 1
        
        cap.release()
        # 保存元数据（前端用于同步）
        metadata = {"fps": float(fps), "delay_frames": delay_frames, "total_frames": frame_idx}
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ 视频处理完成：{output_dir}（共{frame_idx}帧）")

# -------------------------- 4. 文件夹监控与任务调度 --------------------------
class WavMonitor:
    def __init__(self, processor: MuseTalkProcessor):
        self.processor = processor
        self.file_queue = queue.Queue()
        self.processing_log = load_processing_log()
        self.processed_files = set(self.processing_log.keys())

    def start_monitor(self) -> None:
        """启动WAV文件夹监控线程"""
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        print("🔍 WAV文件夹监控已启动（每0.5秒检查一次）")

    def _monitor_loop(self) -> None:
        """监控循环：发现新WAV文件则加入队列"""
        while True:
            try:
                # 遍历监控目录下的所有文件
                for filename in os.listdir(CONFIG["WAV_MONITOR_DIR"]):
                    if not filename.lower().endswith(".wav"):
                        continue  # 只处理WAV文件
                    if filename in self.processed_files:
                        continue  # 跳过已处理的文件
                    
                    wav_src_path = os.path.join(CONFIG["WAV_MONITOR_DIR"], filename)
                    if not check_file_ready(wav_src_path):
                        continue  # 跳过未完全写入的文件
                    
                    # 生成唯一ID（避免文件名重复）
                    base_name = os.path.splitext(filename)[0]
                    unique_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
                    wav_dst_path = os.path.join(CONFIG["PROCESSING_TMP_DIR"], f"{unique_id}.wav")
                    
                    # 移动文件到临时目录（避免重复处理）
                    os.rename(wav_src_path, wav_dst_path)
                    
                    # 记录日志
                    self.processing_log[filename] = {
                        "status": "processing",
                        "start_time": datetime.now().isoformat(),
                        "unique_id": unique_id,
                        "output_dir": os.path.join(CONFIG["OUTPUT_FRAME_DIR"], unique_id)
                    }
                    save_processing_log(self.processing_log)
                    
                    # 加入处理队列
                    self.file_queue.put((wav_dst_path, filename, unique_id))
                    self.processed_files.add(filename)
                    print(f"📥 发现新WAV文件：{filename}（唯一ID：{unique_id}）")
                
                time.sleep(0.5)  # 每0.5秒检查一次
            except Exception as e:
                print(f"⚠️  监控线程出错：{str(e)}")
                time.sleep(2)

    def start_workers(self) -> None:
        """启动处理线程池"""
        for _ in range(CONFIG["NUM_WORKERS"]):
            threading.Thread(target=self._worker_loop, daemon=True).start()
        print(f"🚀 启动{CONFIG['NUM_WORKERS']}个处理线程")

    def _worker_loop(self) -> None:
        """处理队列中的WAV文件"""
        while True:
            try:
                wav_path, original_filename, unique_id = self.file_queue.get()
                output_dir = os.path.join(CONFIG["OUTPUT_FRAME_DIR"], unique_id)
                os.makedirs(output_dir, exist_ok=True)
                
                # 调用MuseTalk处理
                self.processor.process_video_frame(
                    video_path=CONFIG["VIDEO_PATH"],
                    audio_path=wav_path,
                    output_dir=output_dir
                )
                
                # 更新日志为“完成”
                self.processing_log[original_filename]["status"] = "completed"
                self.processing_log[original_filename]["end_time"] = datetime.now().isoformat()
                save_processing_log(self.processing_log)
                print(f"✅ WAV文件处理完成：{original_filename}")
                
            except Exception as e:
                # 更新日志为“失败”
                if original_filename in self.processing_log:
                    self.processing_log[original_filename]["status"] = "failed"
                    self.processing_log[original_filename]["error"] = str(e)[:500]
                    self.processing_log[original_filename]["end_time"] = datetime.now().isoformat()
                    save_processing_log(self.processing_log)
                print(f"❌ 处理WAV文件 {original_filename} 出错：{str(e)}")
            
            finally:
                self.file_queue.task_done()

# -------------------------- 5. 前端播放页面生成（自动生成HTML） --------------------------
def generate_frontend_html() -> None:
    """生成前端播放页面（用于查看原视频+嘴形同步效果）"""
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>MuseTalk 嘴形同步播放</title>
    <style>
        .container { display: flex; gap: 20px; margin: 20px; }
        .video-container { flex: 1; }
        #syncCanvas { border: 1px solid #ccc; }
        #status { margin-top: 10px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-container">
            <h3>原视频</h3>
            <video id="originalVideo" width="640" controls autoplay loop>
                <source src="original.mp4" type="video/mp4">
                您的浏览器不支持视频播放
            </video>
        </div>
        <div class="video-container">
            <h3>嘴形同步视频</h3>
            <canvas id="syncCanvas" width="640" height="360"></canvas>
            <div id="status">等待WAV文件处理...</div>
        </div>
    </div>

    <script>
        // 配置（需与后端一致）
        const OUTPUT_FRAME_DIR = "./output_frames";
        const LOG_FILE = "processing_log.json";
        
        // DOM元素
        const originalVideo = document.getElementById("originalVideo");
        const syncCanvas = document.getElementById("syncCanvas");
        const ctx = syncCanvas.getContext("2d");
        const statusElem = document.getElementById("status");
        
        // 状态变量
        let currentTask = null;  // 当前处理中的任务（含unique_id和metadata）
        let frameIndex = 0;      // 当前要渲染的帧索引
        let renderInterval = null;  // 渲染定时器
        
        // 1. 轮询日志，获取最新处理任务
        async function checkLatestTask() {
            try {
                const response = await fetch(LOG_FILE + "?t=" + Date.now()); // 避免缓存
                if (!response.ok) throw new Error("日志获取失败");
                const log = await response.json();
                
                // 找到最新的“处理中”或“已完成”任务
                let latestTask = null;
                for (const [filename, info] of Object.entries(log)) {
                    if (info.status === "processing" || info.status === "completed") {
                        latestTask = info;
                    }
                }
                
                if (latestTask && latestTask.unique_id !== currentTask?.unique_id) {
                    currentTask = latestTask;
                    statusElem.textContent = `当前任务：${latestTask.status === "processing" ? "处理中" : "已完成"}（ID：${latestTask.unique_id}）`;
                    
                    // 如果任务已完成，加载元数据并开始渲染
                    if (latestTask.status === "completed") {
                        await loadMetadata(latestTask.output_dir);
                    }
                }
            } catch (e) {
                statusElem.textContent = `状态查询出错：${e.message}`;
            }
            setTimeout(checkLatestTask, 1000); // 每秒查询一次
        }
        
        // 2. 加载任务的元数据（FPS、延迟等）
        async function loadMetadata(outputDir) {
            try {
                const response = await fetch(`${outputDir}/metadata.json?t=` + Date.now());
                if (!response.ok) throw new Error("元数据获取失败");
                const metadata = await response.json();
                
                // 停止之前的渲染
                if (renderInterval) clearInterval(renderInterval);
                
                // 按视频FPS设置渲染间隔
                const frameDelay = 1000 / metadata.fps; // 每帧的毫秒数
                frameIndex = 0;
                
                // 启动渲染循环
                renderInterval = setInterval(() => {
                    if (!originalVideo.paused) {
                        renderFrame(outputDir, frameIndex);
                        frameIndex++;
                    }
                }, frameDelay);
            } catch (e) {
                statusElem.textContent = `元数据加载出错：${e.message}`;
            }
        }
        
        // 3. 渲染单帧（从后端加载生成的帧并绘制）
        async function renderFrame(outputDir, index) {
            const framePath = `${outputDir}/frame_${index.toString().padStart(6, "0")}.jpg`;
            try {
                const img = new Image();
                img.src = framePath + "?t=" + Date.now(); // 避免缓存
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, syncCanvas.width, syncCanvas.height);
                };
                img.onerror = () => {
                    // 帧未生成时，绘制原视频当前帧
                    ctx.drawImage(originalVideo, 0, 0, syncCanvas.width, syncCanvas.height);
                };
            } catch (e) {
                ctx.drawImage(originalVideo, 0, 0, syncCanvas.width, syncCanvas.height);
            }
        }
        
        // 初始化：启动任务查询
        checkLatestTask();
    </script>
</body>
</html>
'''
    # 保存HTML文件
    with open("musetalk_player.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 前端播放页面已生成：musetalk_player.html")

# -------------------------- 6. 主运行入口 --------------------------
if __name__ == "__main__":
    # 1. 初始化目录
    init_dirs()
    
    # 2. 生成前端播放页面
    generate_frontend_html()
    
    # 3. 初始化MuseTalk处理器
    processor = MuseTalkProcessor()
    
    # 4. 初始化监控器并启动
    monitor = WavMonitor(processor)
    monitor.start_monitor()
    monitor.start_workers()
    
    # 5. 保持主程序运行
    print("\n🎉 系统已全部启动！操作指引：")
    print(f"1. 将WAV文件放入监控目录：{CONFIG['WAV_MONITOR_DIR']}")
    print("2. 打开前端页面查看效果：musetalk_player.html")
    print("3. 处理日志查看：processing_log.json")
    print("\n按 Ctrl+C 退出系统...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        save_processing_log(monitor.processing_log)
        print("\n👋 系统退出，日志已保存")
        sys.exit(0)