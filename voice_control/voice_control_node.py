#!/usr/bin/env python3
import os
import time
import threading
import pyaudio
import numpy as np
import librosa
import websocket
import json
import base64
import hashlib
import hmac
from urllib.parse import urlencode
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
import ssl
import wave
from pynput import keyboard 

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor

# --- [讯飞 API 参数 和 AudioDeviceManager 类] ---
XFYUN_APPID = os.getenv('XFYUN_APPID')
XFYUN_APIKEY = os.getenv('XFYUN_APIKEY')
XFYUN_APISECRET = os.getenv('XFYUN_APISECRET')
RATE_TARGET = 16000
CHANNELS_TARGET = 1
FORMAT_TARGET = pyaudio.paInt16
CHUNK_SIZE = 1280

class AudioDeviceManager:

    def __init__(self, logger=None):
        self.logger = logger
        self.p = pyaudio.PyAudio()
        self.device_index = None
        self.input_rate = None
        self.input_channels = None
        self.stream = None
        self.resample_required = False
        
    def log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] [AudioDevice] {message}")
    
    def select_device(self, preferred_index=None):
        try:
            default_device = self.p.get_default_input_device_info()
            self.log(f"默认音频设备: {default_device['name']} (索引: {default_device['index']})")
            
            if preferred_index is not None and preferred_index >= 0:
                self.device_index = preferred_index
                device_info = self.p.get_device_info_by_index(preferred_index)
                self.log(f"使用参数指定的设备: {device_info['name']} (索引: {preferred_index})")
            else:
                self.device_index = default_device['index']
                device_info = default_device
                self.log("使用系统默认音频设备")
            
            self.input_rate = int(device_info['defaultSampleRate'])
            self.input_channels = int(device_info['maxInputChannels'])
            self.log(f"设备原生参数: {self.input_rate}Hz, {self.input_channels}通道")
            
            if self.check_device_support(self.device_index, RATE_TARGET, CHANNELS_TARGET, FORMAT_TARGET):
                self.input_rate = RATE_TARGET
                self.input_channels = CHANNELS_TARGET
                self.log("设备支持目标参数，无需重采样")
                self.resample_required = False
            else:
                self.log(f"设备不支持目标参数，需要重采样到{RATE_TARGET}Hz单声道")
                self.resample_required = True
            
            return True
        except Exception as e:
            self.log(f"设备检测失败: {str(e)}", 'error')
            return False
    
    def check_device_support(self, device_index, rate, channels, format):
        try:
            test_stream = self.p.open(
                format=format, channels=channels, rate=rate, input=True,
                input_device_index=device_index, frames_per_buffer=CHUNK_SIZE
            )
            test_stream.stop_stream()
            test_stream.close()
            return True
        except Exception:
            return False
    
    def open_stream(self):
        try:
            self.stream = self.p.open(
                format=FORMAT_TARGET, channels=self.input_channels, rate=self.input_rate,
                input=True, input_device_index=self.device_index,
                frames_per_buffer=CHUNK_SIZE, start=False
            )
            actual_device = self.p.get_device_info_by_index(self.device_index)
            self.log(f"成功打开音频设备: {actual_device['name']}")
            self.stream.start_stream()
            return True
        except Exception as e:
            self.log(f"打开音频流失败: {str(e)}", 'error')
            return False
    
    def get_audio_info(self):
        return {
            'index': self.device_index, 'rate': self.input_rate,
            'channels': self.input_channels, 'resample_required': self.resample_required
        }
    
    def resample_audio(self, data):
        if not self.resample_required:
            return data
        
        audio = np.frombuffer(data, dtype=np.int16)
        
        if self.input_channels > 1:
            audio = audio.reshape(-1, self.input_channels).mean(axis=1).astype(np.int16)
        
        if self.input_rate != RATE_TARGET:
            try:
                audio_float = audio.astype(np.float32) / 32768.0
                resampled_float = librosa.resample(audio_float, orig_sr=self.input_rate, target_sr=RATE_TARGET)
                return (resampled_float * 32768).astype(np.int16).tobytes()
            except Exception as e:
                self.log(f"重采样失败: {str(e)}", 'error')
                return audio.tobytes()
        
        return audio.tobytes()
    
    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.log("音频资源已释放")

class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("语音控制节点启动中...")
        self.stop_event = threading.Event()

        self.audio_mgr = AudioDeviceManager(logger=self.get_logger())
        
        self.declare_parameter('device_index', -1, ParameterDescriptor(description='麦克风索引 (-1=自动)'))
        param_index = self.get_parameter('device_index').get_parameter_value().integer_value

        if not self.audio_mgr.select_device(param_index) or not self.audio_mgr.open_stream():
            self.get_logger().error("音频设备初始化失败，节点将退出")
            self.create_timer(1.0, lambda: self.destroy_node())
            return

        self.get_logger().info(f"🎤 音频设备: {self.audio_mgr.get_audio_info()}")
        
      
        self.is_recording = False
        self.audio_buffer = []
        self.recording_lock = threading.Lock()

        self.commands = {
            "前进": (0.2, 0.0), "向前": (0.2, 0.0), "后退": (-0.2, 0.0),
            "左转": (0.0, 0.5), "右转": (0.0, -0.5), "停": (0.0, 0.0), 
            "停止": (0.0, 0.0), "原地左转": (0.0, 1.0), "原地右转": (0.0, -1.0),
            "加速": (0.4, 0.0), "减速": (0.1, 0.0)
        }

        
        self.audio_capture_thread = threading.Thread(target=self.capture_audio_loop, daemon=True)
        self.audio_capture_thread.start()
        
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.get_logger().info("="*60)
        self.get_logger().info("🚀 按键触发模式已就绪 🚀")
        self.get_logger().info("   - 按下 [空格键] 开始/停止录音")
        self.get_logger().info("   - 按下 [q] 键退出程序")
        self.get_logger().info("="*60)

    def on_key_press(self, key):
        """键盘按键回调函数"""
        try:
            # 退出键
            if key.char == 'q':
                self.get_logger().info("检测到退出键 [q]，正在关闭...")
                self.stop_event.set()
                # 使用Timer确保所有线程都能安全退出
                self.create_timer(0.1, self.destroy_node)
                return False # 停止监听器
        except AttributeError:
            # 特殊按键，空格键
            if key == keyboard.Key.space:
                with self.recording_lock:
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        # 开始录音
                        self.audio_buffer.clear()
                        self.get_logger().info("\n▶️  录音开始... (再次按下空格键结束)")
                    else:
                        # 停止录音并处理
                        self.get_logger().info("⏹️  录音结束，正在处理...")
                        if self.audio_buffer:
                            # 将音频数据复制出来，防止多线程问题
                            recorded_data = b''.join(self.audio_buffer)
                            # 在新线程中处理，避免阻塞键盘监听
                            threading.Thread(target=self.process_audio, args=(recorded_data,)).start()
                        else:
                            self.get_logger().warn("录音内容为空。")
        return True

    def capture_audio_loop(self):
        """一个专门的线程，只负责从麦克风读取数据并根据状态存入缓冲区"""
        while not self.stop_event.is_set():
            raw_data = self.audio_mgr.stream.read(CHUNK_SIZE, exception_on_overflow=False)
            with self.recording_lock:
                if self.is_recording:
                    processed_data = self.audio_mgr.resample_audio(raw_data)
                    self.audio_buffer.append(processed_data)
            # 短暂休眠，避免100% CPU占用
            time.sleep(0.01)

    def process_audio(self, audio_data):
        """一次性处理完整的音频数据"""
        ws_param = self.create_ws_param()
        if not ws_param: return
        
        ws_url = ws_param.create_url()
        try:
            ws = websocket.create_connection(ws_url, sslopt={"cert_reqs": ssl.CERT_NONE})
            
            # 1. 发送开始帧
            ws.send(json.dumps({
                "common": ws_param.CommonArgs, "business": ws_param.BusinessArgs,
                "data": {"status": 0}
            }))
            
            # 2. 一次性发送所有音频数据
            ws.send(json.dumps({
                "data": {"status": 1, "audio": base64.b64encode(audio_data).decode('utf-8')}
            }))
            
            # 3. 发送结束帧
            ws.send(json.dumps({"data": {"status": 2}}))
            
            # 4. 接收最终结果
            final_result = ""
            while True:
                message = ws.recv()
                msg = json.loads(message)
                if msg.get("code") != 0:
                    self.get_logger().error(f"识别错误: {msg.get('message', '未知错误')}")
                    break
                
                data = msg.get("data", {})
                result = data.get("result", {})
                words = [w.get("w", '') for item in result.get("ws", []) for w in item.get("cw", [])]
                final_result += "".join(words)
                
                if data.get("status") == 2: # 收到最终结果
                    break
            
            ws.close()
            self.handle_final_result(final_result)
            
        except Exception as e:
            self.get_logger().error(f"处理音频时发生网络或协议错误: {e}")

    def handle_final_result(self, text):
        """处理最终的识别文本"""
        if not text:
            self.get_logger().warn("识别结果为空。")
            self.get_logger().info("\n请按 [空格键] 开始下一次录音，或按 [q] 退出。")
            return

        self.get_logger().info(f"识别结果: \"{text}\"")
        
        # 模糊匹配，找到第一个就执行
        for cmd, (linear_x, angular_z) in self.commands.items():
            if cmd in text:
                self.get_logger().info(f"✅ 执行指令: '{cmd}'")
                twist_msg = Twist()
                twist_msg.linear.x = float(linear_x)
                twist_msg.angular.z = float(angular_z)
                self.publisher_.publish(twist_msg)
                self.get_logger().info("\n请按 [空格键] 开始下一次录音，或按 [q] 退出。")
                return # 执行完第一个匹配的指令后就返回
        
        self.get_logger().warn(f"在 \"{text}\" 中未找到有效指令。")
        self.get_logger().info("\n请按 [空格键] 开始下一次录音，或按 [q] 退出。")


    def create_ws_param(self):
        """创建WebSocket认证URL和参数"""
        if not all([XFYUN_APPID, XFYUN_APIKEY, XFYUN_APISECRET]):
            self.get_logger().error("讯飞API凭证未完整设置，请检查环境变量。")
            return None

        class Ws_Param:
            def __init__(self, app_id, api_key, api_secret):
                self.APPID, self.APIKey, self.APISecret = app_id, api_key, api_secret
                self.CommonArgs = {"app_id": self.APPID}
                # 注意：这里不再需要vad_eos，因为我们手动结束
                self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin"}

            def create_url(self):
                url = 'wss://ws-api.xfyun.cn/v2/iat'
                now = datetime.now()
                date = format_date_time(mktime(now.timetuple()))
                signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/iat HTTP/1.1"
                signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
                signature = base64.b64encode(signature_sha).decode('utf-8')
                authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
                authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
                v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
                return url + '?' + urlencode(v)
        
        return Ws_Param(XFYUN_APPID, XFYUN_APIKEY, XFYUN_APISECRET)
    
    def destroy_node(self):
        """优雅地关闭节点和所有资源"""
        if not self.stop_event.is_set():
            self.get_logger().info("正在关闭语音控制节点...")
            self.stop_event.set()
        
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
        
        if hasattr(self, 'audio_capture_thread') and self.audio_capture_thread.is_alive():
            self.audio_capture_thread.join(timeout=1.0)
            
        if self.audio_mgr:
            self.audio_mgr.close()
            
        super().destroy_node()
        # 确保所有日志都能在shutdown前刷出
        time.sleep(0.5)
        self.get_logger().info("语音控制节点已关闭。")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceControlNode()
        # 由于我们使用键盘监听来控制生命周期， spin() 只是为了保持节点存活
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info("节点被用户中断 (Ctrl+C)")
    except Exception as e:
        if node: node.get_logger().fatal(f"节点遇到致命错误: {e}", exc_info=True)
    finally:
        # destroy_node 已经由按键 'q' 或 Ctrl+C 触发，这里确保万无一失
        if node and rclpy.ok() and not node.stop_event.is_set():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n程序已退出。")

if __name__ == '__main__':
    main()
