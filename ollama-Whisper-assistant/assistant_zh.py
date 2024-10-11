import sys # 标准库用于系统操作
import json # JSON处理
import wave # WAV文件操作
import time # 时间相关功能
import pyttsx3 # 文字转语音库
import torch
import requests # HTTP请求库
import soundfile # 音频文件处理库
import yaml # YAML文件解析库
import pygame # 游戏开发和多媒体库，这里可能用于图形界面
import pygame.locals
import numpy as np
import pyaudio # 音频I/O库
import whisper # OpenAI的Whisper语音识别模型
import logging # 用于日志记录
import threading # 多线程和队列操作
import queue # 用于创建一个先进先出（FIFO）的队列数据结构
             # 这个队列是线程安全的，这意味着它可以在多线程环境中安全地使用，而不会出现数据竞争或其他并发问题

# Configure logging 设置了基本的日志配置，级别为DEBUG，包含时间戳、日志级别和消息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

'''
颜色常量：定义了背景色、录音指示色和文本颜色。
尺寸常量：定义了录音指示器大小、字体大小、窗口宽高等。
文本显示相关常量：如最大显示文本长度。
'''
BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
REC_SIZE = 80
FONT_SIZE = 24
WIDTH = 320
HEIGHT = 240
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 32

# 定义了音频输入的格式、通道数1、采样率16kHz、块大小等
INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16 # 使用16位整数格式
INPUT_CHANNELS = 1 # 单声道录音
INPUT_RATE = 16000 # 采样率为16kHz
INPUT_CHUNK = 1024 # 每次读取1024个样本
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'} # 用于与Ollama API通信的HTTP头
INPUT_CONFIG_PATH ="assistant_zh.yaml" # 配置文件路径

class Assistant:
    def __init__(self):
        logging.info("正在初始化 Assistant_zh...")
        self.config = self.init_config() # 调用类中方法得到配置信息，并存入数据容器（config对象）

        # 设置pygame窗口和图标
        programIcon = pygame.image.load('assistant.png') # 读个图标进来
        self.clock = pygame.time.Clock() # 设置时钟作用是 控制帧率：限制游戏循环的执行速度
        pygame.display.set_icon(programIcon) # 显示图标
        pygame.display.set_caption("Assistant_zh") # 显示标题

        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32) # 设置显示窗口
        self.font = pygame.font.SysFont(None, FONT_SIZE) # 设置字体显示

        # pyaudio库用于查询可用的音频设备，打开音频流进行录音或播放，设置音频参数（如采样率、通道数、格式等），管理音频缓冲区
        self.audio = pyaudio.PyAudio() # 没有它就无法捕获mic传入的音频

        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', self.tts.getProperty('rate') - 20)


        # 这段代码尝试打开一个音频流，然后立即关闭它。这是为了检查是否能成功访问音频设备
        # 如果失败，它会记录错误并调用wait_exit方法
        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"尝试载入语音出现错误: {str(e)}")
            self.wait_exit() # 如果打开失败就调用这个


        self.display_message(self.config.messages.loadingModel) # 在屏幕打印一条消息
        self.model = whisper.load_model(self.config.whisperRecognition.modelPath) # 加载whisper模型
        self.context = [] # 新建一个存放会话的列表

        self.text_to_speech(self.config.conversation.greeting) # 文转音,并且还要自动读出来
        time.sleep(0.5)
        self.display_message(self.config.messages.pressSpace) # 显示"请按键说话"的提示文字

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput) # 调用类自己的方法显示一条错误消息"no sound"
            self.clock.tick(60) # 控制这个while循环的速度 减少CPU使用率
            # 关注QUIT事件，这通常是用户试图关闭窗口时触发的 如果检测到QUIT事件，调用self.shutdown()方法来安全地关闭程序
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        logging.info("Assistant_zh关闭中...") # 日子记录
        self.audio.terminate() # 它会关闭所有活跃的音频流，并释放PyAudio分配的所有资源
        pygame.quit() # 这个调用会关闭所有Pygame模块，它会释放Pygame使用的所有资源，包括显示窗口、事件队列等
        sys.exit() # 退出终端

    def init_config(self):
        logging.info("初始化配置中...") #方法开始时记录了一条日志，表明配置初始化开始

        class Inst:# 这个空类Inst被用作一个简单的结构体，用于创建嵌套的配置对象
            pass

        # 打开并读取名为'assistant.yaml'的配置文件，使用yaml.safe_load解析YAML内容
        with open('assistant_zh.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)

        config = Inst() # 创建一个主配置对象
        config.messages = Inst() # 在主配置对象中创建一个messages子对象，并从YAML文件中读取相关消息设置
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

        config.conversation = Inst() # 创建一个conversation子对象，设置问候语
        config.conversation.greeting = configYaml["conversation"]["greeting"]

        config.ollama = Inst() # 创建一个ollama子对象，设置URL和模型名称
        config.ollama.url = configYaml["ollama"]["url"]
        config.ollama.model = configYaml["ollama"]["model"]

        config.whisperRecognition = Inst() # 创建一个whisperRecognition子对象，设置模型路径和语言
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]
        
        config.users = Inst()
        config.users.user = configYaml['users']['user']

        return config

    # 当开始录音时，画面变成一个大红圈
    def display_rec_start(self):
        logging.info("显示记录开始:")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_message(self, text):
        logging.info(f"显示消息: {text}") # 日志记录
        self.windowSurface.fill(BACK_COLOR) # 填充背景

        # 文本渲染与截断（超过32个字就斩段）
        label = self.font.render(text
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4] # [2:4] 切片获取矩形的w和h（前两个元素是 x 和 y 坐标）

        # 将文本绘制到窗口表面上 计算位置使文本居中显示
        # X：窗口w的一半减去文本w的一半
        # Y：窗口h的一半减去文本h的一半
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT/2 - size[1]/2))
        pygame.display.flip() # 更新显示


    # 方法接受一个可选参数 key，默认值是 pygame 的空格键常量
    # 返回类型注解表明这个方法会返回一个 NumPy 数组
    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:
        logging.info("正在从麦克风获取语音流")
        self.display_rec_start() # 当开始录音时，画面出现一个大红圆

        # 使用预定义的音频参数（格式、通道数、采样率、缓冲区大小）打开音频输入流
        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = [] # 创建一个空列表来存储捕获的音频帧

        while True:
            pygame.event.pump() # 它更新 Pygame 的内部状态，确保按键状态等信息是最新的
            pressed = pygame.key.get_pressed() # 这个函数返回一个表示所有键盘按键状态的列表，全是布尔值
            if pressed[key]: # 找到空格键的布尔值，如果有：
                data = stream.read(INPUT_CHUNK) # 从麦克风读取一小块音频数据
                frames.append(data) # 读取的数据被添加到 frames 列表中
            else:
                break # 如果指定的键没有被按下，循环就会终止

        stream.stop_stream() # 停止音频流
        stream.close() # 关闭音频流并释放资源

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)
        # 将所有音频帧连接成一个字节字符串。
        # 使用 np.frombuffer 将字节数据转换为 16 位整数数组。
        # 将整数数组转换为 32 位浮点数组。
        # 将数值范围从 [-32768, 32767] 缩放到 [-1.0, 1.0]。


    def speech_to_text(self, waveform):# 方法接受一个 waveform 参数，这是要转换为文本的音频波形数据
        logging.info("正在将语音转换为文字...")
        result_queue = queue.Queue()# 创建一个线程安全的队列，用于存储转录结果(这是非常重要的概念，尤其是在多线程编程中
                                    # 多个线程可以同时访问这个队列，而不会导致数据损坏
        def transcribe_speech(): # 下面的model就是whisper!
            try:
                logging.info("开始转换...")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available()) # language 参数指定了识别的语言
                logging.info("转换完成.")
                text = transcript["text"] # 从转录结果中提取文本
                print('\nMe:\n', text.strip()) # 打印用户的语音内容
                result_queue.put(text) # 将转录文本放入结果队列
            except Exception as e:
                logging.error(f"在转换时出现了一个错误: {str(e)}")
                result_queue.put("") # 如果发生错误，将空字符串放入结果队列

        transcription_thread = threading.Thread(target=transcribe_speech) # 创建一个新线程来运行 音转文 函数
        transcription_thread.start()# 立即启动这个线程
        transcription_thread.join()# 让主线程等待这个transcription线程完成

        return result_queue.get() # 从结果队列中获取转录文本并返回
        # 此程序块采用了 同步方法，确保在获取结果之前转录已经完成


    # 方法接受两个参数：prompt（要发送给模型的文本）和 responseCallback（一个回调函数，用于处理响应）
    # 这个回调函数就是text_to_speech，也就是从ollama收到结果时就直接文字转语音念出来了
    def ask_ollama(self, prompt, responseCallback):
        logging.info(f"正在放入提示语进入OllaMa模型: {prompt}")
        full_prompt = f"请以'{self.config.users.user},'作为开始，如果非必要，请用简短的方式回应以下问题：{prompt}" 
        # full_prompt = prompt
        self.contextSent = True

        # 创建一个字典，包含发送给 OLLaMa 的参数。
        # 指定模型、启用流式传输、包含上下文和提示
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt
        }
        
        try:
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    stream=True,
                                    timeout=30)  # Increase the timeout value
            # 使用 requests.post 发送 POST 请求到 OLLaMa 服务
            # 启用流式传输和设置超时
            response.raise_for_status() # 使用 raise_for_status() 检查请求是否成功

            full_response = ""
            for line in response.iter_lines():   # 逐行读取响应流
                body = json.loads(line)          # 解析每行 JSON 数据
                token = body.get('response', '')
                full_response += token           # 累积响应文本

                if 'error' in body:
                    logging.error(f"一个来自OLLaMa的错误: {body['error']}")
                    responseCallback("Error: " + body['error'])# 有读取时出现错误的话就读出错的提示语
                    return

                # 这就像你在等待朋友的消息
                # body.get('done', False) 你检查朋友是不是说"我说完了" (如果 'done' 键存在，返回其对应的值，否则返回False)
                # 'context' in body 你看看朋友有没有给你那个小纸条（上下文）
                # 如果朋友说完了，而且给了你小纸条。就把小纸条上的内容记下来，以便下次聊天时使用
                if body.get('done', False) and 'context' in body:
                    self.context = body['context'] # 记录小纸条
                    break

            responseCallback(full_response.strip()) # 直接文字转语音念出来

        except requests.exceptions.ReadTimeout as e:
            logging.error(f"在询问OllaMa时出现了超时错误: {str(e)}")
            responseCallback("对不起, 在询问OllaMa时出现了超时错误, 请重新尝试")
        except requests.exceptions.RequestException as e:
            logging.error(f"在询问OllaMa时出现了一个错误: {str(e)}")
            responseCallback("对不起, 在询问OllaMa时出现了一个错误, 请重新尝试.")


    # 这个方法实现了文本到语音的转换功能，并使用了多线程来避免阻塞主程序
    def text_to_speech(self, text): # 方法接受一个 text 参数，这是要转换为语音的文本
        logging.info(f"正在转换文字为语音: {text}") # 日志记录
        print('\nAI:\n', text.strip()) # strip() 方法用于移除文本首尾的空白字符

        def play_speech():
            try:
                logging.info("初始化 TTS 引擎")
                engine = pyttsx3.init() # 初始化 TTS 引擎
                
                # 获取当前语音速率，并将其降低 10 个单位，使ai的语音更慢更清晰
                # rate = engine.getProperty('rate') # 获取速率
                # engine.setProperty('rate', rate - 10) # 降低速率
                
                # 在开始语音转换前添加短暂延迟(可能是为了确保系统准备就绪)
                time.sleep(0.5)  # 可任意修改
                
                logging.info("转换文字为语音...")
                engine.say(text) # 将文本添加到语音引擎的播放队列
                engine.runAndWait() # 运行引擎并等待语音播放完成
                logging.info("语音转换完成.")
            except Exception as e:
                logging.error(f"转换文字为语音时出现了一个错误: {str(e)}")

        speech_thread = threading.Thread(target=play_speech) # 创建一个新线程来运行 play_speech 函数
        speech_thread.start() # 立即启动这个线程来处理单个的 文本转语音 任务
        # 非阻塞操作：使用线程确保语音播放不会阻塞主程序的执行
        # 此程序块 采用了异步方法，允许程序继续执行而不等待语音播放完成

        # 这个方法的目的是开始语音播放。它不需要等待语音播放完成就可以继续执行后续代码，所以不需要调用.join()
        # 不需要返回任何值(这里指文本)。它只是启动了语音播放过程
        # 通常希望语音播放是非阻塞的，这样程序可以继续执行其他任务（比如准备下一个响应），同时语音在后台播放

def main():
    logging.info("开始 Assistant_zh")
    pygame.init()

    ass = Assistant()

    push_to_talk_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:# 当有键按下去了
                if event.key == push_to_talk_key: # 当按下去的键是空格
                    logging.info("Push-to-talk key pressed")
                    speech = ass.waveform_from_mic(push_to_talk_key)

                    transcription = ass.speech_to_text(waveform=speech)

                    ass.ask_ollama(transcription, ass.text_to_speech)

                    time.sleep(1)
                    ass.display_message(ass.config.messages.pressSpace)

                elif event.key == quit_key:
                    logging.info("Quit key pressed")
                    ass.shutdown()


if __name__ == "__main__":
    main()
