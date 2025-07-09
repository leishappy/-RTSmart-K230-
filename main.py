from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d

import ujson
from media.media import *
import _thread
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aidemo
import random
import gc
import sys
import os
from media.media import *   #导入media模块，用于初始化vb buffer
from media.pyaudio import * #导入pyaudio模块，用于采集和播放音频
import media.wave as wave   #导入wav模块，用于保存和加载wav音频文件

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def exit_check():
    try:
        os.exitpoint()
    except KeyboardInterrupt as e:
        print("user stop: ", e)
        return True
    return False

def play_audio(filename):
        wf = wave.open('/sdcard/1.wav', 'rb')#打开wav文件
        CHUNK = int(44100/25)#设置音频chunck
        FORMAT = paInt16 #设置音频采样精度
        CHANNELS = 2 #设置音频声道数
        RATE = 44100 #设置音频采样率
        p = PyAudio()
        p.initialize(CHUNK) #初始化PyAudio对象
#        MediaManager.init()    #vb buffer初始化

        #创建音频输出流，设置的音频参数均为wave中获取到的参数
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,frames_per_buffer=CHUNK)

        data = wf.read_frames(CHUNK)#从wav文件中读取数一帧数据

        while data:
            stream.write(data)  #将帧数据写入到音频输出流中
            data = wf.read_frames(CHUNK) #从wav文件中读取数一帧
        stream.stop_stream() #停止音频输出流
        stream.close()#关闭音频输出流
        p.terminate()#释放音频对象
        wf.close()#关闭wav文件
#        MediaManager.deinit() #释放vb buffer
def play_audio_thread():
    try:
        wf = wave.open('/sdcard/1.wav', 'rb')#打开wav文件
        CHUNK = int(44100/25)#设置音频chunck
        FORMAT = paInt16 #设置音频采样精度
        CHANNELS = 2 #设置音频声道数
        RATE = 44100 #设置音频采样率
        p = PyAudio()
        p.initialize(CHUNK) #初始化PyAudio对象
        MediaManager.init()    #vb buffer初始化

        #创建音频输出流，设置的音频参数均为wave中获取到的参数
        stream = pl.p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,frames_per_buffer=CHUNK)

        data = wf.read_frames(CHUNK)#从wav文件中读取数一帧数据

        while data:
            stream.write(data)  #将帧数据写入到音频输出流中
            data = wf.read_frames(CHUNK) #从wav文件中读取数一帧数据
            if exit_check():
                break
    except BaseException as e:
            print(f"Exception {e}")
    finally:
        stream.stop_stream() #停止音频输出流
        stream.close()#关闭音频输出流
        p.terminate()#释放音频对象
        wf.close()#关闭wav文件
        MediaManager.deinit() #释放vb buffer


class FatigueDetector:
    def __init__(self):
        # 初始化疲劳检测参数
        self.eye_aspect_ratio_threshold = 0.3  # 眼睛纵横比阈值
        self.mouth_aspect_ratio_threshold = 0.3  # 嘴巴纵横比阈值
        self.blink_counter = 0  # 眨眼计数器
        self.yawn_counter = 0  # 打哈欠计数器
        self.nod_counter = 0  # 点头计数器
        # 时间窗口参数（单位：帧数）

        self.time_window = 30  # 3秒窗口（假设30FPS）
        self.ear_history = []  # 眼睛状态历史记录
        self.mar_history = []  # 嘴巴状态历史记录
        self.pitch_history = []  # 俯仰角历史记录
        self.current_ear = 0.0  # 存储当前EAR值
        self.current_mar = 0.0  # 存储当前MAR值
        self.fatigue_status = "Normal"  # 疲劳状态
        self.yawn_count = 0  # 打哈欠次数计数器
        self.blink_count = 0  # 眨眼次数计数器
        self.nod_count = 0  # 点头次数计数器
        self.fatigue_level = 0  # 疲劳等级：0-正常，1-疑似疲劳，2-疲劳
        self.frame_count = 0  # 帧计数器
        self.last_yawn_frame = 0  # 上次打哈欠的帧数
        self.last_blink_frame = 0  # 上次眨眼的帧数
        self.last_nod_frame = 0  # 上次点头的帧数

    def calculate_mouth_aspect_ratio(self, landmarks):
        """计算嘴巴纵横比 (MAR)"""
        # 定义嘴巴关键点索引
        #mouth_indices = [48, 49, 50, 51, 52, 53, 54, 55]
        mouth_indices = [65, 54, 60, 57, 69, 70, 62, 66]
        # 提取嘴巴坐标
        mouth = np.array([
            (landmarks[2*i], landmarks[2*i+1])
            for i in mouth_indices
        ])
        horizontal = np.linalg.norm(mouth[0] - mouth[4])
        vertical = np.linalg.norm(mouth[2] - mouth[6])
        self.current_mar = vertical / horizontal
        return self.current_mar

    def calculate_eye_aspect_ratio(self, landmarks):
        """计算眼睛纵横比 (EAR)"""
        # 定义眼睛关键点索引
        left_eye_indices = [35, 36, 33, 37, 39, 42, 40, 41]
        right_eye_indices = [89, 90, 87, 91, 93, 96, 94, 95]
        # 提取左眼坐标
        left_eye=np.array([(landmarks[2*i], landmarks[2*i+1]) for i in left_eye_indices])
        # 提取右眼坐标
        right_eye = np.array([(landmarks[2*i], landmarks[2*i+1]) for i in right_eye_indices])
        # 计算纵横比
        left_ear = self._calculate_aspect_ratio(left_eye)
        right_ear = self._calculate_aspect_ratio(right_eye)
        self.current_ear = (left_ear + right_ear) / 2.0
        return self.current_ear

    def _calculate_aspect_ratio(self, points):
        """通用纵横比计算"""
        vertical = np.linalg.norm(points[2] - points[6])
        horizontal = np.linalg.norm(points[0] - points[4])
        return vertical / horizontal

    def detect_head_pose(self, landmarks):
        """头部姿态估计（俯仰角）"""
        # 使用鼻尖和下巴点计算俯仰角
        nose_x = landmarks[27 * 2]     # 鼻尖x坐标
        nose_y = landmarks[27 * 2+1]   # 鼻尖y坐标
        chin_x = landmarks[8 * 2]      # 下巴x坐标
        chin_y = landmarks[8 * 2+1]    # 下巴y坐标
        # 计算俯仰角（单位：度）
        angle = np.degrees(np.arctan2(chin_y - nose_y, chin_x - nose_x))
        return angle

    def update_fatigue_status(self, landmarks):
        """更新疲劳状态"""
        # 更新帧计数器
        self.frame_count += 1

        # 计算当前帧特征
        ear = self.calculate_eye_aspect_ratio(landmarks)
        mar = self.calculate_mouth_aspect_ratio(landmarks)
        pitch = self.detect_head_pose(landmarks)

        # 记录历史数据
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.pitch_history.append(pitch)

        # 维护时间窗口
        if len(self.ear_history) > self.time_window:
            self.ear_history.pop(0)
            self.mar_history.pop(0)
            self.pitch_history.pop(0)

        # 判断眨眼条件（连续3帧低于阈值）
        blink_condition = False
        if len(self.ear_history) >= 3:
            blink_condition = all(e < self.eye_aspect_ratio_threshold for e in self.ear_history[-3:])

        # 判断点头条件（俯仰角变化超过15度）
        nod_condition = False
        if len(self.pitch_history) >= 2:
            pitch_change = abs(self.pitch_history[-1] - self.pitch_history[-2])
            nod_condition = pitch_change > 15

        # 判断打哈欠条件（嘴巴纵横比高于阈值）
        yawn_condition = mar > self.mouth_aspect_ratio_threshold

        # 检测眨眼
        if blink_condition:
            # 防止连续帧重复计数
            if self.frame_count - self.last_blink_frame > 10:  # 10帧内不重复计数
                self.blink_count += 1
                self.last_blink_frame = self.frame_count
                print(f"Blink detected! Count: {self.blink_count}")

        # 检测点头
        if nod_condition:
            # 防止连续帧重复计数
            if self.frame_count - self.last_nod_frame > 10:  # 10帧内不重复计数
                self.nod_count += 1
                self.last_nod_frame = self.frame_count
                print(f"Nod detected! Count: {self.nod_count}")

        # 检测打哈欠
        if yawn_condition:
            # 防止连续帧重复计数
            if self.frame_count - self.last_yawn_frame > 20:  # 10帧内不重复计数
                self.yawn_count += 1
                self.last_yawn_frame = self.frame_count
                print(f"Yawn detected! Count: {self.yawn_count}")

        # 根据行为次数判断疲劳等级
        if self.yawn_count*5+self.blink_count>= 10 :
            self.fatigue_level = 2  # 疲劳
            self.fatigue_status = "Fatigue Detected!"
        elif self.yawn_count*5+self.blink_count>= 5:
            self.fatigue_level = 1  # 疑似疲劳
            self.fatigue_status = "Suspected Fatigue"
        else:
            self.fatigue_level = 0
            self.fatigue_status = "Normal"

        # 每100帧重置计数器
        if self.frame_count % 100 == 0:
            self.yawn_count = 0
            self.blink_count = 0
            self.nod_count = 0
            self.fatigue_level = 0
            self.fatigue_status = "Normal"

        return self.fatigue_status, self.fatigue_level

    def get_current_values(self):
        """获取当前EAR和MAR值"""
        return self.current_ear, self.current_mar
    def play_prompt_audio(self):
            """循环播放提示音（当疲劳时）"""
            while self.fatigue_status == "Fatigue Detected!":
                try:
                    playsound(AUDIO_PROMPT_PATH, block=False)  # 非阻塞播放（需库支持）
                    time.sleep(5)  # 每5秒重复播放一次（可调整）
                except Exception as e:
                    print(f"播放音频失败: {e}")
                    break
# 自定义人脸检测任务类
class FaceDetApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,anchors,confidence_threshold=0.25,nms_threshold=0.3,rgb888p_size=[1280,720],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        # kmodel路径
        self.kmodel_path=kmodel_path
        # 检测模型输入分辨率
        self.model_input_size=model_input_size
        # 置信度阈值
        self.confidence_threshold=confidence_threshold
        # nms阈值
        self.nms_threshold=nms_threshold
        # 检测任务锚框
        self.anchors=anchors
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug模式
        self.debug_mode=debug_mode
        # 实例化Ai2d，用于实现模型预处理
        self.ai2d=Ai2d(debug_mode)
        # 设置Ai2d的输入输出格式和类型
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    # 配置预处理操作
    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            # 初始化ai2d预处理配置
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            # 设置padding预处理
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            # 设置resize预处理
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # 构建预处理流程
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    # 自定义后处理
    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            res = aidemo.face_det_post_process(self.confidence_threshold,self.nms_threshold,self.model_input_size[0],self.anchors,self.rgb888p_size,results)
            if len(res)==0:
                return res
            else:
                return res[0]

    # 计算padding参数
    def get_pad_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        # 计算最小的缩放比例，等比例缩放
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return [0,0,0,0,top, bottom, left, right]

# 自定义人脸关键点任务类
class FaceLandMarkApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        # kmodel路径
        self.kmodel_path=kmodel_path
        # 关键点模型输入分辨率
        self.model_input_size=model_input_size
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug模式
        self.debug_mode=debug_mode
        # 目标矩阵
        self.matrix_dst=None
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    # 配置预处理操作
    def config_preprocess(self,det,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            # 初始化ai2d预处理配置
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            # 计算目标矩阵，并获取仿射变换矩阵
            self.matrix_dst = self.get_affine_matrix(det)
            affine_matrix = [self.matrix_dst[0][0],self.matrix_dst[0][1],self.matrix_dst[0][2],
                             self.matrix_dst[1][0],self.matrix_dst[1][1],self.matrix_dst[1][2]]
            # 设置仿射变换预处理
            self.ai2d.affine(nn.interp_method.cv2_bilinear,0, 0, 127, 1,affine_matrix)
            # 构建预处理流程
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    # 自定义后处理
    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            pred=results[0]
            # 将人脸关键点输出变换模型输入
            half_input_len = self.model_input_size[0] // 2
            pred = pred.flatten()
            for i in range(len(pred)):
                pred[i] += (pred[i] + 1) * half_input_len
            # 获取仿射矩阵的逆矩阵
            matrix_dst_inv = aidemo.invert_affine_transform(self.matrix_dst)
            matrix_dst_inv = matrix_dst_inv.flatten()
            # 对每个关键点进行逆变换
            half_out_len = len(pred) // 2
            for kp_id in range(half_out_len):
                old_x = pred[kp_id * 2]
                old_y = pred[kp_id * 2 + 1]
                # 逆变换公式
                new_x = old_x * matrix_dst_inv[0] + old_y * matrix_dst_inv[1] + matrix_dst_inv[2]
                new_y = old_x * matrix_dst_inv[3] + old_y * matrix_dst_inv[4] + matrix_dst_inv[5]
                pred[kp_id * 2] = new_x
                pred[kp_id * 2 + 1] = new_y
            return pred

    def get_affine_matrix(self,bbox):
        # 获取仿射矩阵
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            # 从边界框提取坐标和尺寸
            x1, y1, w, h = map(lambda x: int(round(x, 0)), bbox[:4])
            # 计算缩放比例
            scale_ratio = (self.model_input_size[0]) / (max(w, h) * 1.5)
            # 计算边界框中心点
            cx = (x1 + w / 2) * scale_ratio
            cy = (y1 + h / 2) * scale_ratio
            # 计算模型输入空间的一半长度
            half_input_len = self.model_input_size[0] / 2
            # 创建仿射矩阵
            matrix_dst = np.zeros((2, 3), dtype=np.float)
            matrix_dst[0, 0] = scale_ratio
            matrix_dst[0, 1] = 0
            matrix_dst[0, 2] = half_input_len - cx
            matrix_dst[1, 0] = 0
            matrix_dst[1, 1] = scale_ratio
            matrix_dst[1, 2] = half_input_len - cy
            return matrix_dst

# 人脸标志解析
class FaceLandMark:
    def __init__(self,face_det_kmodel,face_landmark_kmodel,det_input_size,landmark_input_size,anchors,confidence_threshold=0.25,nms_threshold=0.3,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        # 人脸检测模型路径
        self.face_det_kmodel=face_det_kmodel
        # 人脸标志解析模型路径
        self.face_landmark_kmodel=face_landmark_kmodel
        # 人脸检测模型输入分辨率
        self.det_input_size=det_input_size
        # 人脸标志解析模型输入分辨率
        self.landmark_input_size=landmark_input_size
        # anchors
        self.anchors=anchors
        # 置信度阈值
        self.confidence_threshold=confidence_threshold
        # nms阈值
        self.nms_threshold=nms_threshold
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug_mode模式
        self.debug_mode=debug_mode

        self.fatigue_detector = FatigueDetector()
        self.t=0
        # 人脸关键点不同部位关键点列表
        self.dict_kp_seq = [
            [43, 44, 45, 47, 46, 50, 51, 49, 48],              # left_eyebrow
            [97, 98, 99, 100, 101, 105, 104, 103, 102],        # right_eyebrow
            [35, 36, 33, 37, 39, 42, 40, 41],                  # left_eye
            [89, 90, 87, 91, 93, 96, 94, 95],                  # right_eye
            [34, 88],                                          # pupil
            [72, 73, 74, 86],                                  # bridge_nose
            [77, 78, 79, 80, 85, 84, 83],                      # wing_nose
            [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64],  # out_lip
            [65, 54, 60, 57, 69, 70, 62, 66],                  # in_lip
            [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]  # basin

        ]

        # 人脸关键点不同部位颜色配置
        self.color_list_for_osd_kp = [
            (255, 0, 255, 0),
            (255, 0, 255, 0),
            (255, 255, 0, 255),
            (255, 255, 0, 255),
            (255, 255, 0, 0),
            (255, 255, 170, 0),
            (255, 255, 255, 0),
            (255, 0, 255, 255),
            (255, 255, 220, 50),
            (255, 30, 30, 255)
        ]
        # 人脸检测实例
        self.face_det=FaceDetApp(self.face_det_kmodel,model_input_size=self.det_input_size,anchors=self.anchors,confidence_threshold=self.confidence_threshold,nms_threshold=self.nms_threshold,rgb888p_size=self.rgb888p_size,display_size=self.display_size,debug_mode=0)
        # 人脸标志解析实例
        self.face_landmark=FaceLandMarkApp(self.face_landmark_kmodel,model_input_size=self.landmark_input_size,rgb888p_size=self.rgb888p_size,display_size=self.display_size)
        # 配置人脸检测的预处理
        self.face_det.config_preprocess()

    # run函数
    def run(self,input_np):
        # 执行人脸检测
        det_boxes=self.face_det.run(input_np)
        landmark_res=[]
        for det_box in det_boxes:
            # 对每一个检测到的人脸解析关键部位
            self.face_landmark.config_preprocess(det_box)
            res=self.face_landmark.run(input_np)
            landmark_res.append(res)
        return det_boxes,landmark_res

    # 绘制人脸解析效果
    def draw_result(self,pl,dets,landmark_res):
        pl.osd_img.clear()
        if dets:
            width = (self.display_size[0] // 8) * 8
            height = self.display_size[1]
            draw_img_np = np.zeros((self.display_size[1],(self.display_size[0]//8)*8,4),dtype=np.uint8)
            draw_img = image.Image(self.display_size[0], (self.display_size[1]//8)*8, image.ARGB8888, alloc=image.ALLOC_REF,data = draw_img_np)
            for pred in landmark_res:
                # 获取单个人脸框对应的人脸关键点
                for sub_part_index in range(len(self.dict_kp_seq)):
                    # 构建人脸某个区域关键点集
                    sub_part = self.dict_kp_seq[sub_part_index]
                    face_sub_part_point_set = []
                    for kp_index in range(len(sub_part)):
                        real_kp_index = sub_part[kp_index]
                        x, y = pred[real_kp_index * 2], pred[real_kp_index * 2 + 1]
                        x = int(x * self.display_size[0] // self.rgb888p_size[0])
                        y = int(y * self.display_size[1] // self.rgb888p_size[1])
                        face_sub_part_point_set.append((x, y))
                    # 画人脸不同区域的轮廓
                    if sub_part_index in (9, 6):
                        color = np.array(self.color_list_for_osd_kp[sub_part_index],dtype = np.uint8)
                        face_sub_part_point_set = np.array(face_sub_part_point_set)
                        aidemo.polylines(draw_img_np, face_sub_part_point_set,False,color,5,8,0)
                    elif sub_part_index == 4:
                        color = self.color_list_for_osd_kp[sub_part_index]
                        for kp in face_sub_part_point_set:
                            x,y = kp[0],kp[1]
                            draw_img.draw_circle(x,y ,2, color, 1)
                    else:
                        color = np.array(self.color_list_for_osd_kp[sub_part_index],dtype = np.uint8)
                        face_sub_part_point_set = np.array(face_sub_part_point_set)
                        aidemo.contours(draw_img_np, face_sub_part_point_set,-1,color,2,8)

            # 更新疲劳状态
            if landmark_res:
                status, level = self.fatigue_detector.update_fatigue_status(landmark_res[0])

                # 设置状态文本和颜色
            if level == 0:  # 正常
                status_color = (0, 255, 0)  # 绿色
                self.t = 1
            elif level == 1:  # 疑似疲劳
                status_color = (255, 255, 0)  # 黄色
                self.t = 1
            else:  # 疲劳
                status_color = (255, 0, 0)  # 红色
                if self.t==1:
                    _thread.start_new_thread(play_audio,('/sdcard/1.wav', ))
                    self.t = 0



            # 在左上角显示疲劳状态
            draw_img.draw_string(10, 10, status, color=status_color, scale=3)

            # 在右上角显示关键指标
            info_x = width - 300  # 右上角位置
            ear, mar = self.fatigue_detector.get_current_values()
            draw_img.draw_string(info_x, 10, f"EAR: {ear:.3f}", color=(255, 255, 0), scale=2)
            draw_img.draw_string(info_x, 40, f"MAR: {mar:.3f}", color=(255, 255, 0), scale=2)
            draw_img.draw_string(info_x, 70, f"Yawn: {self.fatigue_detector.yawn_count}", color=(255, 255, 0), scale=2)
            draw_img.draw_string(info_x, 100, f"Blink: {self.fatigue_detector.blink_count}", color=(255, 255, 0), scale=2)
            draw_img.draw_string(info_x, 130, f"Tired: {self.fatigue_detector.blink_count+5*self.fatigue_detector.blink_count}", color=(255, 255, 0), scale=2)
            #draw_img.draw_string(info_x, 130, f"Nod: {self.fatigue_detector.nod_count}", color=(255, 255, 0), scale=2)
            # 显示行为计数目标
            #draw_img.draw_string(10, 50, f"Blink/Nod Target: 5", color=(0, 255, 255), scale=2)

            pl.osd_img.copy_from(draw_img)


if __name__=="__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("audio sample start")
#    _thread.start_new_thread(play_audio,('/sdcard/output.wav', ))

    wf = wave.open('/sdcard/1.wav', 'rb')#打开wav文件
    CHUNK = int(44100/25)#设置音频chunck
    FORMAT = paInt16 #设置音频采样精度
    CHANNELS = 2 #设置音频声道数
    RATE = 44100 #设置音频采样率
    p = PyAudio()
    p.initialize(CHUNK) #初始化PyAudio对象
    
    audio_file = '/sdcard/1.wav'
    display_mode="NT35516"
    display_size=[960,540]


    face_det_kmodel_path="/sdcard/examples/kmodel/face_detection_320.kmodel"
    # 人脸关键标志模型路径
    face_landmark_kmodel_path="/sdcard/examples/kmodel/face_landmark.kmodel"
    # 其它参数
    anchors_path="/sdcard/examples/utils/prior_data_320.bin"
    rgb888p_size=[1920,1080]
    face_det_input_size=[320,320]
    face_landmark_input_size=[192,192]
    confidence_threshold=0.5
    nms_threshold=0.2
    anchor_len=4200
    det_dim=4
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len,det_dim))

    # 初始化PipeLine
    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    pl.create()
    flm=FaceLandMark(face_det_kmodel_path,face_landmark_kmodel_path,det_input_size=face_det_input_size,landmark_input_size=face_landmark_input_size,anchors=anchors,confidence_threshold=confidence_threshold,nms_threshold=nms_threshold,rgb888p_size=rgb888p_size,display_size=display_size)

    clock = time.clock()
    while True:
        clock.tick()
        img=pl.get_frame()                          # 获取当前帧
        det_boxes,landmark_res=flm.run(img)         # 推理当前帧
        flm.draw_result(pl,det_boxes,landmark_res)  # 绘制推理结果
        pl.show_image()                             # 展示推理效果
        gc.collect()
