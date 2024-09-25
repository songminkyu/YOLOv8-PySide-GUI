from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ultralytics.utils import DEFAULT_CFG

from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from core_kr import YoloPredictor

from pathlib import Path
from utils.rtsp_win import Window
import traceback
import json
import sys
import cv2
import os
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # 메인 창은 YOLO 인스턴스에 실행 신호를 보냅니다.
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # 기본 인터페이스 설정
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 둥근 투명 모서리
        self.setWindowFlags(Qt.FramelessWindowHint)     # 창 플래그 설정: 창 테두리 숨기기
        UIFuncitons.uiDefinitions(self)                 # 사용자 정의 인터페이스 정의

        # 초기 페이지
        self.rate = 0 # Delay(ms) 사이즈
        self.task = ''
        self.PageIndex = 1
        self.content.setCurrentIndex(self.PageIndex)
        self.pushButton_detect.clicked.connect(self.button_detect)
        self.pushButton_pose.clicked.connect(self.button_pose)
        self.pushButton_classify.clicked.connect(self.button_classify)
        self.pushButton_segment.clicked.connect(self.button_segment)
        self.pushButton_track.clicked.connect(self.button_track)

        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        self.src_home_button.clicked.connect(self.return_home)
        ####################################image or video####################################
        # 디스플레이 모듈 섀도우
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # YOLO-v8 스레드
        self.model_refresh_sencond = 2000
        overrides = {'batch': 1} # 배치사이즈 사전 설정 (기본값 16)
        self.yolo_predict = YoloPredictor(cfg=DEFAULT_CFG,overrides=overrides) # YOLO 인스턴스 생성
        self.select_model = self.model_box.currentText()              # 기본 모델
         
        self.yolo_thread = QThread()                                  # YOLO 스레드 생성
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video, 'img'))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video, 'img'))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))        
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))      
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))    
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x))) 
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        self.Qtimer_ModelBox = QTimer(self)     # 타이머: 2초마다 모델 파일 변경 사항을 모니터링.
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(self.model_refresh_sencond)

        # 모델 매개변수
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))    # iou 스핀 텍스트 박스
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))      # iou 슬라이더
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf 스핀 텍스트 박스
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))    # conf 슬라이더
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))# speed 스핀 텍스트 박스
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed 슬라이더
        # 프롬프트 창 초기화
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # 폴더 위치 선택
        self.src_file_button.clicked.connect(self.open_src_file)  # 로컬 파일 선택
        # 단일 파일
        self.src_img_button.clicked.connect(self.open_src_img)  # 로컬 파일 선택
        # 테스트 시작 버튼
        self.run_button.clicked.connect(self.run_or_continue)   # 일시중지/시작
        self.stop_button.clicked.connect(self.stop)             # 종료

        # 기타 기능 버튼
        self.save_res_button.toggled.connect(self.is_save_res)  # 사진 옵션 저장
        self.save_txt_button.toggled.connect(self.is_save_txt)  # 라벨 옵션 저장
        ####################################image or video####################################
        ####################################camera####################################
        # 캠 모듈 그림자 표시
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-cam 스레드
        overrides = {'batch': 1}  # 배치사이즈 사전 설정 (기본값 16)
        self.yolo_predict_cam = YoloPredictor(cfg=DEFAULT_CFG,overrides=overrides)  # YOLO 인스턴스 생성
        self.select_model_cam = self.model_box_cam.currentText()          # 기본 모델
        
        self.yolo_thread_cam = QThread()                                  # YOLO 스레드 생성
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.show_status(c))        
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))      
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))    
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c))) 
        self.yolo_predict_cam.yolo2main_progress.connect(self.progress_bar_cam.setValue(0))
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

        self.Qtimer_ModelBox_cam = QTimer(self)     # 타이머: 2초마다 모델 파일 변경을 모니터
        self.Qtimer_ModelBox_cam.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox_cam.start(self.model_refresh_sencond)

        # 캠 모델 매개변수
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)     
        self.iou_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_spinbox_cam'))    # iou 스핀 텍스트 박스
        self.iou_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_slider_cam'))      # iou 슬라이더
        self.conf_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_spinbox_cam'))  # conf 스핀 텍스트 박스
        self.conf_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_slider_cam'))    # conf 슬라이더
        self.speed_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_spinbox_cam'))# speed 스핀 텍스트 박스
        self.speed_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_slider_cam'))  # speed 슬라이더

        # 프롬프트 창 초기화
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
        self.Model_name_cam.setText(self.select_model_cam)
        
        # 탐지 위치 선택
        self.src_cam_button.clicked.connect(self.cam_button) # 카메라 선택
        
        # 테스트 시작 버튼
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)   # 일시중지/시작
        self.stop_button_cam.clicked.connect(self.cam_stop)             # 종료
        
        # 기타 기능 버튼
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)  # 사진 옵션 저장
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)  # 라벨 옵션 저장
        ####################################camera####################################
        ####################################rtsp####################################
        self.src_rtsp_button.clicked.connect(self.rtsp_button)
        ####################################rtsp####################################
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # 왼쪽 탐색 버튼
        # 초기화
        self.load_config()
        self.show_status("YOLOv8 감지 시스템 사용을 환영합니다. Mode를 선택하세요.")

    def switch_mode(self, task):
        self.task = task
        self.yolo_predict.task = task
        self.yolo_predict_cam.task = task
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # 오른쪽 상단의 설정 버튼

        # 모델 폴더 읽기
        self.pt_list = os.listdir(f'./models/{task.lower()}/')
        self.pt_list = [file for file in self.pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'./models/{task.lower()}/' + x))  # 파일 크기별로 정렬
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = f"./models/{task.lower()}/{self.select_model}"
        self.yolo_predict_cam.new_model_name = f"./models/{task.lower()}/{self.select_model_cam}"

        # Cam 모델 폴더 읽기
        self.pt_list_cam = os.listdir(f'./models/{task.lower()}/')
        self.pt_list_cam = [file for file in self.pt_list_cam if file.endswith(('.pt', 'onnx', 'engine'))]
        self.pt_list_cam.sort(key=lambda x: os.path.getsize(f'./models/{task.lower()}/' + x))  # 파일 크기별로 정렬
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status(f"현재 페이지：image or video 감지 페이지，Mode：{task}")

    def button_classify(self):
        self.switch_mode('Classify')

    def button_detect(self):
        self.switch_mode('Detect')

    def button_pose(self):
        self.switch_mode('Pose')

    def button_segment(self):
        self.switch_mode('Segment')

    def button_track(self):
        self.switch_mode('Track')

    def return_home(self):
        # 메인 페이지로 돌아가서 상태 및 버튼 재설정
        self.PageIndex = 1
        self.content.setCurrentIndex(1)
        self.yolo_predict_cam.source = ''
        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
        self.settings_button.setEnabled(False)
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
        self.cam_stop()
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()
        self.stop()
        self.show_status("YOLOv8 감지 시스템 사용을 환영합니다. Mode를 선택하세요.")

    ####################################image or video####################################
    # 로컬 파일 선택
    def open_src_file(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        # 작업 유형에 따른 상태 정보 표시
        mode_status = {
            'Classify': "현재 페이지：image or video 감지 페이지，Mode：Classify",
            'Detect': "현재 페이지：image or video 감지 페이지，Mode：Detect",
            'Pose': "현재 페이지：image or video 감지 페이지，Mode：Pose",
            'Segment': "현재 페이지：image or video 감지 페이지，Mode：Segment",
            'Track': "현재 페이지：image or video 감지 페이지，Mode：Track"
        }
        
        # 리소스를 절약 하기위한 Cam Thread 종료
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        # 작업 유형에 따른 상태 정보 표시
        if self.task in mode_status:
            self.show_status(mode_status[self.task])
        
        # 설정 파일 경로 설정
        config_file = 'config/fold.json'
        
        # 구성 파일의 내용을 읽고 마지막으로 열린 폴더의 경로를 가져옴. 하지만 폴더가 없으면 현재 작업 디렉터리 그대로 사용.
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config.get('open_fold', os.getcwd())
        
        # 사용자가 파일 대화 상자를 통해 폴더를 선택하도록 허용 (OpenFolderDialog을 의미함.)
        FolderPath = QFileDialog.getExistingDirectory(self, '폴더 선택', open_fold)
        
        # 사용자가 폴더를 선택하는 경우
        if FolderPath:
            FileFormat = [".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            if Foldername:
                # 선택한 아카이브의 경로를 yolo_predict의 Source로 설정합니다.
                self.yolo_predict.source = Foldername
                # 파일 로딩 상태 표시
                self.show_status('폴더 로드：{}'.format(os.path.basename(FolderPath)))
                # 설정 파일에서 마지막으로 열린 폴더 경로를 업데이트
                config['open_fold'] = os.path.dirname(FolderPath)
                
                # 업데이트 된 설정 파일을 파일에 다시 작성
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                
                # 감지 중지
                self.stop()
            else:
                self.show_status('폴더에 사진이 없네요...')
         
    # 로컬 파일 선택
    def open_src_img(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)
        # 작업 유형에 따라 다양한 상태 정보 표시
        mode_status = {
            'Classify': "현재 페이지：image or video 감지 페이지，Mode：Classify",
            'Detect': "현재 페이지：image or video 감지 페이지，Mode：Detect",
            'Pose': "현재 페이지：image or video 감지 페이지，Mode：Pose",
            'Segment': "현재 페이지：image or video 감지 페이지，Mode：Segment",
            'Track': "현재 페이지：image or video 감지 페이지，Mode：Track"
        }
        if self.task in mode_status:
            self.show_status(mode_status[self.task])
        
        # 리소스를 절약 하기위한 Cam Thread 종료
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()
            self.cam_stop()

        # 설정 파일 경로 설정
        config_file = 'config/fold.json'
        
        # 설정 파일 읽기
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        
        # 마지막으로 열린 폴더의 경로를 가져옴.
        open_fold = config.get('open_fold', os.getcwd())
        
        # 사용자가 파일 대화 상자를 통해 이미지 또는 비디오 파일을 선택할 수 있도록 허용
        if self.task == 'Track':
            title = 'Video'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv)"
        else:
            title = 'Video/image'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)"
        
        name, _ = QFileDialog.getOpenFileName(self, title, open_fold, filters)
        
        # 사용자가 아카이브를 선택한 경우
        if name:
            # 선택한 아카이브의 경로를 yolo_predict의 source로 설정합니다.
            self.yolo_predict.source = name
            
            # 파일 로딩 상태 표시
            self.show_status('파일 로드：{}'.format(os.path.basename(name)))
            
            # 설정 파일에서 마지막으로 열린 폴더 경로를 업데이트합니다.
            config['open_fold'] = os.path.dirname(name)
            
            # 업데이트된 설정 파일을 파일에 다시 씁니다.
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            
            # 감지 중지
            self.stop()
                
    # 메인 창에는 원본 사진과 감지 결과가 표시
    @staticmethod
    def show_image(img_src, label, flag, instance=None):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img_src, dtype=np.uint8), -1)

            # 원본 이미지의 높이, 너비, 채널 수를 가져옴
            ih, iw, _ = img_src.shape

            # 라벨의 너비와 높이를 가져옴
            w, h = label.geometry().width(), label.geometry().height()

            # 원래 데이터 비율을 유지하고 조정된 크기를 계산함
            if iw / w > ih / h:
                scal = w / iw
                nw, nh = w, int(scal * ih)
            else:
                scal = h / ih
                nw, nh = int(scal * iw), h

            # 이미지 크기를 조정하고 RGB 형식으로 변환
            frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

            # 이미지 데이터를 Qt 이미지 객체로 변환하여 라벨에 표시
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 예외를 처리하고 오류 메시지를 출력
            traceback.print_exc()
            print(f"Error: {e}")
            if instance is not None:
                instance.show_status('%s' % e)

    # 제어 시작/일시 중지 감지
    def run_or_continue(self):
        def handle_no_source():
            self.show_status('감지를 시작하기 전에 이미지 또는 비디오 소스를 선택하십시오...')
            self.run_button.setChecked(False)

        def start_detection():
            self.save_txt_button.setEnabled(False)  # 감지 시작 후 확인 및 저장 비활성화
            self.save_res_button.setEnabled(False)
            self.show_status('감지 중 입니다...')
            self.yolo_predict.continue_dtc = True  # YOLO 일시중지 여부 제어

            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict.continue_dtc = False
            self.show_status("감지가 일시 중지 되었습니다...")
            self.run_button.setChecked(False)  # 정지 버튼

        if not self.yolo_predict.source:
            handle_no_source()
        else:
            self.yolo_predict.stop_dtc = False

            if self.run_button.isChecked():  # 시작버튼이 체크되어 있을경우
                start_detection()
            else:  # 시작 버튼을 체크하지 않으면 감지가 중단된 것.
                pause_detection()

    # 테스트 결과 저장 버튼 -- 사진/비디오
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # 이미지 실행 결과가 저장되지 않는다는 메시지가 표시됨.
            self.show_status('NOTE：실행 중인 이미지 결과는 저장되지 않습니다.')
            
            # YOLO 인스턴스의 저장된 결과 플래그를 False로 설정합니다.
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # 이미지 실행 결과가 저장된다는 메시지를 표시
            self.show_status('NOTE：이미지 실행 결과가 저장됩니다.')
            
            # YOLO 인스턴스의 저장된 결과 플래그를 True로 설정.
            self.yolo_predict.save_res = True

    # 테스트 결과 저장 버튼 -- 라벨 (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # 라벨 결과가 저장되지 않는다는 메시지를 표시.
            self.show_status('NOTE：Label 결과는 저장되지 않습니다.')

            # YOLO 인스턴스의 저장된 레이블 플래그를 False로 설정.
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # 라벨 결과가 저장된다는 메시지를 표시.
            self.show_status('NOTE：Label 결과가 저장됩니다.')

            # YOLO 인스턴스의 save-label 플래그를 True로 설정.
            self.yolo_predict.save_txt = True

    # 종료 버튼 및 관련 상태 처리
    def stop(self):
        def stop_yolo_thread():
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 쓰레드 종료
            self.yolo_predict.stop_dtc = True

        def reset_ui_elements():
            self.run_button.setChecked(False)
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.pre_video.clear()
            self.res_video.clear()
            self.progress_bar.setValue(0)
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

        stop_yolo_thread()
        self.show_status('감지가 종료되었습니다.')
        reset_ui_elements()

    # 감지 매개변수 변경
    def change_val(self, x, flag):
        def update_iou():
            value = x / 100
            self.iou_spinbox.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict.iou_thres = value

        def update_conf():
            value = x / 100
            self.conf_spinbox.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict.conf_thres = value

        def update_speed():
            self.speed_spinbox.setValue(x)
            self.show_status(f'Delay: {x} ms')
            self.yolo_predict.speed_thres = x  # 밀리초

        update_actions = {
            'iou_spinbox': lambda: self.iou_slider.setValue(int(x * 100)),
            'iou_slider': update_iou,
            'conf_spinbox': lambda: self.conf_slider.setValue(int(x * 100)),
            'conf_slider': update_conf,
            'speed_spinbox': lambda: self.speed_slider.setValue(x),
            'speed_slider': update_speed
        }

        if flag in update_actions:
            update_actions[flag]()
    
    # 모델 변경
    def change_model(self, x):
        # 현재 선택된 모델명을 가져옴.
        self.select_model = self.model_box.currentText()
        
        # 작업에 따라 모델 경로의 접두사를 설정.
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # YOLO 인스턴스의 새 모델 이름 설정
        self.yolo_predict.new_model_name = f"{model_prefix}/{self.select_model}"

        # 모델이 변경되었다는 메시지 표시
        self.show_status(f'Change Model: {self.select_model}')
        
        # 인터페이스에 새 모델 이름 표시
        self.Model_name.setText(self.select_model)
    ####################################image or video####################################

    ####################################camera####################################
    def cam_button(self):
        self.yolo_predict_cam.source = "0"
        self.show_status('현재 페이지: Webcam 감지 페이지')
        # 리소스를 절약하려면 image or video 스레드를 종료해야함.
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            self.yolo_thread.quit() # 쓰레드 종료
            self.yolo_thread_cam.quit()
            self.stop()
            self.cam_stop()

        if self.PageIndex != 2:
            self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))   # 오른쪽 상단의 설정 버튼
            
    # cam 제어 시작/일시 중지 감지
    def cam_run_or_continue(self):
        def handle_no_camera():
            self.show_status('카메라가 감지되지 않음')
            self.run_button_cam.setChecked(False)

        def start_detection():
            self.run_button_cam.setChecked(True)  # 시작 버튼
            self.save_txt_button_cam.setEnabled(False)  # 감지 시작 후 확인 및 저장 비활성화
            self.save_res_button_cam.setEnabled(False)
            self.show_status('감지 중 입니다...')
            self.yolo_predict_cam.continue_dtc = True

            if not self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict_cam.continue_dtc = False
            self.show_status("감지가 일시중지되었습니다....")
            self.run_button_cam.setChecked(False)  # 정지 버튼

        if self.yolo_predict_cam.source == '':
            handle_no_camera()
        else:
            self.yolo_predict_cam.stop_dtc = False

            if self.run_button_cam.isChecked():
                start_detection()
            else:
                pause_detection()

    # cam 메인 창에는 원본 사진과 감지 결과가 표시됩니다.
    @staticmethod
    def cam_show_image(img_src, label, instance=None):
        try:
            # 원본 이미지의 높이, 너비, 채널 수 가져옴
            ih, iw, _ = img_src.shape

            # 라벨의 너비와 높이를 가져옴.
            w, h = label.geometry().width(), label.geometry().height()

            # 원래 데이터 비율을 유지하고 조정된 크기를 계산함.
            if iw / w > ih / h:
                scal = w / iw
                nw, nh = w, int(scal * ih)
            else:
                scal = h / ih
                nw, nh = int(scal * iw), h

            # 이미지 크기를 조정하고 RGB 형식으로 변환
            frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

            # 이미지 데이터를 Qt 이미지 객체로 변환하여 라벨에 표시
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # 예외 처리 및 오류 메시지 출력
            traceback.print_exc()
            print(f"Error: {e}")
            if instance is not None:
                instance.show_status('%s' % e)

    # 감지 매개변수 변경
    def cam_change_val(self, c, flag):
        def update_iou():
            value = c / 100
            self.iou_spinbox_cam.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict_cam.iou_thres = value

        def update_conf():
            value = c / 100
            self.conf_spinbox_cam.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict_cam.conf_thres = value

        def update_speed():
            self.speed_spinbox_cam.setValue(c)
            self.show_status(f'Delay: {c} ms')
            self.yolo_predict_cam.speed_thres = c  # 밀리초

        update_actions = {
            'iou_spinbox_cam': lambda: self.iou_slider_cam.setValue(int(c * 100)),
            'iou_slider_cam': update_iou,
            'conf_spinbox_cam': lambda: self.conf_slider_cam.setValue(int(c * 100)),
            'conf_slider_cam': update_conf,
            'speed_spinbox_cam': lambda: self.speed_spinbox_cam.setValue(c),
            'speed_slider_cam': update_speed,
        }

        if flag in update_actions:
            update_actions[flag]()

    # 모델 변경
    def cam_change_model(self, c):
        # 현재 선택된 모델명을 가져옴.
        self.select_model_cam = self.model_box_cam.currentText()
        
        # 작업에 따라 모델 경로의 접두사를 설정.
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # YOLO 인스턴스의 새 모델 이름 설정
        self.yolo_predict_cam.new_model_name = f"{model_prefix}/{self.select_model_cam}"

        # 모델이 변경되었다는 메시지 표시
        self.show_status(f'Change Model: {self.select_model_cam}')
        
        # 인터페이스에 새 모델 이름 표시
        self.Model_name_cam.setText(self.select_model_cam)

    # 테스트 결과 저장 버튼 -- 사진/비디오
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 이미지 실행 결과가 저장되지 않는다는 메시지가 표시.
            self.show_status('NOTE：Webcam 결과는 저장되지 않습니다.')
            
            # YOLO 인스턴스의 저장된 결과 플래그를 False로 설정.
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # 이미지 실행 결과가 저장된다는 메시지를 표시.
            self.show_status('NOTE：Webcam 결과가 저장됩니다.')
            
            # YOLO 인스턴스의 저장된 결과 플래그를 True로 설정.
            self.yolo_thread_cam.save_res = True

    # 테스트 결과 저장 버튼 -- 라벨（txt）
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # 라벨 결과가 저장되지 않는다는 메시지를 표시.
            self.show_status('NOTE：Label 결과는 저장되지 않습니다.')
            
            # YOLO 인스턴스의 저장된 레이블 플래그를 False로 설정.
            self.yolo_thread_cam.save_txt_cam = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # 라벨 결과가 저장된다는 메시지를 표시.
            self.show_status('NOTE：Label 결과가 저장됩니다.')
            
            # YOLO 인스턴스의 save-label 플래그를 True로 설정.
            self.yolo_thread_cam.save_txt_cam = True

    # cam 종료 버튼 및 관련 상태 처리
    def cam_stop(self):
        def stop_yolo_thread():
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # 結束線程
            self.yolo_predict_cam.stop_dtc = True

        def reset_ui_elements():
            self.run_button_cam.setChecked(False)
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)
            self.pre_cam.clear()
            self.res_cam.clear()
            self.Class_num_cam.setText('--')
            self.Target_num_cam.setText('--')
            self.fps_label_cam.setText('--')

        stop_yolo_thread()
        self.show_status('감지가 종료되었습니다.')
        reset_ui_elements()
    ####################################camera####################################
    ####################################rtsp####################################
    # rtsp 입력 주소
    def rtsp_button(self):
        def stop_yolo_threads():
            # YOLO 스레드가 실행 중이면 스레드를 종료.
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()

            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()

            # YOLO 인스턴스 감지 중지
            self.stop()
            self.cam_stop()

        def load_rtsp_window():
            self.rtsp_window = Window()
            config_file = 'config/ip.json'

            if not os.path.exists(config_file):
                ip = "rtsp://admin:admin888@192.168.1.2:555"
                new_config = {"ip": ip}
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, ensure_ascii=False, indent=2)
            else:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                ip = config['ip']

            self.rtsp_window.rtspEdit.setText(ip)
            self.rtsp_window.show()
            self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

        self.yolo_predict_cam.stream_buffer = True
        # 리소스를 절약 하기위한 Video Thread 종료
        stop_yolo_threads()

        # RTSP 감지 페이지로 전환
        self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.show_status('현재 페이지: RTSP 감지 페이지')

        # RTSP 설정 창 로드
        load_rtsp_window()

        # 우측상단 설정버튼 연결기능 설정
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))

    # 웹 소스 로드
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Hint', text='RTSP 로드...', time=1000, auto=True).exec()
            self.yolo_predict_cam.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)
    ####################################rtsp####################################
    ####################################공용####################################
    # 하단 상태 표시줄 정보 표시
    def show_status(self, msg):
        self.status_bar.setText(msg)
        def handle_page_0():
            if msg == '테스트 완료':
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)

                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()

            elif msg == '감지가 종료되었습니다.':
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)
                self.progress_bar.setValue(0)

                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()

                self.pre_video.clear()
                self.res_video.clear()
                self.Class_num.setText('--')
                self.Target_num.setText('--')
                self.fps_label.setText('--')

        def handle_page_2():
            if msg == '감지가 종료되었습니다.':
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                self.run_button_cam.setChecked(False)
                self.progress_bar_cam.setValue(0)

                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()

                self.pre_cam.clear()
                self.res_cam.clear()
                self.Class_num_cam.setText('--')
                self.Target_num_cam.setText('--')
                self.fps_label_cam.setText('--')

        # 다양한 페이지에 따라 다양한 상태 처리
        if self.PageIndex == 0:
            handle_page_0()
        elif self.PageIndex == 2:
            handle_page_2()

    # 모델 파일 변경 사항을 주기적으로 모니터링
    def ModelBoxRefre(self):
        def update_model_box(folder):
            pt_list = os.listdir(folder)
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)))
            return pt_list

        folder_paths = {
            'Classify': './models/classify',
            'Detect': './models/detect',
            'Pose': './models/pose',
            'Segment': './models/segment',
            'Track': './models/track'
        }

        if self.task in folder_paths:
            pt_list = update_model_box(folder_paths[self.task])
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

    # 마우스 위치 가져오기(제목 표시줄을 누른 상태에서 창을 드래그하는 데 사용됨)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # 창 크기 조정 시 조정 최적화(창 오른쪽 하단 가장자리를 드래그하여 창 크기 조정)
    def resizeEvent(self, event):
        # 크기 조정 핸들 업데이트
        UIFuncitons.resize_grips(self)

    # 상단 우측 버튼 설정 초기화
    def load_config(self):
        config_file = 'config/setting.json'
        
        default_config = {
            "iou": 0.26,
            "conf": 0.33,
            "rate": 10,
            "save_res": 0,
            "save_txt": 0,
            "save_res_cam": 0,
            "save_txt_cam": 0
        }

        config = default_config.copy()
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        
        def update_ui(config):
            ui_elements = {
                "save_res": (self.save_res_button, self.yolo_predict, "save_res"),
                "save_txt": (self.save_txt_button, self.yolo_predict, "save_txt"),
                "save_res_cam": (self.save_res_button_cam, self.yolo_predict_cam, "save_res_cam"),
                "save_txt_cam": (self.save_txt_button_cam, self.yolo_predict_cam, "save_txt_cam"),
            }

            for key, (button, instance, attr) in ui_elements.items():
                button.setCheckState(Qt.Checked if config[key] else Qt.Unchecked)
                setattr(instance, attr, config[key] != 0)

            self.run_button.setChecked(False)
            self.run_button_cam.setChecked(False)

            self.rate = config.get('rate',0)
            self.speed_spinbox.setValue(self.rate)  # speed 스핀 텍스트 박스 설정 값 으로 초기화
            self.speed_slider.setValue(self.rate)  # speed 슬라이더 설정 값 으로 초기화 (Delay(ms))
            self.speed_spinbox_cam.setValue(self.rate)  # speed 스핀 테스트 박스 설정 값 으로 초기화
            self.speed_slider_cam.setValue(self.rate)  # speed 슬라이드 설정 값 으로 초기화 (Delay(ms))
        
        update_ui(config)

    # 이벤트를 닫고 스레드를 종료한 후 설정을 저장
    def closeEvent(self, event):
        # 설정 파일에 구성 저장
        config_file = 'config/setting.json'
        config = {
            "iou": self.iou_spinbox.value(),
            "conf": self.conf_spinbox.value(),
            "rate": self.speed_spinbox.value(),
            "save_res": 0 if self.save_res_button.checkState() == Qt.Unchecked else 2,
            "save_txt": 0 if self.save_txt_button.checkState() == Qt.Unchecked else 2,
            "save_res_cam": 0 if self.save_res_button_cam.checkState() == Qt.Unchecked else 2,
            "save_txt_cam": 0 if self.save_txt_button_cam.checkState() == Qt.Unchecked else 2
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 스레드 및 애플리케이션 종료
        def quit_threads():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()

            self.yolo_predict_cam.stop_dtc = True
            self.yolo_thread_cam.quit()
            
            # 종료 프롬프트를 표시하고 3초 동안 기다림.
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            
            # 애플리케이션 종료
            sys.exit(0)
        
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            quit_threads()
        else:
            sys.exit(0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        
    def dropEvent(self, event):
        def handle_directory(directory):
            image_formats = {".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"}
            image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.splitext(filename)[1].lower() in image_formats]

            if image_files:
                self.yolo_predict.source = image_files
                self.show_image(self.yolo_predict.source[0], self.pre_video, 'path')
                self.show_status('폴더 로드：{}'.format(os.path.basename(directory)))
            else:
                self.show_status('폴더에 사진이 없네요...')

        def handle_file(file):
            self.yolo_predict.source = file
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in {".avi", ".mp4"}:
                self.cap = cv2.VideoCapture(self.yolo_predict.source)
                ret, frame = self.cap.read()
                if ret:
                    self.show_image(frame, self.pre_video, 'img')
            else:
                self.show_image(self.yolo_predict.source, self.pre_video, 'path')

            self.show_status('파일 로드：{}'.format(os.path.basename(self.yolo_predict.source)))

        try:
            file = event.mimeData().urls()[0].toLocalFile()
            if file:
                if os.path.isdir(file):
                    handle_directory(file)
                else:
                    handle_file(file)
        except Exception as e:
            self.show_status('오류：{}'.format(e))

####################################공용####################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
