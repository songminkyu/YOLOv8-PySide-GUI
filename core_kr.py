####################ultralytics==8.2.66####################
####################cuda11.8####################
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.engine.predictor import BasePredictor, STREAM_WARNING
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.files import increment_path
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz, check_imshow, check_yaml
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.cfg import get_cfg, get_save_dir
from moviepy.editor import VideoFileClip
from ultralytics.trackers import track
from ultralytics import YOLO

from utils.image_save import ImageSaver
from PySide6.QtCore import Signal, QObject
from collections import defaultdict
from pathlib import Path

import numpy as np
from datetime import datetime
from PIL import Image
import threading
import traceback
import re
import time
import json
import torch
import cv2
import os

def seconds_to_hms(seconds):
    if seconds > 59:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
    else:
        hours = 0
        minutes = 0
    return hours, minutes, seconds

class YoloPredictor(BasePredictor, QObject):
    # 다른 부분과 통신하는 데 사용되는 신호 정의
    yolo2main_pre_img = Signal(np.ndarray)   # 원본 이미지 신호
    yolo2main_res_img = Signal(np.ndarray)   # 테스트 결과 신호
    yolo2main_status_msg = Signal(str)       # 감지/일시중지/중지/테스트 완료/오류 보고 신호
    yolo2main_fps = Signal(str)              # 프레임 속도 신호
    yolo2main_labels = Signal(dict)          # 탐지대상 결과(항목별 개수)
    yolo2main_time = Signal(str)
    yolo2main_silder_range = Signal(str)     # 무결성 신호
    yolo2main_progress = Signal(int)
    yolo2main_class_num = Signal(int)        # 감지된 카테고리 수
    yolo2main_target_num = Signal(int)       # 감지된 타겟 수

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # 상위 클래스의 초기화 메소드 호출
        super(YoloPredictor, self).__init__()
        # PyQt의 QObject 초기화
        QObject.__init__(self)

        # 구성 파일 구문 분석
        self.args = get_cfg(cfg, overrides)
        # 모델 저장 디렉터리 설정
        self.save_dir = get_save_dir(self.args)
        # 모델이 워밍업(warmup)을 완료했는지 여부를 표시하는 플래그를 초기화.
        self.done_warmup = False
        # 이미지를 표시할지 확인
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

        # GUI 관련 속성
        self.used_model_name = None  # 사용할 감지 모델의 이름
        self.new_model_name = None   # 실시간으로 변경되는 모델명
        self.source = ''             # 입력 소스
        self.stop_dtc = False        # 탐지 종료 플래그
        self.continue_dtc = True     # 테스트를 일시 중지하려면 서명하세요.
        self.save_res = False        # 테스트 결과를 저장하는 플래그
        self.save_txt = False        # 태그(txt) 파일을 저장하기 위한 플래그
        self.save_res_cam = False    # webcam 테스트 결과를 저장하도록 플래그 지정
        self.save_txt_cam = False    # webcam 태그(txt) 파일을 저장하기 위한 플래그
        self.iou_thres = 0.45        # IoU 임계값
        self.conf_thres = 0.25       # 신뢰 임계값
        self.speed_thres = 0         # 지연, 밀리초
        self.labels_dict = {}        # 탐지 결과 라벨을 반환
        self.progress_value = 0      # 진행률 표시줄 값
        self.task = ''
        self.stream_buffer = False

        # 설정이 완료되면 다음 속성을 사용할 수 있습니다.
        self.model = None
        self.task_frist = ''
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self.frames = None
        self.fps = None
        self.start_time = None  # 타이밍 시작 시간
        self.elapsed_time = 0  # 지나간 시간
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    # main for detect
    @smart_inference_mode()
    def run(self, *args, **kwargs):
        try:
            if self.args.verbose:
                LOGGER.info('')

            self.yolo2main_status_msg.emit('모델 로딩...')  # 모델 로딩 메시지 보내기
            self.initialize_model()  # 모델 초기화

            with self._lock:  # 스레드 안전성 보장의 결과
                if self.task == 'Track':
                    track_history = defaultdict(lambda: [])
                else:
                    track_history = None

                self.setup_source(self.source or self.args.source)  # 소스 설정
                self.check_save_dirs()  # 저장 디렉터리를 확인하세요.
                self.set_video_total_time()

                if not self.done_warmup:
                    self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))  # 모델 워밍업
                    self.done_warmup = True

                self.frames = self.dataset.frames if hasattr(self.dataset, 'frames') else None
                if isinstance(self.frames, list):
                    self.frames = self.frames[0]
                elif isinstance(self.frames, int):
                    pass
                else:
                    self.frames = None
                self.seen, self.windows, self.batch = 0, [], None
                profilers = [ops.Profile(device=self.device) for _ in range(3)]

                batch = iter(self.dataset)
                while True:
                    self.check_model_update()  # 모델 업데이트 확인

                    if 'obb' in self.used_model_name and self.task == 'Detect':
                        self.task = 'obb'
                        self.task_frist = 'Detect'
                    if self.task == 'obb' and 'obb' not in self.used_model_name:
                        self.task = self.task_frist

                    if self.continue_dtc:
                        if self.start_time is None:  # 타이밍이 시작되었는지 확인
                            self.start_time = datetime.now()  # 시작 타이밍
                        else:
                            # 경과 시간 업데이트
                            self.elapsed_time += (datetime.now() - self.start_time).total_seconds()
                            self.start_time = datetime.now()  # 시작 시간 재설정

                        try:
                            batch = next(self.dataset)
                        except StopIteration:
                            break
                        self.batch = batch
                        self.yolo2main_status_msg.emit('감지 중...')  # 감지 메시지 보내기

                        paths, im0s, s = self.batch
                        im = self.preprocess_batch(im0s, profilers[0])  # 배치 전처리
                        preds = self.inference(im, profilers[1], *args, **kwargs)  # 추론
                        if self.task == 'Track':
                            self.results, self.track_pointlist = self.postprocess_results(preds, im, im0s, track_history, profilers[2])  # 후처리 결과
                        else:
                            self.results = self.postprocess_results(preds, im, im0s, track_history, profilers[2])

                        self.run_callbacks('on_predict_postprocess_end')
                        self.handle_results(im, im0s, paths, s, profilers)  # 처리 결과

                    else:
                        # 타이밍을 일시 중지하고 경과 시간을 유지.
                        if self.start_time is not None:
                            self.elapsed_time += (datetime.now() - self.start_time).total_seconds()
                            self.start_time = None  # 시작 시간 재설정

                    if self.check_completion():  # 완료되었는지 확인
                        break

                    if self.stop_dtc:
                        self.release_video_writers()  # 영상데이터 출력 아니면 저장인지??
                        self.yolo2main_status_msg.emit('감지가 종료되었습니다.')
                        break

                    if self.args.verbose and self.seen:
                        t = tuple(x.t / self.seen * 1e3 for x in profilers)

                    if self.save_txt or self.save_txt_cam or self.args.save_crop:
                        nl = len(list(self.save_dir.glob("labels/*.txt")))
                        s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if (self.save_txt or self.save_txt_cam) else ""

                if not self.source_type.stream and self.frames is None or self.frame is None:
                    self.yolo2main_status_msg.emit('테스트 완료')  # 감지 완료 메시지 보내기
                elif self.source_type.stream and self.frames is not None:
                    self.yolo2main_status_msg.emit('테스트 완료')  # 감지 완료 메시지 보내기

                if self.start_time is not None:  # 경과 시간 업데이트
                    self.elapsed_time = 0
                    self.start_time = None

        except Exception as e:
            self.yolo2main_status_msg.emit(f'오류: {str(e)}')  # 오류 메시지 보내기
            LOGGER.error(f'Error in run: {str(e)}')

    def initialize_model(self):
        if self.task == 'Track':
            self.track_model = YOLO(self.new_model_name)  # 추적 모델 초기화
        self.setup_model(self.new_model_name)  # 모델 설정
        self.used_model_name = self.new_model_name

    def check_model_update(self):
        if self.used_model_name != self.new_model_name:
            if self.task == 'Track':
                self.track_model = YOLO(self.used_model_name)  # 추적 모델 업데이트
            self.setup_model(self.new_model_name)  # 새 모델 설정
            self.used_model_name = self.new_model_name

    def preprocess_batch(self, im0s, profiler):
        with profiler:
            return self.classify_preprocess(im0s) if self.task == 'Classify' else self.preprocess(im0s)  # 배치 전처리

    def postprocess_results(self, preds, im, im0s, track_history, profiler):
        with profiler:
            if self.task == 'Classify':
                return self.classify_postprocess(preds, im, im0s)  # 분류 후 처리
            elif self.task == 'Track':
                return self.track_postprocess(self.track_model, track_history, preds, im, im0s)  # 후처리 추적
            else:
                postprocess_methods = {
                    'Detect': self.postprocess,
                    'obb': self.obb_postprocess,
                    'Segment': self.segment_postprocess,
                    'Pose': self.pose_postprocess,
                }
                return postprocess_methods[self.task](preds, im, im0s)  # 기타 후처리 작업

    def handle_results(self, im, im0s, paths, s, profilers):
        n = len(im0s)
        for i in range(n):
            self.seen += 1
            self.results[i].speed = {
                'preprocess': profilers[0].dt * 1E3 / n,  # 전처리 시간
                'inference': profilers[1].dt * 1E3 / n,  # 추론 시간
                'postprocess': profilers[2].dt * 1E3 / n  # 사후 처리 시간
            }

            self.class_nums = 0
            self.target_nums = 0
            s[i] += self.write_results(i, Path(paths[i]), im, s)  # 결과 쓰기
            im0 = None if self.source_type.tensor else im0s[i].copy()
            if 'no detections' in s:
                self.im = im0

            self.update_progress()  # 업데이트 진행
            self.send_results(im0)  # 결과 보내기

    def update_progress(self):
        if isinstance(self.frame, int) and (not isinstance(self.frames, list) and self.frames is not None):
            self.progress_value = int(self.frame / self.frames * 1000)  # 진행 가치 계산
        elif not self.source or self.frames is None or self.frame is None:
            self.progress_value = int(1000)  # 진행률 값을 1000으로 설정
        self.yolo2main_progress.emit(self.progress_value)  # Emit progress
        
    def send_results(self, im0):
        self.yolo2main_pre_img.emit(im0 if isinstance(im0, np.ndarray) else im0[0])  # 전처리된 이미지 보내기
        self.yolo2main_res_img.emit(self.im)  # 결과 이미지 보내기
        if self.task != 'Classify':
            self.yolo2main_class_num.emit(self.class_nums)  # 전송 카테고리 수
            self.yolo2main_target_num.emit(self.target_nums)  # 목표 수량 보내기
        if not isinstance(self.frames, list) and self.frames is not None:
            self.yolo2main_fps.emit(str(self.fps))  # FPS 보내기
        if self.speed_thres != 0:
            time.sleep(self.speed_thres / 1000)  # 지연
        self.set_video_current_time()

    def check_completion(self):
        if (self.frame == self.frames) and self.frames is not None and self.frame is not None:
            self.release_video_writers()  # 영상데이터 출력 아니면 저장인지??
            self.yolo2main_status_msg.emit('테스트 완료')  # 완료 상태 보내기
            return True
        return False

    def release_video_writers(self):
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()  # 영상데이터 출력

    def set_video_total_time(self):
        if self.source.endswith((".avi", ".mp4")):
            clip = VideoFileClip(self.source)
            if clip:
                self.duration = int(clip.duration)  # 影片長度（秒）
                self.total_hours, self.total_minutes, self.total_seconds = seconds_to_hms(self.duration)
                # print(f"비디오 길이：{self.total_hours}:{self.total_minutes:02}:{self.total_seconds:02}")
        else:
            self.duration = 1
            self.total_hours, self.total_minutes, self.total_seconds = '', '', ''

    def set_video_current_time(self):
        if self.frames is not None and self.dataset.mode != "stream":     
            new_fps = round(self.frames / self.duration, 2)
            current = round(self.frame / new_fps)
            current_hours, current_minutes, current_seconds = seconds_to_hms(current)
            current_time = f"{current_hours:02}:{current_minutes:02}:{current_seconds:02}"
            total_time = f"{self.total_hours:02}:{self.total_minutes:02}:{self.total_seconds:02}"
            self.yolo2main_time.emit(f"{current_time} / {total_time}")
            self.yolo2main_silder_range.emit(f"{current},{self.duration}")

        elif self.frames is None:
            current = 1
            current_hours, current_minutes, current_seconds = '', '', ''
            current_time = f"{current_hours:02}:{current_minutes:02}:{current_seconds:02}"
            total_time = f"{self.total_hours:02}:{self.total_minutes:02}:{self.total_seconds:02}"
            self.yolo2main_time.emit(f"{current_time} / {total_time}")
            self.yolo2main_silder_range.emit(f"{current},{self.duration}")

        elif self.dataset.mode == "stream":
            self.cam_hours, self.cam_minutes, self.cam_seconds = seconds_to_hms(int(self.elapsed_time))
            cam_time = f"{self.cam_hours}:{self.cam_minutes:02}:{self.cam_seconds:02}"
            self.yolo2main_time.emit("프로그램 가동시간：" + f"{cam_time}")

    def check_save_dirs(self):
        if self.save_res or self.save_txt or self.save_res_cam or self.save_txt_cam:
            (self.save_dir / 'labels' if (self.save_txt or self.save_txt_cam) else self.save_dir).mkdir(parents=True, exist_ok=True)  # 저장 디렉터리를 확인하세요.

    def inference(self, im, profiler, *args, **kwargs):
        with profiler:
            """Runs inference on a given image using the specified model and arguments."""
            visualize = (
                increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
                if self.args.visualize and (not self.source_type.tensor)
                else False
            )
            return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def preprocess(self, img):
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)

        img = img.to(self.device, non_blocking=True)  # non_blocking for async transfer
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # Non-max suppression
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Convert orig_imgs to numpy if needed
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # Scale boxes and create results in a single loop
        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def obb_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        # Non-max suppression
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        # Convert orig_imgs to numpy if needed
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # Regularize and scale rotated bounding boxes
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # Concatenate rotated boxes with confidence and class predictions
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results

    def classify_preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device, non_blocking=True)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def classify_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, probs=pred))
        return results

    def segment_postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def pose_postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results

    def track_postprocess(self, model, track_history, preds, img, orig_imgs):
        # Set the track model for track line
        track_result = model.track(orig_imgs, persist=True)

        # Set the track preds
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs,
                          list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            # Store result
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=track_result[0].boxes.data))
            
            # Get the boxes and track IDs
            boxes = track_result[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = track_result[0].boxes.id.int().cpu().tolist()
            output = []
            # Plot the tracks
            if results[0].boxes.id is not None:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                # Get the points
                    points = np.hstack(track).astype(np.int32).reshape(
                    (-1, 1, 2))
                    output.append(points)
        return results, output

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction)
            )
            if self.task == "Classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.stream_buffer,
        )

        self.source_type = self.dataset.source_type

        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def write_results(self, i, p, im, s):
        """Write inference results to a file or directory."""
        string = ""  # print string

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            self.frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            self.frame = int(match.group(1)) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{self.frame}"))

        string += f"%gx%g " % im.shape[2:]

        result = self.results[i]
        result.save_dir = str(self.save_dir)

        string += result.verbose() + f"{result.speed['inference']:.1f}ms"

        if self.task == 'Classify':
            prob = result.probs
            # prob.top5는 필요에 따라 여기에서 출력하거나 처리할 수 있습니다.
        else:
            det = result.boxes if self.task != 'obb' else result.obb

            if len(det) == 0:
                string += "(no detections)"
            else:
                for c in det.cls.unique():
                    n = (det.cls == c).sum()
                    self.target_nums += int(n)
                    self.class_nums += 1

        self.plotted_img = result.plot(
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels,
            im_gpu=None if self.args.retina_masks else im[i],
        )

        if self.save_txt or self.save_txt_cam:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))

        self.save_predicted_images(str(self.save_dir / p.name), self.frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """지정된 경로에 비디오 예측을 mp4로 저장합니다."""
        self.im = self.plotted_img
        if self.task == 'Track':
            for points in self.track_pointlist:
                cv2.polylines(self.im, [points], isClosed=False, color=(203, 224, 252), thickness=5)

        if self.dataset.mode in {"stream", "video"}:
            self.fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.rsplit(".", 1)[0]}_frames/'

            if save_path not in self.vid_writer and (self.save_res or self.save_res_cam):
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=self.fps,
                    frameSize=(self.im.shape[1], self.im.shape[0]),
                )

            if self.save_res or self.save_res_cam:
                self.vid_writer[save_path].write(self.im)

            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", self.im)
        else:
            if self.save_res or self.save_res_cam:
                image_saver = ImageSaver(self.im)
                image_saver.save_image(save_path)
