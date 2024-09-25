from main_kr import *
from custom_grips import CustomGrip
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QEvent, QTimer
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import time

GLOBAL_STATE = False    # max min flag
GLOBAL_TITLE_BAR = True


class UIFuncitons(MainWindow):
    # 왼쪽 메뉴 펼치기
    def toggleMenu(self, enable):
        if enable:
            standard = 68  # 왼쪽 메뉴의 표준 너비
            maxExtend = 180  # 확장 시 왼쪽 메뉴의 최대 너비
            width = self.LeftMenuBg.width()  # 현재 메뉴 너비

            if width == 68:  # 현재 메뉴가 축소되어 있는 경우
                widthExtended = maxExtend  # 展開後的寬度
            else:
                widthExtended = standard  # 확장된 너비

            # 動畫效果
            self.animation = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.animation.setDuration(500)  # 애니메이션 시간(밀리초)
            self.animation.setStartValue(width)  # 애니메이션의 시작 너비
            self.animation.setEndValue(widthExtended)  # 애니메이션의 끝 너비
            self.animation.setEasingCurve(QEasingCurve.InOutQuint)  # 애니메이션 이징 곡선
            self.animation.start()  # 애니메이션 시작

    # 오른쪽의 설정 메뉴를 확장하세요.
    def settingBox(self, enable):
        if enable:
            # 너비 얻기
            widthRightBox = self.prm_page.width()  # 오른쪽 메뉴 너비
            widthLeftBox = self.LeftMenuBg.width()  # 왼쪽 메뉴의 너비
            maxExtend = 220  # 메뉴 확장 시 최대 너비 설정
            standard = 0

            # 최대 너비 설정
            if widthRightBox == 0:  # 현재 오른쪽의 설정 메뉴가 접혀 있는 경우
                widthExtended = maxExtend  # 확장된 너비
            else:
                widthExtended = standard  # 축소된 너비

            # 왼쪽 메뉴에 애니메이션 설정
            self.left_box = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)  # 애니메이션 시간(밀리초)
            self.left_box.setStartValue(widthLeftBox)  # 애니메이션의 시작 너비
            self.left_box.setEndValue(68)  # 애니메이션 끝 너비(축소된 너비)
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)  # 애니메이션 이징 곡선

            # 오른쪽 설정 메뉴의 애니메이션 설정
            self.right_box = QPropertyAnimation(self.prm_page, b"minimumWidth")
            self.right_box.setDuration(500)  # 애니메이션 시간(밀리초)
            self.right_box.setStartValue(widthRightBox)  # 애니메이션의 시작 너비
            self.right_box.setEndValue(widthExtended)  # 애니메이션의 끝 너비
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)  # 애니메이션 이징 곡선

            # 병렬 애니메이션 그룹 만들기
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()  # 애니메이션 시작

    # 오른쪽 설정 메뉴 확장
    def cam_settingBox(self, enable):
        if enable:
            # 너비 얻기
            widthRightBox = self.prm_page_cam.width()  # 오른쪽 메뉴의 너비를 설정합니다.
            widthLeftBox = self.LeftMenuBg.width()  # 왼쪽 메뉴의 너비
            maxExtend = 220  # 메뉴 확장 시 최대 너비 설정
            standard = 0

            # 최대 너비 설정
            if widthRightBox == 0:  # 현재 오른쪽 설정메뉴가 접혀있는 경우
                widthExtended = maxExtend  # 확장된 너비
            else:
                widthExtended = standard  # 축소된 너비

            # 왼쪽 메뉴에 애니메이션 설정
            self.left_box = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)  # 애니메이션 시간(밀리초)
            self.left_box.setStartValue(widthLeftBox)  # 애니메이션의 시작 너비
            self.left_box.setEndValue(68)  # 애니메이션의 끝 너비(축소된 너비)
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)  # 애니메이션 이징 곡선

            # 오른쪽 설정 메뉴의 애니메이션 설정
            self.right_box = QPropertyAnimation(self.prm_page_cam, b"minimumWidth")
            self.right_box.setDuration(500)  # 애니메이션 시간(밀리초)
            self.right_box.setStartValue(widthRightBox)  # 애니메이션의 시작 너비
            self.right_box.setEndValue(widthExtended)  # 애니메이션의 끝 너비
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)  # 애니메이션 이징 곡선

            # 병렬 애니메이션 그룹 만들기
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()  # 애니메이션 시작

    # 最大化/還原視窗
    def maximize_restore(self):
        global GLOBAL_STATE  # 전역 변수 사용
        status = GLOBAL_STATE  # 전역 변수의 값 가져오기
        if status == False:  # 창이 최대화되지 않은 경우
            GLOBAL_STATE = True  # 전역 변수를 True(최대화된 상태)로 설정.
            self.showMaximized()  # 창 최대화
            self.max_sf.setToolTip("Restore")  # 최대화 버튼의 도구 설명 텍스트 변경
            self.frame_size_grip.hide()  # 창 크기 조정 버튼 숨기기
            self.left_grip.hide()  # 4면 조정 버튼 숨기기
            self.right_grip.hide()
            self.top_grip.hide()
            self.bottom_grip.hide()
        else:
            GLOBAL_STATE = False  # 전역 변수를 False(최대화되지 않은 상태)로 설정
            self.showNormal()  # 복원 창(최소화)
            self.resize(self.width() + 1, self.height() + 1)  # 최소화 후 창 크기 수정
            self.max_sf.setToolTip("Maximize")  # 최대화 버튼의 도구 설명 텍스트 변경
            self.frame_size_grip.show()  # 창 크기 조정 버튼 표시
            self.left_grip.show()   # 4면 조정을 위한 버튼 표시
            self.right_grip.show()  # 4면 조정을 위한 버튼 표시
            self.top_grip.show()    # 4면 조정을 위한 버튼 표시
            self.bottom_grip.show() # 4면 조정을 위한 버튼 표시
    
    # 창 제어의 정의
    def uiDefinitions(self):
        # 최대화 복원하려면 제목 표시줄을 두 번 클릭 해서 복원 하도록 설정
        def dobleClickMaximizeRestore(event):
            if event.type() == QEvent.MouseButtonDblClick:
                QTimer.singleShot(250, lambda: UIFuncitons.maximize_restore(self))
        self.top.mouseDoubleClickEvent = dobleClickMaximizeRestore
        
        # 창 이동/최대화/복원
        def moveWindow(event):
            if GLOBAL_STATE:  # 창이 최대화되면 복원된 상태로 전환
                UIFuncitons.maximize_restore(self)
            if event.buttons() == Qt.LeftButton:  # 창 이동
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
        self.top.mouseMoveEvent = moveWindow
        
        # 맞춤형 스트레치 버튼
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)

        # 창 최소화
        self.min_sf.clicked.connect(lambda: self.showMinimized())
        # 창 최대화/복원
        self.max_sf.clicked.connect(lambda: UIFuncitons.maximize_restore(self))
        # 프로그램 종료
        self.close_button.clicked.connect(self.close)

    # 창 네 면의 늘이기 제어
    def resize_grips(self):
        # 왼쪽 늘이기 버튼의 위치와 크기 설정
        self.left_grip.setGeometry(0, 10, 10, self.height())
        # 오른쪽 늘이기 버튼의 위치와 크기 설정
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        # 위쪽 스트레치 버튼의 위치와 크기를 설정
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        # 하단 스트레치 버튼의 위치와 크기를 설정
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 그림자 효과를 추가하는 디스플레이 모드
    def shadow_style(self, widget, Color):
        shadow = QGraphicsDropShadowEffect(self)  # 그림자 효과 개체 만들기
        shadow.setOffset(8, 8)  # 그림자의 오프셋 설정
        shadow.setBlurRadius(38)  # 그림자의 흐림 반경 설정
        shadow.setColor(Color)  # 그림자의 색상을 설정
        widget.setGraphicsEffect(shadow)  # 지정된 위젯에 그림자 효과 적용