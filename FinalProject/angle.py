import sys
import platform
import cv2
import time
import math
import mediapipe as mp
import numpy as np
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from comtypes.client import CreateObject
from comtypes import GUID

# Kivy Imports
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ListProperty
from kivy.core.window import Window

# --- OS CHECK ---
if platform.system() != "Windows":
    raise SystemExit("This script runs only on Windows.")

try:
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator
except ImportError:
    raise SystemExit("Install required packages: pip install pycaw comtypes kivy opencv-python mediapipe")

# --- CONFIG ---
eCapture = 1  # 1 = MICROPHONE
eConsole = 0
GUID_MMDEVICE = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"

# --- CUSTOM WIDGET ---
class CurvedNeonBox(Widget):
    glow_color = ListProperty([0, 1, 1, 1])

# --- KIVY KV DESIGN ---
KV = '''
#:import hex kivy.utils.get_color_from_hex

<CurvedNeonBox>:
    canvas.before:
        Color:
            rgba: (self.glow_color[0], self.glow_color[1], self.glow_color[2], 0.15)
        RoundedRectangle:
            pos: (self.x - 2, self.y - 2)
            size: (self.width + 4, self.height + 4)
            radius: [15,]
        Color:
            rgba: self.glow_color
        Line:
            width: 1.2
            rounded_rectangle: (self.x, self.y, self.width, self.height, 15)

<MainUI>:
    canvas.before:
        Color:
            rgba: hex('#020205')
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgba: (1, 1, 1, 0.03)
        Line:
            points: (0, self.height*0.3, self.width, self.height*0.3)
        Line:
            points: (0, self.height*0.7, self.width, self.height*0.7)

    Label:
        text: "AirVolume"
        font_size: '48sp'
        font_name: 'Roboto'
        bold: True
        color: hex('#00FFFF')
        pos_hint: {'center_x': 0.5, 'top': 0.98}
        size_hint: (None, None)
        size: (300, 60)
        canvas.before:
            Color:
                rgba: (0, 1, 1, 0.1)
            Ellipse:
                pos: (self.center_x - 100, self.center_y - 20)
                size: (200, 40)

    Label:
        text: "Function Trace Debugger"
        font_size: '14sp'
        letter_spacing: 2
        color: hex('#005577')
        pos_hint: {'center_x': 0.5, 'top': 0.91}
        size_hint: (None, None)
        size: (300, 30)

    # --- LEFT PANEL (LOGS) ---
    FloatLayout:
        pos_hint: {'x': 0.02, 'center_y': 0.5}
        size_hint: (0.35, 0.65)
        
        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (0, 1, 0.5, 1) # Spring Green

        Label:
            text: "/// PYTHON_STACK_TRACE"
            color: hex('#00FF88')
            pos_hint: {'x': 0.05, 'top': 0.96}
            size_hint: (0.9, 0.05)
            text_size: self.size
            halign: 'left'
            valign: 'middle'
            bold: True
            font_size: '12sp'

        Label:
            text: root.log_text
            markup: True 
            font_size: '11sp'
            font_name: 'Roboto'
            line_height: 1.4
            pos_hint: {'x': 0.05, 'top': 0.89}
            size_hint: (0.9, 0.85)
            text_size: self.size
            halign: 'left'
            valign: 'top'

    # --- CENTER PANEL (CAMERA) ---
    FloatLayout:
        pos_hint: {'center_x': 0.55, 'center_y': 0.5}
        size_hint: (0.38, 0.65)

        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (0, 1, 1, 1) # Cyan

        Image:
            id: cam_feed
            allow_stretch: True
            keep_ratio: False
            size_hint: (0.96, 0.96)
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            canvas.before:
                Color:
                    rgba: (0, 0, 0, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [12,]

    # --- RIGHT PANEL (CONTROLS) ---
    FloatLayout:
        pos_hint: {'right': 0.98, 'center_y': 0.5}
        size_hint: (0.20, 0.65)

        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (1, 0, 0.5, 1) # Magenta

        Label:
            text: "VECTOR ANGLE"
            font_size: '10sp'
            bold: True
            color: hex('#AAAAAA')
            pos_hint: {'center_x': 0.5, 'top': 0.95}

        Label:
            text: str(int(root.current_angle)) + "°"
            font_size: '24sp'
            bold: True
            color: hex('#00FFFF')
            pos_hint: {'center_x': 0.5, 'top': 0.88}

        Label:
            text: str(int(root.volume_percent)) + "%"
            font_size: '50sp'
            bold: True
            color: hex('#FFFFFF')
            pos_hint: {'center_x': 0.5, 'center_y': 0.65}
            
        Label:
            text: "GAIN LEVEL"
            font_size: '10sp'
            bold: True
            color: hex('#AAAAAA')
            pos_hint: {'center_x': 0.5, 'center_y': 0.55}

        Widget:
            id: vol_bar_bg
            pos_hint: {'center_x': 0.5, 'center_y': 0.50}
            size_hint: (0.8, 0.06)
            canvas:
                Color:
                    rgba: (0.1, 0.1, 0.1, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [5,]
                Color:
                    rgba: (0, 1, 1, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: (self.width * (root.volume_percent / 100), self.height)
                    radius: [5,]

        Button:
            id: status_btn
            text: "MUTE" if root.volume_percent == 0 else "ACTIVE"
            font_size: '14sp'
            bold: True
            background_color: (0,0,0,0)
            color: (1, 0.2, 0.2, 1) if root.volume_percent == 0 else (0.2, 1, 0.5, 1)
            pos_hint: {'center_x': 0.5, 'y': 0.05}
            size_hint: (0.8, 0.12)
            canvas.before:
                Color:
                    rgba: (0.8, 0, 0, 0.2) if root.volume_percent == 0 else (0, 0.8, 0.2, 0.2)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10,]
                Color:
                    rgba: (1, 0, 0, 1) if root.volume_percent == 0 else (0, 1, 0, 1)
                Line:
                    width: 1.5
                    rounded_rectangle: (self.x, self.y, self.width, self.height, 10)
'''

# --- AUDIO HELPERS ---
def _create_mmdevice_enumerator():
    try:
        return CreateObject("MMDeviceEnumerator.MMDeviceEnumerator", interface=IMMDeviceEnumerator)
    except Exception:
        clsid = GUID(GUID_MMDEVICE)
        return CreateObject(clsid, interface=IMMDeviceEnumerator)

def _get_new_interface():
    # Calling internal helper
    enumerator = _create_mmdevice_enumerator()
    default_device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)
    iface = default_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))

def _percent_to_scalar(p):
    return max(0.0, min(1.0, p / 100.0))

class MainUI(FloatLayout):
    log_text = StringProperty("[color=#888888]Loading Runtime...[/color]\n")
    volume_percent = NumericProperty(0)
    current_angle = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_list = [] 
        self.smooth_vol = 0
        self.volume_interface = None
        self.prev_time = 0
        self.frame_counter = 0  
        self.hand_present_prev = False
        
        # LOGGING ACTUAL FUNCTION SIGNATURE
        self.add_log("FUNC", "def __init__(self, **kwargs):", "FF00FF")
        self.add_log("EXEC", "Clock.schedule_once(self.start_resources)", "888888")
        Clock.schedule_once(self.start_resources, 0.1)

    def start_resources(self, dt):
        self.add_log("FUNC", "def start_resources(self, dt):", "FF00FF")
        try:
            self.add_log("EXEC", "CoInitialize()", "888888")
            CoInitialize()
            
            self.add_log("EXEC", "self.volume_interface = _get_new_interface()", "888888")
            self.volume_interface = _get_new_interface()
            
            self.add_log("EXEC", "self.capture = cv2.VideoCapture(0)", "888888")
            self.capture = cv2.VideoCapture(0)
            
            if not self.capture.isOpened():
                self.add_log("ERR", "raise IOError('Camera Failed')", "FF0000")
            else:
                self.add_log("RTN", "return True", "00FF00")

            self.add_log("EXEC", "self.mp_hands = mp.solutions.hands", "888888")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.add_log("EXEC", "Clock.schedule_interval(self.update, 1/30)", "00FFFF")
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            
        except Exception as e:
            self.add_log("EXCPT", str(e), "FF0000")

    def add_log(self, tag, msg, color_hex):
        from datetime import datetime
        time_str = datetime.now().strftime("%S.%f")[:-3] 
        new_entry = f"[color=#666666]{time_str}[/color] [b][color=#{color_hex}]{tag}:[/color][/b] {msg}"
        self.log_list.insert(0, new_entry)
        if len(self.log_list) > 17: 
            self.log_list.pop()
        self.log_text = "\n".join(self.log_list)

    def update(self, dt):
        if not hasattr(self, 'capture') or not self.capture.isOpened():
            return
            
        self.frame_counter += 1
        
        # --- EXECUTION TRACE ---
        # Log the function call every 60 frames to visualize flow without lag
        should_trace = (self.frame_counter % 60 == 0) 
        
        if should_trace:
             self.add_log("FUNC", "def update(self, dt):", "FFFFFF")

        # --- FPS CALCULATION ---
        curr_time = time.time()
        fps = 0
        if self.prev_time != 0:
            fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        if should_trace: self.add_log("EXEC", "ret, frame = self.capture.read()", "888888")
        ret, frame = self.capture.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if should_trace: self.add_log("EXEC", "results = self.hands.process(rgb_frame)", "888888")
        results = self.hands.process(rgb_frame)
        
        hand_present = False

        if results.multi_hand_landmarks:
            hand_present = True
            
            if not self.hand_present_prev:
                self.add_log("IF", "if results.multi_hand_landmarks: True", "00FF00")
            
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                h, w, c = frame.shape

                # --- 1. EXTRACT POINTS ---
                x17, y17 = int(lm[17].x * w), int(lm[17].y * h)
                x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
                x8, y8 = int(lm[8].x * w), int(lm[8].y * h)

                # --- 2. VECTOR MATH ---
                if should_trace: self.add_log("EXEC", "np.dot(v1, v2) # Dot Product", "888888")
                
                v1 = np.array([x4 - x17, y4 - y17]) 
                v2 = np.array([x8 - x17, y8 - y17]) 

                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                angle_deg = 0
                if norm_v1 != 0 and norm_v2 != 0:
                    dot_product = np.dot(v1, v2)
                    cosine_angle = dot_product / (norm_v1 * norm_v2)
                    cosine_angle = max(-1.0, min(1.0, cosine_angle))
                    angle_deg = np.degrees(np.arccos(cosine_angle))

                self.current_angle = float(angle_deg)

                # --- 3. DRAW VISUALS ---
                cv2.arrowedLine(frame, (x17, y17), (x4, y4), (0, 255, 255), 4, tipLength=0.1) 
                cv2.arrowedLine(frame, (x17, y17), (x8, y8), (0, 255, 255), 4, tipLength=0.1)
                cv2.line(frame, (x4, y4), (x8, y8), (0, 255, 0), 2)
                cv2.circle(frame, (x17, y17), 10, (255, 0, 255), cv2.FILLED) 

                cv2.putText(frame, f"{int(angle_deg)} deg", (x17 + 20, y17), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # --- 4. MAP TO VOLUME ---
                target_vol = np.interp(angle_deg, [20, 80], [0, 100])
                
                # --- 5. SMOOTHING ---
                self.smooth_vol = (0.9 * self.smooth_vol) + (0.1 * target_vol)
                final_vol = 0 if self.smooth_vol < 2 else self.smooth_vol
                self.volume_percent = float(final_vol)

                # --- 6. SET VOLUME ---
                if should_trace: self.add_log("EXEC", "self.volume_interface.SetMasterVolume()", "888888")
                
                if self.volume_interface:
                    try:
                        scalar_val = _percent_to_scalar(final_vol)
                        self.volume_interface.SetMasterVolumeLevelScalar(scalar_val, None)
                        
                        is_muted = 1 if final_vol == 0 else 0
                        self.volume_interface.SetMute(is_muted, None)
                        
                        if should_trace:
                             self.add_log("DATA", f"self.current_angle = {int(angle_deg)}", "00FFFF")

                    except Exception:
                        pass
        else:
            if self.hand_present_prev:
                 self.add_log("ELSE", "else: Hand Lost", "FFAA00")

        self.hand_present_prev = hand_present

        # Draw FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Kivy Conversion
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

    def clean_up(self):
        self.add_log("FUNC", "def clean_up(self):", "FF00FF")
        if hasattr(self, 'capture'):
            self.capture.release()
        try:
            CoUninitialize()
        except:
            pass

class AirVolumeApp(App):
    def build(self):
        Window.clearcolor = (0.01, 0.01, 0.02, 1)
        Window.size = (1100, 680)
        Builder.load_string(KV)
        self.ui = MainUI()
        return self.ui

    def on_stop(self):
        if hasattr(self, 'ui'):
            self.ui.clean_up()

if __name__ == '__main__':
    AirVolumeApp().run()