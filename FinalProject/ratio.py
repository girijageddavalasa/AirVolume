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
from kivy.uix.boxlayout import BoxLayout
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
        # Glow Shadow
        Color:
            rgba: (self.glow_color[0], self.glow_color[1], self.glow_color[2], 0.15)
        RoundedRectangle:
            pos: (self.x - 2, self.y - 2)
            size: (self.width + 4, self.height + 4)
            radius: [15,]
        
        # Main Border
        Color:
            rgba: self.glow_color
        Line:
            width: 1.2
            rounded_rectangle: (self.x, self.y, self.width, self.height, 15)

<MainUI>:
    canvas.before:
        Color:
            rgba: hex('#020205') # Ultra Dark Background
        Rectangle:
            pos: self.pos
            size: self.size
        # Subtle Grid Background
        Color:
            rgba: (1, 1, 1, 0.03)
        Line:
            points: (0, self.height*0.3, self.width, self.height*0.3)
        Line:
            points: (0, self.height*0.7, self.width, self.height*0.7)

    # --- TITLE ---
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
        text: "Pinch Distance Analysis"
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

        # HEADER
        Label:
            text: "/// EXECUTION_LOG"
            color: hex('#00FF88')
            pos_hint: {'x': 0.05, 'top': 0.96}
            size_hint: (0.9, 0.05)
            text_size: self.size
            halign: 'left'
            valign: 'middle'
            bold: True
            font_size: '12sp'

        # LOG LIST
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

        # --- SECTION 1: STATUS (TOP) ---
        Label:
            text: "DISTANCE RATIO"
            font_size: '10sp'
            bold: True
            color: hex('#AAAAAA')
            pos_hint: {'center_x': 0.5, 'top': 0.95}

        Label:
            text: str(round(root.current_ratio, 2))
            font_size: '24sp'
            bold: True
            color: hex('#00FFFF')
            pos_hint: {'center_x': 0.5, 'top': 0.88}

        # --- SECTION 2: PERCENTAGE (MIDDLE-HIGH) ---
        Label:
            text: str(int(root.volume_percent)) + "%"
            font_size: '50sp'
            bold: True
            color: hex('#FFFFFF')
            pos_hint: {'center_x': 0.5, 'center_y': 0.65}
            
        # --- SECTION 3: BAR (MIDDLE-LOW) ---
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

        # --- SECTION 4: BUTTON (BOTTOM) ---
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
    enumerator = _create_mmdevice_enumerator()
    default_device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)
    iface = default_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))

def _percent_to_scalar(p):
    return max(0.0, min(1.0, p / 100.0))

class MainUI(FloatLayout):
    log_text = StringProperty("[color=#888888]Booting System...[/color]\n")
    volume_percent = NumericProperty(0)
    current_ratio = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_list = [] 
        self.smooth_vol = 0 # To smooth out jitter
        self.volume_interface = None
        self.prev_time = 0
        
        # Schedule the loading process to run after the UI builds
        Clock.schedule_once(self.start_resources, 0.1)

    def start_resources(self, dt):
        """
        Loads resources step-by-step and logs the specific calls being made.
        """
        try:
            # --- STEP 1: COM LIBRARY ---
            self.add_log("SYS", "Calling CoInitialize()", "BBBBBB")
            CoInitialize()
            
            # --- STEP 2: AUDIO INTERFACE ---
            self.add_log("AUDIO", "Init: IMMDeviceEnumerator", "00FFFF")
            self.add_log("AUDIO", "Call: GetDefaultAudioEndpoint()", "00AAAA")
            
            try:
                self.volume_interface = _get_new_interface()
                self.add_log("SUCCESS", "Audio Interface Connected", "00FF00")
            except Exception as audio_e:
                self.add_log("ERROR", f"Audio Fail: {str(audio_e)}", "FF0000")

            # --- STEP 3: CAMERA ---
            self.add_log("VIDEO", "Call: cv2.VideoCapture(0)", "FFAA00")
            self.capture = cv2.VideoCapture(0)
            
            if not self.capture.isOpened():
                self.add_log("CRITICAL", "Camera Access Denied/Null", "FF0000")
            else:
                w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.add_log("SUCCESS", f"Stream Buffer: {w}x{h}", "00FF00")

            # --- STEP 4: MEDIAPIPE AI ---
            self.add_log("AI_CORE", "Import: mp.solutions.hands", "FF00FF")
            self.mp_hands = mp.solutions.hands
            
            self.add_log("AI_CORE", "Init: Hands(min_conf=0.7)", "DD00DD")
            self.hands = self.mp_hands.Hands(
                max_num_hands=1, 
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.add_log("SUCCESS", "Gesture Engine Ready", "00FF00")
            
            # --- STEP 5: START LOOP ---
            self.add_log("KERNEL", "Sched: Clock.interval(1/30s)", "FFFFFF")
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            
        except Exception as e:
            self.add_log("FATAL", str(e), "FF0000")

    def add_log(self, tag, msg, color_hex):
        """ Adds a functional log entry """
        from datetime import datetime
        time_str = datetime.now().strftime("%S.%f")[:-4] 
        # Format: [Time] [TAG]: Message
        new_entry = f"[color=#555555]{time_str}[/color] [b][color=#{color_hex}]{tag}:[/color][/b] {msg}"
        
        self.log_list.insert(0, new_entry)
        if len(self.log_list) > 16: # Keep buffer small
            self.log_list.pop()
        self.log_text = "\n".join(self.log_list)

    def update(self, dt):
        if not hasattr(self, 'capture') or not self.capture.isOpened():
            return
            
        # --- FPS CALCULATION ---
        curr_time = time.time()
        fps = 0
        if self.prev_time != 0:
            fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        ret, frame = self.capture.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get raw landmarks list
                lm = hand_landmarks.landmark
                h, w, c = frame.shape

                # --- 1. GET COORDINATES ---
                # Tip of Index (8)
                x1, y1 = int(lm[8].x * w), int(lm[8].y * h)
                # Tip of Thumb (4)
                x2, y2 = int(lm[4].x * w), int(lm[4].y * h)
                
                # Wrist (0)
                x3, y3 = int(lm[0].x * w), int(lm[0].y * h)
                # Pinky Knuckle (17)
                x4, y4 = int(lm[17].x * w), int(lm[17].y * h)

                # --- 2. DRAW LINES ---
                # Draw the control line (Index to Thumb)
                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Draw the reference line (Wrist to Pinky) - Visual Debug
                cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

                # --- 3. CALCULATE MATH ---
                # Distance between Index and Thumb
                length_active = math.hypot(x2 - x1, y2 - y1)
                
                # Distance between Wrist and Pinky Knuckle (Reference)
                length_ref = math.hypot(x4 - x3, y4 - y3)

                if length_ref == 0: length_ref = 1 # Safety division

                # Calculate Ratio (Normalizes Z-Axis depth)
                # Usually: Ratio is ~0.2 when pinching, ~1.2 when fully open
                ratio = length_active / length_ref
                self.current_ratio = float(ratio)

                # Map ratio to 0-100 Volume
                # We assume ratio < 0.3 is 0%, ratio > 1.3 is 100%
                target_vol = np.interp(ratio, [0.3, 1.3], [0, 100])
                
                # --- 4. SMOOTHING (Weighted Average) ---
                # This prevents the slider from jittering
                self.smooth_vol = (0.8 * self.smooth_vol) + (0.2 * target_vol)
                
                # Snap to 0 if very low to allow full mute
                final_vol = 0 if self.smooth_vol < 2 else self.smooth_vol
                
                # FIXED: Convert to float to avoid Kivy ValueError
                self.volume_percent = float(final_vol)

                # --- 5. SET SYSTEM VOLUME ---
                if self.volume_interface:
                    try:
                        scalar_val = _percent_to_scalar(final_vol)
                        self.volume_interface.SetMasterVolumeLevelScalar(scalar_val, None)
                        
                        # Manage Mute State
                        if final_vol == 0:
                            self.volume_interface.SetMute(1, None)
                        else:
                            self.volume_interface.SetMute(0, None)
                            
                    except Exception as e:
                        print(f"Driver Error: {e}")

        # --- DRAW FPS ON FRAME ---
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Kivy Texture Conversion
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

    def clean_up(self):
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