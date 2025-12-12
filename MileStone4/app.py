import sys
import platform
import cv2
import time
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

<Bubble@Widget>:
    active: False
    canvas:
        Color:
            rgba: (0, 1, 1, 1) if self.active else (0.15, 0.15, 0.2, 1)
        Ellipse:
            pos: self.pos
            size: self.size
        Color:
            rgba: (0, 1, 1, 0.6) if self.active else (0, 0, 0, 0)
        Line:
            circle: (self.center_x, self.center_y, self.width/2 + 2)
            width: 2

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
        text: "Geometric Angle Analysis"
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

        # --- SECTION 1: FINGERS (TOP) ---
        Label:
            text: "ACTIVE FINGERS"
            font_size: '10sp'
            bold: True
            color: hex('#AAAAAA')
            pos_hint: {'center_x': 0.5, 'top': 0.95}

        BoxLayout:
            id: bubbles_container
            orientation: 'horizontal'
            spacing: 5
            padding: 5
            pos_hint: {'center_x': 0.5, 'top': 0.90}
            size_hint: (0.9, 0.08)
            
            Bubble:
                active: root.finger_count >= 1
            Bubble:
                active: root.finger_count >= 2
            Bubble:
                active: root.finger_count >= 3
            Bubble:
                active: root.finger_count >= 4
            Bubble:
                active: root.finger_count >= 5

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
            text: "MUTE" if root.volume_percent == 0 else "MIC"
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
    log_text = StringProperty("[color=#888888]Allocating Memory...[/color]\n")
    volume_percent = NumericProperty(0)
    finger_count = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_list = [] 
        self.prev_fingers = -1
        self.volume_interface = None
        self.prev_time = 0
        Clock.schedule_once(self.start_resources, 0.1)

    def start_resources(self, dt):
        try:
            self.add_log("INIT", "Calling CoInitialize()...", "888888")
            CoInitialize()
            
            self.add_log("FUNC", "_get_new_interface() -> IAudioEndpointVolume", "CCCCCC")
            self.volume_interface = _get_new_interface()
            
            self.add_log("FUNC", "cv2.VideoCapture(0) -> Initializing Stream", "CCCCCC")
            self.capture = cv2.VideoCapture(0)
            
            if not self.capture.isOpened():
                self.add_log("EXCEPTION", "Camera Init Failed (IOError)", "FF0000")
            else:
                self.add_log("SUCCESS", "Stream Buffer Allocated", "00FF00")

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Styles
            self.draw_spec_lines = self.mp_drawing.DrawingSpec(color=(139, 0, 0), thickness=3) 
            self.draw_spec_points = self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=4, circle_radius=3)
            self.draw_spec_mute = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)

            self.add_log("MAIN_LOOP", "Clock.schedule_interval(update, 1/30)", "00FFFF")
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            
        except Exception as e:
            self.add_log("CRITICAL", str(e), "FF0000")

    def add_log(self, tag, msg, color_hex):
        """ Adds a functional log entry """
        from datetime import datetime
        time = datetime.now().strftime("%S.%f")[:-3] 
        new_entry = f"[color=#666666]{time}[/color] [b][color=#{color_hex}]{tag}:[/color][/b] {msg}"
        
        self.log_list.insert(0, new_entry)
        if len(self.log_list) > 14: # Keep buffer small
            self.log_list.pop()
        self.log_text = "\n".join(self.log_list)

    # --- INNOVATION: ANGLE-BASED LOGIC ---
    def count_fingers_logic(self, lm):
        cnt = 0

        # Helper: Calculate angle between 3 points (A, B, C) where B is the joint
        def get_angle(a, b, c):
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine_angle))
            return angle

        # 1. THUMB ANGLE CHECK (Fixed for easy bending)
        # We calculate the angle at the IP Joint (Landmark 3)
        # Using Points: MCP(2) -> IP(3) -> TIP(4)
        thumb_angle = get_angle(lm[2], lm[3], lm[4])
        
        # If angle is > 150 degrees, it's straight (OPEN).
        # If it bends even slightly (< 150), it counts as closed.
        if thumb_angle > 150:
            cnt += 1

        # 2. FINGER ANGLE CHECK
        finger_joints = [
            (5, 6, 8),    # Index: MCP, PIP, TIP
            (9, 10, 12),  # Middle
            (13, 14, 16), # Ring
            (17, 18, 20)  # Pinky
        ]

        for mcp, pip, tip in finger_joints:
            angle = get_angle(lm[mcp], lm[pip], lm[tip])
            
            # Threshold: > 140 degrees is OPEN
            if angle > 140:
                cnt += 1
                
        return cnt

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

        current_fingers = 0
        detected_hands = 0

        if results.multi_hand_landmarks:
            detected_hands = len(results.multi_hand_landmarks)
            
            # --- DOUBLE HAND = HARD MUTE ---
            if detected_hands == 2:
                current_fingers = 0 
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.draw_spec_mute, self.draw_spec_mute
                    )
                cv2.putText(frame, "!! HARD MUTE !!", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            else:
                # --- SINGLE HAND = VOLUME ---
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.draw_spec_points, self.draw_spec_lines
                    )
                    lm = hand_landmarks.landmark
                    
                    # USE NEW ANGLE LOGIC
                    current_fingers = self.count_fingers_logic(lm)

        # --- DRAW FPS ON FRAME ---
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- LOGIC BRANCHING & AUDIO CALLS ---
        if current_fingers != self.prev_fingers or detected_hands == 2:
            self.finger_count = current_fingers
            self.volume_percent = current_fingers * 20
            
            if self.volume_interface:
                try:
                    # BRANCH 1: HARD MUTE
                    if detected_hands == 2:
                        if self.prev_fingers != -999:
                             self.add_log("LOGIC", f"detected_hands={detected_hands} >> Mute Override", "FF00CC")
                             self.add_log("DRIVER", "SetMute(1, None) -> INVOKED", "FF5555")
                             self.volume_interface.SetMute(1, None) 
                             self.volume_interface.SetMasterVolumeLevelScalar(0.0, None)
                             self.prev_fingers = -999
                    
                    # BRANCH 2: VOLUME ADJUST
                    else:
                        scalar_val = _percent_to_scalar(self.volume_percent)
                        
                        # Only log if state changed from Mute or value changed
                        if self.prev_fingers == -999: 
                            self.add_log("DRIVER", "SetMute(0, None) -> RESTORING", "00FF00")
                            self.volume_interface.SetMute(0, None)

                        if self.prev_fingers != current_fingers:
                            self.add_log("LOGIC", f"count_fingers_logic() returned {current_fingers}", "CCCCCC")
                            self.add_log("DRIVER", f"SetMasterVolumeLevelScalar({scalar_val:.1f})", "00FFFF")
                            self.volume_interface.SetMasterVolumeLevelScalar(scalar_val, None)
                            self.prev_fingers = current_fingers

                except Exception as e:
                    print(f"Volume Error: {e}")

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