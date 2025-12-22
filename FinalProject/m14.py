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
from kivy.properties import StringProperty, NumericProperty, ListProperty, ObjectProperty, BooleanProperty
from kivy.core.window import Window
from kivy.uix.spinner import Spinner

# --- OS CHECK ---
if platform.system() != "Windows":
    raise SystemExit("This script runs only on Windows.")

try:
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator, AudioUtilities, ISimpleAudioVolume
except ImportError:
    raise SystemExit("Install required packages: pip install pycaw comtypes kivy opencv-python mediapipe")

# --- CONFIG ---
eRender = 0
eCapture = 1 
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
    mode_spinner: mode_spinner_id
    device_spinner: device_spinner_id

    canvas.before:
        Color:
            rgba: hex('#020205')
        Rectangle:
            pos: self.pos
            size: self.size
        # Center Divider Line to show Zones
        Color:
            rgba: (1, 1, 1, 0.1)
        Line:
            points: (self.width * 0.58, self.height * 0.1, self.width * 0.58, self.height * 0.9)
            dash_length: 10
            dash_offset: 5

    # --- TOP HEADER ---
    Label:
        text: "AirVolume - GESTURE CONTROL CENTER"
        font_size: '32sp'
        font_name: 'Roboto'
        bold: True
        color: hex('#00FFFF')
        pos_hint: {'center_x': 0.5, 'top': 0.98}
        size_hint: (None, None)
        size: (400, 50)
        canvas.before:
            Color:
                rgba: (0, 1, 1, 0.05)
            RoundedRectangle:
                pos: (self.center_x - 200, self.y)
                size: (400, 50)
                radius: [10,]

    Label:
        text: "Team: Oviya | Rashad | Dhanush | Girija"
        font_size: '12sp'
        letter_spacing: 1.5
        color: hex('#0088AA')
        pos_hint: {'center_x': 0.5, 'top': 0.92}
        size_hint: (None, None)
        size: (300, 30)

    # --- LEFT PANEL (LOGS) ---
    FloatLayout:
        pos_hint: {'x': 0.02, 'center_y': 0.48}
        size_hint: (0.34, 0.8)
        
        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (0, 1, 0.5, 1) # Spring Green

        Label:
            text: "/// SYSTEM_LOGS"
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
            line_height: 1.3
            pos_hint: {'x': 0.05, 'top': 0.89}
            size_hint: (0.9, 0.85)
            text_size: self.size
            halign: 'left'
            valign: 'top'

    # --- CENTER PANEL (CAMERA) ---
    FloatLayout:
        pos_hint: {'center_x': 0.58, 'center_y': 0.48}
        size_hint: (0.38, 0.8)

        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (0, 1, 1, 1) # Cyan

        Image:
            id: cam_feed
            allow_stretch: True
            keep_ratio: False
            size_hint: (0.94, 0.94)
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            canvas.before:
                Color:
                    rgba: (0, 0, 0, 0.8)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10,]

        # --- LOCK INDICATOR OVERLAY ---
        Label:
            text: "LOCKED" if root.is_locked else "UNLOCKED"
            font_size: '14sp'
            bold: True
            color: (1, 0, 0, 1) if root.is_locked else (0, 1, 0, 1)
            pos_hint: {'right': 0.95, 'top': 0.95}
            size_hint: (None, None)
            size: (100, 30)
            canvas.before:
                Color:
                    rgba: (0, 0, 0, 0.6)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [5,]
                Color:
                    rgba: (1, 0, 0, 1) if root.is_locked else (0, 1, 0, 1)
                Line:
                    width: 1.5
                    rounded_rectangle: (self.x, self.y, self.width, self.height, 5)

        # --- ZONE INDICATORS ---
        Label:
            text: "LOCK ZONE (LEFT)"
            font_size: '10sp'
            color: (1, 1, 1, 0.3)
            pos_hint: {'x': 0.05, 'bottom': 0.05}
        
        Label:
            text: "VOLUME ZONE (RIGHT)"
            font_size: '10sp'
            color: (1, 1, 1, 0.3)
            pos_hint: {'right': 0.95, 'bottom': 0.05}

    # --- RIGHT PANEL (CONTROLS) ---
    FloatLayout:
        pos_hint: {'right': 0.98, 'center_y': 0.48}
        size_hint: (0.18, 0.8)

        CurvedNeonBox:
            pos_hint: {'x': 0, 'y': 0}
            size_hint: (1, 1)
            glow_color: (1, 0, 0.5, 1) # Magenta

        BoxLayout:
            orientation: 'vertical'
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            size_hint: (0.9, 0.95)
            spacing: 5
            padding: 5

            Label:
                text: "SETTINGS"
                font_size: '14sp'
                bold: True
                color: hex('#FF00AA')
                size_hint_y: 0.05
                canvas.before:
                    Color:
                        rgba: (1, 0, 0.5, 0.1)
                    Rectangle:
                        pos: self.pos
                        size: self.size

            Label:
                text: "AUDIO DEVICE"
                font_size: '10sp'
                bold: True
                color: hex('#AAAAAA')
                size_hint_y: 0.05
                valign: 'bottom'

            Spinner:
                id: device_spinner_id
                text: 'Output (Speakers)'
                values: ['Output (Speakers)', 'Input (Microphone)']
                size_hint_y: 0.1
                background_normal: ''
                background_color: (0.1, 0.1, 0.2, 1)
                color: (0, 1, 0.8, 1)
                font_size: '11sp'
                bold: True
                on_text: root.on_device_change(self.text)
                canvas.after:
                    Color:
                        rgba: (0, 1, 1, 0.3)
                    Line:
                        width: 1
                        rounded_rectangle: (self.x, self.y, self.width, self.height, 5)

            Label:
                text: "TARGET APP"
                font_size: '10sp'
                bold: True
                color: hex('#AAAAAA')
                size_hint_y: 0.05
                valign: 'bottom'

            Spinner:
                id: mode_spinner_id
                text: 'Master Volume'
                values: ['Master Volume', 'chrome.exe', 'Crisp.exe', 'Discord.exe']
                size_hint_y: 0.1
                background_normal: ''
                background_color: (0.1, 0.1, 0.2, 1)
                color: (1, 0, 1, 1)
                font_size: '11sp'
                bold: True
                canvas.after:
                    Color:
                        rgba: (1, 0, 1, 0.3)
                    Line:
                        width: 1
                        rounded_rectangle: (self.x, self.y, self.width, self.height, 5)

            Widget:
                size_hint_y: 0.05

            Label:
                text: "GESTURE ANGLE"
                font_size: '9sp'
                color: hex('#AAAAAA')
                size_hint_y: 0.05

            Label:
                text: str(int(root.current_angle)) + "°"
                font_size: '22sp'
                bold: True
                color: hex('#00FFFF')
                size_hint_y: 0.08

            Label:
                text: str(int(root.volume_percent)) + "%"
                font_size: '48sp'
                bold: True
                color: hex('#FFFFFF')
                size_hint_y: 0.15
                
            Label:
                text: "GAIN LEVEL"
                font_size: '9sp'
                bold: True
                color: hex('#AAAAAA')
                size_hint_y: 0.05

            Widget:
                id: vol_bar_bg
                size_hint_y: 0.05
                canvas:
                    Color:
                        rgba: (0.15, 0.15, 0.15, 1)
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [3,]
                    Color:
                        rgba: (0, 1, 0.5, 1)
                    RoundedRectangle:
                        pos: self.pos
                        size: (self.width * (root.volume_percent / 100), self.height)
                        radius: [3,]

            Widget:
                size_hint_y: 0.05

            Button:
                id: status_btn
                text: "LOCKED" if root.is_locked else "ACTIVE"
                font_size: '12sp'
                bold: True
                background_color: (0,0,0,0)
                color: (1, 0.2, 0.2, 1) if root.is_locked else (0.2, 1, 0.5, 1)
                size_hint_y: 0.12
                canvas.before:
                    Color:
                        rgba: (0.8, 0, 0, 0.15) if root.is_locked else (0, 0.8, 0.2, 0.15)
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [6,]
                    Color:
                        rgba: (1, 0, 0, 0.8) if root.is_locked else (0, 1, 0, 0.8)
                    Line:
                        width: 1
                        rounded_rectangle: (self.x, self.y, self.width, self.height, 6)
'''

# --- AUDIO HELPERS ---
def _create_mmdevice_enumerator():
    try:
        return CreateObject("MMDeviceEnumerator.MMDeviceEnumerator", interface=IMMDeviceEnumerator)
    except Exception:
        clsid = GUID(GUID_MMDEVICE)
        return CreateObject(clsid, interface=IMMDeviceEnumerator)

def _get_new_interface(data_flow_type):
    enumerator = _create_mmdevice_enumerator()
    default_device = enumerator.GetDefaultAudioEndpoint(data_flow_type, eConsole)
    iface = default_device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))

def _percent_to_scalar(p):
    return max(0.0, min(1.0, p / 100.0))

class MainUI(FloatLayout):
    log_text = StringProperty("[color=#888888]Initializing AI Core...[/color]\n")
    volume_percent = NumericProperty(0)
    current_angle = NumericProperty(0)
    is_locked = BooleanProperty(False) 
    mode_spinner = ObjectProperty(None)
    device_spinner = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_list = [] 
        self.smooth_vol = 0 
        self.volume_interface = None
        self.current_data_flow = eRender 
        self.prev_time = 0
        self.frame_counter = 0  
        self.hand_present_prev = False
        
        self.add_log("SYS", "UI Loaded successfully.", "FFFFFF")
        Clock.schedule_once(self.start_resources, 0.1)

    def start_resources(self, dt):
        try:
            self.add_log("INIT", "Connecting to Audio API...", "00FFFF")
            CoInitialize()
            self.volume_interface = _get_new_interface(self.current_data_flow)
            self.add_log("OK", "Audio Endpoint Connected.", "00FF00")
            
            self.add_log("INIT", "Starting Computer Vision...", "00FFFF")
            self.capture = cv2.VideoCapture(0)
            
            if not self.capture.isOpened():
                self.add_log("ERR", "Camera Not Found!", "FF0000")
            else:
                self.add_log("OK", "Camera Active.", "00FF00")

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            
        except Exception as e:
            self.add_log("CRIT", str(e), "FF0000")

    def on_device_change(self, text):
        try:
            if "Speakers" in text:
                self.current_data_flow = eRender 
                self.add_log("CFG", "Switched to: SPEAKERS", "FFAA00")
            else:
                self.current_data_flow = eCapture 
                self.add_log("CFG", "Switched to: MICROPHONE", "FFAA00")
            self.volume_interface = _get_new_interface(self.current_data_flow)
        except Exception as e:
            self.add_log("ERR", f"Device Switch Failed: {e}", "FF0000")

    def add_log(self, tag, msg, color_hex):
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        new_entry = f"[color=#555555]{time_str}[/color] [b][color=#{color_hex}]{tag}:[/color][/b] {msg}"
        self.log_list.insert(0, new_entry)
        if len(self.log_list) > 18: 
            self.log_list.pop()
        self.log_text = "\n".join(self.log_list)

    def set_application_volume(self, app_name, volume_scalar):
        sessions = AudioUtilities.GetAllSessions()
        found = False
        for session in sessions:
            if session.Process and session.Process.name() == app_name:
                volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                volume.SetMasterVolume(volume_scalar, None)
                found = True
        return found

    def update(self, dt):
        if not hasattr(self, 'capture') or not self.capture.isOpened():
            return
            
        self.frame_counter += 1
        curr_time = time.time()
        fps = 0
        if self.prev_time != 0:
            fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        ret, frame = self.capture.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- FIX: DEFINE DIMENSIONS HERE SO THEY EXIST EVEN IF NO HANDS ARE SEEN ---
        h, w, c = frame.shape
        
        results = self.hands.process(rgb_frame)
        
        hand_present = False

        if results.multi_hand_landmarks:
            hand_present = True
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                hand_label = handedness.classification[0].label 
                lm = hand_landmarks.landmark
                
                # Get Wrist X-Coordinate (0.0 = Left, 1.0 = Right)
                wrist_x = lm[0].x

                # --- 1. LOCKING ZONE (LEFT HAND MUST BE ON LEFT SIDE) ---
                # Rule: Check 'Left' Label AND Position < 0.55 (Left Half + small margin)
                if hand_label == 'Left' and wrist_x < 0.55:
                    
                    thumb_tip_y = lm[4].y
                    thumb_ip_y = lm[3].y # IP joint
                    
                    # Thumbs UP (Unlock)
                    if thumb_tip_y < thumb_ip_y:
                        if self.is_locked:
                            self.is_locked = False
                            self.add_log("CMD", "UNLOCKED (Left Zone)", "00FF00")
                            
                    # Thumbs DOWN (Lock)
                    elif thumb_tip_y > thumb_ip_y:
                        if not self.is_locked:
                            self.is_locked = True
                            self.add_log("CMD", "LOCKED (Left Zone)", "FF0000")

                    # Draw Visual
                    status_text = "LOCKED" if self.is_locked else "FREE"
                    color = (0, 0, 255) if self.is_locked else (0, 255, 0)
                    cv2.putText(frame, f"ZONE-L: {status_text}", (int(lm[4].x*w), int(lm[4].y*h)-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 2. VOLUME ZONE (RIGHT HAND LOGIC) ---
                # We process volume if label is 'Right' OR if it's on the Right Side (fail-safe)
                elif hand_label == 'Right': 
                    
                    # FAIL-SAFE: If this "Right Hand" is actually on the far Left side, IGNORE IT
                    if wrist_x < 0.4:
                        continue 

                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if not self.is_locked:
                        x17, y17 = int(lm[17].x * w), int(lm[17].y * h)
                        x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
                        x8, y8 = int(lm[8].x * w), int(lm[8].y * h)

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

                        cv2.arrowedLine(frame, (x17, y17), (x4, y4), (0, 255, 255), 4, tipLength=0.1) 
                        cv2.arrowedLine(frame, (x17, y17), (x8, y8), (0, 255, 255), 4, tipLength=0.1)

                        target_vol = np.interp(angle_deg, [20, 80], [0, 100])
                        self.smooth_vol = (0.9 * self.smooth_vol) + (0.1 * target_vol)
                        final_vol = 0 if self.smooth_vol < 2 else self.smooth_vol
                        self.volume_percent = float(final_vol)

                        current_app_mode = self.mode_spinner.text 
                        scalar_val = _percent_to_scalar(final_vol)
                        
                        if current_app_mode == "Master Volume":
                            if self.volume_interface:
                                try:
                                    self.volume_interface.SetMasterVolumeLevelScalar(scalar_val, None)
                                    is_muted = 1 if final_vol == 0 else 0
                                    self.volume_interface.SetMute(is_muted, None)
                                except Exception: pass
                        else:
                            try:
                                self.set_application_volume(current_app_mode, scalar_val)
                            except Exception: pass
                    else:
                        cx, cy = int(lm[9].x * w), int(lm[9].y * h)
                        cv2.putText(frame, "LOCKED", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            if self.hand_present_prev:
                 pass 

        self.hand_present_prev = hand_present

        # Draw FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw Zone Divider Line
        cv2.line(frame, (int(w*0.5), 0), (int(w*0.5), h), (255, 255, 255), 1)

        # Kivy Conversion
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

    def clean_up(self):
        self.add_log("SYS", "Shutting down...", "FF00FF")
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