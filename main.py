import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import yaml
from tkinter import filedialog, messagebox
from extract_skeleton import mediapipe
import math
import threading
from pose2video import pose2video
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from movement_calculation import movement
import video_synthesis

gray_color = '#3c3c3c'
background_color = '#464a4d'
sidebar_color = '#333739'
button_color = '#bebebe'
button_activate = '#646464'
gold_color = "#FFC733"

min_canvas_width = 1300
min_canvas_height = 750

face_connections = frozenset({(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)})

connections = [(0,1),(0,2),(0,4),(1,5),(1,3),(4,5),(3,6),(2,27),(6,7),(7,8),
          (8,9),(9,10),(6,11),(11,12),(12,13),(13,14),(11,15),(15,19),(19,23),(23,6),
          (15,16),(16,17),(17,18),(19,20),(20,21),(21,22),(23,24),(24,25),(25,26),
          (27,28),(27,32),(27,44),(28,29),(29,30),(30,31),(32,33),(33,34),(34,35),
          (36,37),(37,38),(38,39),(40,41),(41,42),(42,43),(44,45),(45,46),(46,47),
          (44,40),(32,36),(36,40)]

lines_color = [(240,90,17),(240,90,17),(242,155,17),(242,155,17),(242,55,17),(219,154,64),(219,103,64),(219,103,64),(100,64,219),(100,64,219),(100,64,219),(100,64,219),(100,64,219),(149,64,227),(149,64,227),(149,64,227),
               (149,64,227),(149,64,227),(149,64,227),(217,59,138),(203,64,227),(203,64,227),(203,64,227),(227,64,114),(227,64,114),(227,64,114),(217,59,88),(217,59,88),(217,59,88),(142,227,82),(91,227,82),
               (53,119,232),(142,227,82),(142,227,82),(142,227,82),(53,232,187),(53,232,187),(53,232,187),(48,191,199),(48,191,199),(48,191,199),(48,141,199),(48,141,199),(48,141,199),(48,93,199),(48,93,199),(48,93,199),(48,93,199),
               (53,232,187),(48,191,199)]

mark_colors = [
    "#FF5733",  # Red-Orange
    "#33FF57",  # Lime Green
    "#3357FF",  # Bright Blue
    "#FF33A1",  # Pink
    "#33FFF6",  # Cyan
    "#8D33FF",  # Purple
    "#FF3380",  # Rose
    "#33FF96",  # Mint Green
    "#FF8C33",  # Light Orange
    "#3385FF",  # Light Blue
    "#FF33C4",  # Magenta
    "#33FFA5",  # Sea Green
    "#A633FF",  # Violet
    "#FF3357",  # Red-Pink
    "#33FFD7",  # Aqua
    "#FFEE33",  # Sunflower
    "#6933FF",  # Indigo
    "#FF3366",  # Cherry
]

titles = {'0':'left shoulder', '1': 'right shouldr', '2':'left elbow', '3':'right elbow', '4':'left hip', '5':'right hip', '6':'wrist',
         '7':'thumb cmc', '8': 'thumb mcp', '9':'thumb ip', '10':'thumb tip', '11':'index finger mcp', '12':'index finger pip', '13':'index finger dip', '14':'index finger tip', '15':'wmiddle finger mcp',
         '16':'middle finger pip', '17': 'middle finger dip', '18':'middle finger tip', '19':'ring finger mcp', '20':'ring finger pip', '21':'ring finger dip', '22':'ring finger tip', '23':'pinky mcp', '24':'pinky pip',
         '25':'pinky dip', '26': 'pinky tip', '27':'wrist','28':'thumb cmc', '29': 'thumb mcp', '30':'thumb ip', '31':'thumb tip', '32':'index finger mcp', '33':'index finger pip', '34':'index finger dip', '35':'index finger tip', '36':'wmiddle finger mcp',
         '37':'middle finger pip', '38': 'middle finger dip', '39':'middle finger tip', '40':'ring finger mcp', '41':'ring finger pip', '42':'ring finger dip', '43':'ring finger tip', '44':'pinky mcp', '45':'pinky pip',
         '46':'pinky dip', '47': 'pinky tip'}

def _normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width,
    image_height):
  
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def _pixel_coordinates_to_normalized(pixel_x, pixel_y, image_width, image_height):
    """
    Converts pixel coordinates to normalized value pair.

    Args:
        pixel_x (int): x-coordinate in pixels.
        pixel_y (int): y-coordinate in pixels.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        tuple: Normalized coordinates (normalized_x, normalized_y).
    """
    normalized_x = pixel_x / image_width
    normalized_y = pixel_y / image_height
    return normalized_x, normalized_y

def hex_to_bgr(hex_color):
    """Convert a hex color string to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr


class PoseEditor(tk.Tk):
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Editor")
        self.root.configure(bg = background_color)  # Set the background color to gray

        self.style = ttk.Style()
        #self.style.configure("TButton", padding=6, relief="flat", background="#2b3d54")
        self.style.configure("TButton",
                padding=0,
                relief="flat",
                background = background_color,   # Button background color
                foreground = "#000000",   # Button text color
                bordercolor = gray_color,  # Button border color
                highlightthickness=0)          # Highlight border thickness)     # Button font)   # Focus border color

        # Load the YAML file
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        # Set up a canvas with the desired resolution
        self.frame_strip_height = 100
        self.hand_landmark_guideline_width = 400
        self.hand_landmark_guideline_height = 200
        self.pose_landmark_guideline_width = 400
        self.pose_landmark_guideline_height = 200
        self.canvas_width = self.config['canvas_width']
        self.canvas_height = self.config['canvas_height']
        self.sidebar_width = self.config['sidebar_width']
        self.sidebar_button_width = 8
        self.sidebar_button_height = 1
        self.frame_width = int(self.canvas_width / 10)  # Width of each frame in the strip
        self.num_displayed_frames = 10  # Number of frames to display in the strip
        self.scroll_increment = 2  # Scroll increment for frame strip
        self.selected_frame_index = None
        self.video = None
        self.current_frame_offset = 0  # Offset for the currently displayed frames
        self.capture_radius = 8
        self.tooltip = None
        self.recording = False
        self.out = None
        self.recording_thread = None
        self.recorded_video_path = self.config['recorded_video_path']
        self.output_video_path = os.path.join(self.config['synthetic_video_file_path'],'output_video.mp4')
        self.ref_img_path = os.path.join(self.config['synthetic_video_file_path'],'image.png')
        self.synthetic_video_path = os.path.join(self.config['synthetic_video_file_path'],'synthetic_video.mp4')
        self.pose_visible = False
        self.pose_mode = False
        self.right_hand_ref_visible = False
        self.left_hand_ref_visible = False
        self.adding_pose_flag = False
        self.adding_rh_flag = False
        self.adding_lh_flag = False
        self.escape_new_keypoint = False
        self.eraser_mode = False
        self.current_frame = None
        self.is_playing = False
        self.slider = None
        self.frame_time_frame = None
        self.frame_label = None
        self.time_label = None
        self.video_synthesis_flag = True
        self.general_mode = 'frame level'
        self.mode = 'replacing keypoint'
        self.face_skeleton = {}
        self.skeleton = {}
        self.skeleton_px = {}
        self.keypoints = []
        self.keypoint_positions = {}
        self.face_keypoints = []
        self.lines = []
        self.face_lines= []
        self.video_frames = []
        self.frame_buttons = []
        self.frame_bottom_info = {}
        self.ref_texts = []
        self.ref_lines = []
        self.ref_keypoints = []
        self.ref_right_hand_keypoints = []
        self.ref_right_hand_lines = []
        self.ref_right_hand_texts = []
        self.ref_left_hand_keypoints = []
        self.ref_left_hand_lines = []
        self.ref_left_hand_texts = []
        self.button_images = []
        self.segment_mode_buttons = []
        self.marked_positions = []  # Store marked positions
        self.mark_lines = []
        self.mark_line_ids = {}
        self.selected_marks = []
        self.selected_marks_line = []
        self.mark_buttons = []
        self.pose_estimator = True
        self.drag_data = {"x": 0, "y": 0, "frame_idx":0, "keypoint_idx":0,"item": None}
        # Create a frame to contain the canvas and scrollbars
        self.main_frame = ttk.Frame(root, style = 'TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Set up the canvas for displaying frames
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg = background_color, highlightbackground='#555555', highlightthickness=2)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.update_canvas)

        self.frame_mode_canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg = background_color, highlightbackground='#555555', highlightthickness=2)
        self.frame_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.frame_mode_canvas.bind("<Configure>", self.update_frame_mode_canvas)
        self.frame_mode_canvas.pack_forget()

        self.segment_mode_canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg = background_color, highlightbackground='#555555', highlightthickness=2)
        self.segment_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.segment_mode_canvas.pack_forget()

        # Set up a frame strip to display video frames
        self.frame_strip = ttk.Frame(self.frame_mode_canvas)
        self.frame_mode_canvas.create_window((0, self.canvas_height - self.frame_strip_height), window=self.frame_strip, anchor=tk.NW)

        self.sidebar = tk.Frame(self.main_frame, width=self.sidebar_width, bg=sidebar_color)
        self.sidebar.pack_propagate(False)  # Prevent sidebar from resizing
        self.segment_mode_sidebar = tk.Frame(self.main_frame, width=self.sidebar_width, bg=sidebar_color)

        self.frame_mode_canvas.bind("<Motion>", self.on_mouse_move)
        self.frame_mode_canvas.bind("<Leave>", self.hide_tooltip)
        
        self.frame_mode_canvas.bind("<Button-1>", self.on_keypoint_press)
        self.frame_mode_canvas.bind("<B1-Motion>", self.on_keypoint_motion)
        self.frame_mode_canvas.bind("<ButtonRelease-1>", self.on_keypoint_release)

        self.root.bind("<Escape>", self.cancel_action)

        self.dragging_keypoint = None

        self.load_logo()
        self.load_image_button()

    def load_logo(self):
        load_logo = Image.open("D:/uzh_project/interface/images/logo.png")  # Replace with your image path
        self.logo = ImageTk.PhotoImage(load_logo)
        self.logo_first = self.canvas.create_image(self.canvas_width/2, self.canvas_height * 0.8, image=self.logo, anchor=tk.CENTER)

    def load_image_button(self):
        # Load the image
        load_img1 = Image.open("D:/uzh_project/interface/images/upload-video.png")  # Replace with your image path
        load_img2 = Image.open("D:/uzh_project/interface/images/upload-pose.png")  # Replace with your image path

        img_width, img_height = load_img1.size

        self.img1 = ImageTk.PhotoImage(load_img1)
        self.img2 = ImageTk.PhotoImage(load_img2)

        # Create a canvas item for the image
        self.img_button1 = self.canvas.create_image(self.canvas_width/2 + img_width//2, self.canvas_height/2, image=self.img1, anchor=tk.CENTER)

        # Create a canvas item for the image
        self.img_button2 = self.canvas.create_image(self.canvas_width/2 - img_width//2, self.canvas_height/2, image=self.img2, anchor=tk.CENTER)

        # Bind the click event to the canvas item
        self.canvas.tag_bind(self.img_button1, '<Button-1>', self.load_video)
        self.canvas.tag_bind(self.img_button1,"<Enter>", self.on_enter)
        self.canvas.tag_bind(self.img_button1,"<Leave>", self.on_leave)

        # Bind the click event to the canvas item
        self.canvas.tag_bind(self.img_button2, '<Button-1>', self.load_pose)
        self.canvas.tag_bind(self.img_button2,"<Enter>", self.on_enter)
        self.canvas.tag_bind(self.img_button2,"<Leave>", self.on_leave)

    def update_canvas(self, event):
        load_img1 = Image.open("D:/uzh_project/interface/images/upload-video.png")  # Replace with your image path
        canvas_width = event.width
        canvas_height = event.height
        img_width, img_height = load_img1.size

        self.canvas.coords(self.logo_first, canvas_width*0.5, canvas_height*0.8)
        self.canvas.coords(self.img_button1, canvas_width/2 + img_width//2, canvas_height/2)
        self.canvas.coords(self.img_button2, canvas_width/2 - img_width//2, canvas_height/2)

    def load_video(self, event=None):
        file_path = filedialog.askopenfilename()
        self.file_path = file_path
        if file_path:
            self.video = cv2.VideoCapture(file_path)
            self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            if self.video_synthesis_flag:
                # Read the first frame from the video
                ret, frame = self.video.read()
                cv2.imwrite(self.ref_img_path, frame)
            self.select_mode()

    def load_pose(self, event=None):
        self.pose_estimator = False
        self.pose_mode = True
        file_path = filedialog.askopenfilename()
        self.pose_file_path = file_path
        saving_path = self.config['pose_video_path']
        empty_video_path = 'empty_video.mp4'
        if file_path:
            self.skeleton, self.face_skeleton = np.load(self.pose_file_path,allow_pickle=True)
            self.pose_to_video(self.skeleton, self.face_skeleton,saving_path,flag= False)
            self.create_empty_video(frame_number = len(self.skeleton), des_path= empty_video_path,width= 800, height= 600, fps= 25)
            self.select_mode()

    def pose_to_video(self, pose, face_pose, saving_path,flag):
        pose2video(pose = pose,
                face_pose = face_pose,
                saving_path = saving_path,
                width = 800,
                height = 600,
                fps = 25,
                connections = connections,
                face_connections = face_connections,
                lines_color = lines_color)
        if not flag:
            self.video = cv2.VideoCapture(saving_path)
            self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
        else:
            self.uploaded_video = cv2.VideoCapture(saving_path)

    def create_empty_video(self, frame_number, des_path,width, height, fps):
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        video_writer = cv2.VideoWriter(des_path, fourcc, fps, (width, height))
        for _ in range(frame_number):
            # Create a gray frame
            gray_frame = np.full((height, width, 3), hex_to_bgr(background_color), dtype=np.uint8)
            
            # Write the frame to the video file
            video_writer.write(gray_frame)
        video_writer.release()

        self.empty_video = cv2.VideoCapture(des_path)


    def on_enter(self, event):
        event.widget.config(cursor="hand2")  # Change the cursor to a hand

    def on_leave(self, event):
        event.widget.config(cursor="")  # Revert the cursor to the default

    def on_enter_frames(self, event, button, idx):
        button.config(cursor="hand2")  # Change cursor to hand when mouse enters button
        self.show_tooltip(event, index = idx, title='', mode = 'frames')


    def select_mode(self, event=None):
        #################### Select Mode ####################
        self.canvas.delete(self.img_button1)
        self.canvas.delete(self.img_button2)
        self.canvas.delete(self.logo_first)
        select_mode_img = Image.open("D:/uzh_project/interface/images/select_mode.png")  # Replace with your image path
        self.select_mode_img = ImageTk.PhotoImage(select_mode_img)
        # Create a canvas item for the image
        self.select_mode_img_canvas = self.canvas.create_image(self.canvas_width/2, self.canvas_height*0.3, image=self.select_mode_img, anchor=tk.CENTER)

        ################# Edit frames mode button #################
        edit_frames_mode_img = Image.open("D:/uzh_project/interface/images/edit_frames.png")  # Replace with your image path
        self.edit_frames_mode_img = ImageTk.PhotoImage(edit_frames_mode_img)
        # Create a canvas item for the image
        self.edit_frames_img_canvas = self.canvas.create_image(self.canvas_width*0.35, self.canvas_height*0.45, image=self.edit_frames_mode_img, anchor=tk.CENTER)
        # Bind the click event to the canvas item
        self.canvas.tag_bind(self.edit_frames_img_canvas, '<Button-1>', self.edit_frame)
        self.canvas.tag_bind(self.edit_frames_img_canvas,"<Enter>", self.on_enter)
        self.canvas.tag_bind(self.edit_frames_img_canvas,"<Leave>", self.on_leave)

        ################## Edit frames mode image #################
        edit_frames_mode_description_img = Image.open("D:/uzh_project/interface/images/edit_frames_description.png")  # Replace with your image path
        self.edit_frames_mode_description_img = ImageTk.PhotoImage(edit_frames_mode_description_img)
        # Create a canvas item for the image
        self.edit_frames_description_img_canvas = self.canvas.create_image(self.canvas_width*0.35, self.canvas_height*0.555, image=self.edit_frames_mode_description_img, anchor=tk.CENTER)

        ################# Edit segments mode button #################
        edit_segments_mode_img = Image.open("D:/uzh_project/interface/images/edit_segments.png")  # Replace with your image path
        self.edit_segments_mode_img = ImageTk.PhotoImage(edit_segments_mode_img)
        # Create a canvas item for the image
        self.edit_segments_img_canvas = self.canvas.create_image(self.canvas_width*0.65, self.canvas_height*0.45, image=self.edit_segments_mode_img, anchor=tk.CENTER)
        # Bind the click event to the canvas item
        self.canvas.tag_bind(self.edit_segments_img_canvas, '<Button-1>', self.edit_segments)
        self.canvas.tag_bind(self.edit_segments_img_canvas,"<Enter>", self.on_enter)
        self.canvas.tag_bind(self.edit_segments_img_canvas,"<Leave>", self.on_leave)

        ################## Edit frames mode image #################
        edit_segments_mode_description_img = Image.open("D:/uzh_project/interface/images/edit_segments_description.png")  # Replace with your image path
        self.edit_segments_mode_description_img = ImageTk.PhotoImage(edit_segments_mode_description_img)
        # Create a canvas item for the image
        self.edit_segments_description_img_canvas = self.canvas.create_image(self.canvas_width*0.65, self.canvas_height*0.55, image=self.edit_segments_mode_description_img, anchor=tk.CENTER)


    def edit_frame(self, event:None):
        ############ Change Canvas #############
        self.canvas.pack_forget()
        self.segment_mode_canvas.pack_forget()
        self.frame_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.edit_frame_sidebar()
        self.load_frames()
        if self.pose_estimator and not self.pose_mode:
            self.skeleton, self.face_skeleton = mediapipe(self.file_path)

    def edit_segments(self, event:None):
        self.canvas.pack_forget()
        self.frame_mode_canvas.pack_forget()
        self.segment_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.segment_mode_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.edit_segment_sidebar()
        if self.pose_estimator and not self.pose_mode:
            self.skeleton, self.face_skeleton = mediapipe(self.file_path)
        self.setup_layout()
        self.general_mode = 'segment_level'
        
    def edit_frame_sidebar(self):

        self.frame_mode_sidebar_buttons = []

        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        load_logo = Image.open("D:/uzh_project/interface/images/sidebar_logo.png")  # Replace with your image path
        self.logo_img = ImageTk.PhotoImage(load_logo)
        self.logo_label = tk.Label(self.sidebar, image=self.logo_img, bg=sidebar_color)
        self.logo_label.pack(side=tk.TOP, pady=10)  # Adjust x and y to position the logo correctly

        button2 = tk.Button(self.sidebar, text="Mode", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.button_mode)
        button2.pack(side=tk.TOP, pady=5)
        button2.bind("<Enter>", lambda event: button2.config(bg=button_activate))
        button2.bind("<Leave>", lambda event: button2.config(bg=sidebar_color))
        self.frame_mode_sidebar_buttons.append(button2)

        button0 = tk.Button(self.sidebar, text="New", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.frame_edit_new_video)
        button0.pack(side=tk.TOP, pady=5)
        button0.bind("<Enter>", lambda event: button0.config(bg=button_activate))
        button0.bind("<Leave>", lambda event: button0.config(bg=sidebar_color))
        self.frame_mode_sidebar_buttons.append(button0)

        button1 = tk.Button(self.sidebar, text="File", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.button_file)
        button1.pack(side=tk.TOP, pady=5)
        button1.bind("<Enter>", lambda event: button1.config(bg=button_activate))
        button1.bind("<Leave>", lambda event: button1.config(bg=sidebar_color))
        self.frame_mode_sidebar_buttons.append(button1)

        button3 = tk.Button(self.sidebar, text="Add", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.add_keypoint)
        button3.pack(side=tk.TOP, pady=5)
        button3.bind("<Enter>", lambda event: button3.config(bg=button_activate))
        button3.bind("<Leave>", lambda event: button3.config(bg=sidebar_color))
        self.frame_mode_sidebar_buttons.append(button3)

        button4 = tk.Button(self.sidebar, text="Help", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.references)
        button4.pack(side=tk.TOP, pady=5)
        button4.bind("<Enter>", lambda event: button4.config(bg=button_activate))
        button4.bind("<Leave>", lambda event: button4.config(bg=sidebar_color))
        self.frame_mode_sidebar_buttons.append(button4)

    def frame_edit_new_video(self):
        file_path = filedialog.askopenfilename()
        self.file_path = file_path
        self.video = cv2.VideoCapture(file_path)
        self.load_frames()
        self.skeleton, self.face_skeleton = mediapipe(self.file_path)

    def references(self):
        options_menu = tk.Menu(self.root, tearoff=0, bg= background_color, fg= button_color, borderwidth=0)
        options_menu.add_command(label="Pose", command=lambda: self.pose_reference())
        options_menu.add_command(label="Right Hand", command=lambda: self.right_hand_reference())
        options_menu.add_command(label="Left Hand", command=lambda: self.left_hand_reference())
        options_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())


    def button_file(self):
        file_menu = tk.Menu(self.root, tearoff=0, bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), activebackground= button_activate, borderwidth= 0)
        file_menu.add_command(label="Save Keypoints", command=lambda: self.save_keypoints())
        file_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def save_keypoints(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy array", "*.npy")])
        
        if save_path:
            skeleton = [self.skeleton, self.face_skeleton]
            np.save(save_path, skeleton)
            print(f"Pose data saved to {save_path}")


    def button_mode(self):
        file_menu = tk.Menu(self.root, tearoff=0, bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), borderwidth=0, activebackground= button_activate)
        file_menu.add_command(label="Edit Segments", command=lambda: self.change_mode_from_frame2segment())
        file_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def change_mode_from_frame2segment(self):

        self.general_mode = 'segment_level'
        self.frame_mode_canvas.pack_forget()
        self.sidebar.pack_forget()
        self.delete_forget_frame_mode_features()
        self.segment_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.segment_mode_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.edit_segment_sidebar()
        self.setup_layout()
        self.pose_estimator = False

    def delete_forget_frame_mode_features(self):

        for sidebar_button in self.frame_mode_sidebar_buttons:
            sidebar_button.pack_forget()

        self.logo_label.pack_forget()

    def load_frames(self):
        self.video_frames = []
        success, frame = self.video.read()
        while True:
            success, frame = self.video.read()
            if not success:
                break
            self.video_frames.append(frame)

        if self.pose_mode:
            self.empty_video_frames = []
            success, frame = self.empty_video.read()
            while True:
                success, frame = self.empty_video.read()
                if not success:
                    break
                self.empty_video_frames.append(frame)
        self.display_frames()
        self.add_navigation_circles()

    def display_frames(self):
        for button in self.frame_buttons:
            button.destroy()
        self.frame_buttons = []

        start = self.current_frame_offset
        end = min(start + self.num_displayed_frames, len(self.video_frames))

        for i in range(start, end):
            frame = self.video_frames[i]
            resized_frame = cv2.resize(frame, (self.frame_width, self.frame_strip_height))
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(img)

            # Create a button for each frame in the strip
            frame_button = tk.Button(self.frame_strip, image=tk_img, command=lambda idx=i: self.select_frame(idx),  highlightcolor='#3c3c3c', highlightbackground='#3c3c3c', highlightthickness=2 , name=str(i))
            frame_button.image = tk_img
            frame_button.grid(row=0, column=i - start)

            frame_button.bind("<Enter>", lambda event, idx = i, btn = frame_button : self.on_enter_frames(event, btn,idx))
            frame_button.bind("<Leave>", self.hide_tooltip)

            self.frame_buttons.append(frame_button)
            x1, y1, x2, y2 = frame_button.bbox('all')
            self.frame_bottom_info[frame_button] = (x1, y1, x2, y2, i)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def add_navigation_circles(self):

        left_image_path = "D:/uzh_project/interface/images/left-navigate.png"  # Replace with your left arrow image path
        right_image_path = "D:/uzh_project/interface/images/right-navigate.png"  # Replace with your right arrow image path

        self.left_image = Image.open(left_image_path)
        self.left_img = ImageTk.PhotoImage(self.left_image)

        self.right_image = Image.open(right_image_path)
        width, height = self.right_image.size
        self.right_img = ImageTk.PhotoImage(self.right_image)

        # Position the left navigation button on the canvas
        self.left_button = self.frame_mode_canvas.create_image(0, self.canvas_height - self.frame_strip_height - height, image=self.left_img, anchor=tk.NW)
        self.frame_mode_canvas.tag_bind(self.left_button, "<Button-1>", self.show_previous_frames)

        # Position the right navigation button on the canvas
        self.right_button = self.frame_mode_canvas.create_image(self.canvas_width - width, self.canvas_height - self.frame_strip_height - height, image=self.right_img, anchor=tk.NW)
        self.frame_mode_canvas.tag_bind(self.right_button, "<Button-1>", self.show_next_frames)

        # Bind events for mouse hover
        self.frame_mode_canvas.tag_bind(self.left_button, "<Enter>", self.on_enter)
        self.frame_mode_canvas.tag_bind(self.left_button, "<Leave>", self.on_leave)
        self.frame_mode_canvas.tag_bind(self.right_button, "<Enter>", self.on_enter)
        self.frame_mode_canvas.tag_bind(self.right_button, "<Leave>", self.on_leave)

    def show_previous_frames(self, event = None):
        if self.current_frame_offset - self.scroll_increment >= 0:
            self.current_frame_offset -= self.scroll_increment
            self.display_frames()

    def show_next_frames(self, event = None):
        if self.current_frame_offset + self.scroll_increment < len(self.video_frames):
            self.current_frame_offset += self.scroll_increment
            self.display_frames()

    def select_frame(self, index):

        # Deselect previous frame (if any)
        if self.selected_frame_index is not None:
            self.frame_buttons[self.selected_frame_index - self.current_frame_offset].config(relief=tk.FLAT)
        # Select the new frame
        self.selected_frame_index = index
        self.frame_buttons[self.selected_frame_index - self.current_frame_offset].config(relief=tk.SUNKEN)
        # Load keypoints for the new frame
        self.display_image()
        self.clear_keypoints()
        self.pose_display_edit()
                
    def pose_display_edit(self):
        x_ , y_ = self.frame_mode_img_start_location
        #self.clear_keypoints()
        skeleton = self.skeleton[self.selected_frame_index]          
        face = self.face_skeleton[self.selected_frame_index]
        self.draw_lines()
        for i in range(48):

            color = (245, 47, 86)
            color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            x = skeleton[0,i]
            y = skeleton[1,i]

            if x and y <= 1:
                x_px, y_px = _normalized_to_pixel_coordinates(x, y, self.image_width, self.image_height)
                if x and y != 0:
                    keypoint_id = self.frame_mode_canvas.create_oval(x_px+x_-2, y_px+y_-2, x_px+x_+2, y_px+y_+2, fill=color, outline = color,tags="keypoint",width=0.5)
                    self.keypoints.append(keypoint_id)
                    self.keypoint_positions[keypoint_id] = [self.selected_frame_index,i, x_px, y_px]

        for j in range(11):
            x = face[0,j]
            y = face[1,j]

            if x and y <= 1:
                x_px, y_px = _normalized_to_pixel_coordinates(x, y, self.image_width, self.image_height)
                if x and y != 0:
                    face_keypoint_id = self.frame_mode_canvas.create_oval(x_px+x_-2, y_px+y_-2, x_px+x_+2, y_px+y_+2, fill='red', outline = color,tags="keypoint",width=0.5)
                    self.face_keypoints.append(face_keypoint_id)

    def clear_keypoints(self):
        for item in self.keypoints:
            self.frame_mode_canvas.delete(item)
        self.keypoints = []
        self.keypoint_positions = {}

    def draw_lines(self):
        skeleton = self.skeleton[self.selected_frame_index]
        face = self.face_skeleton[self.selected_frame_index]
        x_ , y_ = self.frame_mode_img_start_location
        
        for counter in range(len(connections)):
            
            connection = connections[counter]
            start_idx = connection[0]
            end_idx = connection[1]
            x_start = skeleton[0, start_idx]
            y_start = skeleton[1, start_idx]
            color = lines_color [counter]
            color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            x_end = skeleton[0, end_idx]
            y_end = skeleton[1, end_idx]

            if x_start<1 and x_end<1 and y_end<1 and y_start < 1.0:

                x_start, y_start = _normalized_to_pixel_coordinates(x_start , y_start , self.image_width , self.image_height)
                x_end, y_end = _normalized_to_pixel_coordinates(x_end, y_end, self.image_width, self.image_height)
                if (x_end != 0 and y_end != 0) and (x_start != 0 and y_start != 0):
                    line = self.frame_mode_canvas.create_line(x_start + x_, y_start + y_, x_end + x_, y_end + y_, fill=color,width=3)
                    self.lines.append(line)
                    counter += 1

        for connection in face_connections:

            start_idx = connection[0]
            end_idx = connection[1]

            x_start = face[0, start_idx]
            y_start = face[1, start_idx]
            x_end = face[0, end_idx]
            y_end = face[1, end_idx]

            if x_start<1 and x_end<1 and y_end<1 and y_start < 1.0:
                x_start, y_start = _normalized_to_pixel_coordinates(x_start , y_start , self.image_width, self.image_height)
                x_end, y_end = _normalized_to_pixel_coordinates(x_end, y_end, self.image_width, self.image_height)
                line = self.frame_mode_canvas.create_line(x_start + x_, y_start + y_, x_end + x_, y_end + y_, fill="green",width=3)
                self.face_lines.append(line)

    def update_frame_mode_canvas(self, event):
        pass

    def on_keypoint_press(self, event):
        # Check if the user clicked on a keypoint within the capture radius
        if self.mode == 'replacing keypoint':

            for keypoint in self.keypoints:
                x0, y0, x1, y1 = self.coords(keypoint)
                keypoint_center_x = (x0 + x1) / 2
                keypoint_center_y = (y0 + y1) / 2
                distance = math.sqrt((event.x - keypoint_center_x) ** 2 + (event.y - keypoint_center_y) ** 2)
                if distance <= self.capture_radius:

                    self.drag_data["item"] = keypoint
                    self.drag_data["frame_idx"] = self.selected_frame_index
                    self.drag_data["keypoint_idx"] = (self.keypoint_positions[keypoint])[1]
                    self.drag_data["x"] = event.x
                    self.drag_data["y"] = event.y            

                    break
        elif self.mode == 'adding keypoint':
                self.add_new_keypoint(event)

    def on_mouse_move(self, event):
        for keypoint in self.keypoints:
            x0, y0, x1, y1 = self.coords(keypoint)
            keypoint_center_x = (x0 + x1) / 2
            keypoint_center_y = (y0 + y1) / 2
            distance = math.sqrt((event.x - keypoint_center_x) ** 2 + (event.y - keypoint_center_y) ** 2)
            if distance <= self.capture_radius:
                title = titles[str((self.keypoint_positions[keypoint])[1])]
                if not self.tooltip:
                    self.show_tooltip(event, (self.keypoint_positions[keypoint])[1], title, mode = 'keypoint')
                else:
                    self.tooltip.geometry(f"+{event.x + self.root.winfo_rootx()}+{event.y + self.root.winfo_rooty() - 20}")
                    #self.tooltip_label.config(text=f"Index: {(self.keypoint_positions[keypoint])[1]}, {title}")
                return


        self.hide_tooltip()
        
    def show_tooltip(self, event, index, title,mode):

        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        #self.tooltip.geometry(f"+{event.x + self.root.winfo_rootx()}+{event.y + self.root.winfo_rooty() - 20}")
        if mode == 'keypoint':
            label = tk.Label(self.tooltip, text=f"Index: {index}, {title}", fg= button_color, background=sidebar_color, relief='solid', borderwidth=0)
        else:
            #self.tooltip.geometry(f"+{(index+1)*160+100}+{1150}")
            #self.tooltip.geometry(f"+{0}+{0}")
            label = tk.Label(self.tooltip, text=f"Frame {index}", fg= button_color,background=sidebar_color, relief='solid', borderwidth=0)
            self.tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            self.tooltip.deiconify()
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def on_keypoint_motion(self, event):
        # Move the keypoint if it's being dragged

        if self.drag_data["item"]:
            
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            self.move(self.drag_data["item"], dx, dy)
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            keypoint_idx = self.drag_data["keypoint_idx"]
            x_, y_ = self.frame_mode_img_start_location
            event_x_, event_y_ = event.x - x_ , event.y - y_
            normal_x,normal_y = _pixel_coordinates_to_normalized(event_x_, event_y_, self.image_width, self.image_height)
            (self.skeleton[self.selected_frame_index])[0,keypoint_idx],(self.skeleton[self.selected_frame_index])[1,keypoint_idx] = normal_x, normal_y
            self.clear_lines()
            self.clear_keypoints()
            self.pose_display_edit()

    def clear_lines(self):
        for line in self.lines:
            self.frame_mode_canvas.delete(line)
        self.lines = []

    def on_keypoint_release(self, event):

        if self.drag_data["item"]:
            
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            keypoint_idx = self.drag_data["keypoint_idx"]
            x_, y_ = self.frame_mode_img_start_location
            event_x_, event_y_ = event.x - x_ , event.y - y_
            normal_x,normal_y = _pixel_coordinates_to_normalized(event_x_, event_y_, self.image_width, self.image_height)
            (self.skeleton[self.selected_frame_index])[0,keypoint_idx],(self.skeleton[self.selected_frame_index])[1,keypoint_idx] = normal_x, normal_y

            # Reset the drag data
            self.drag_data["item"] = None
            self.drag_data["x"] = 0
            self.drag_data["y"] = 0
            self.drag_data['keypoint_idx'] = None
            self.drag_data['frame_idx'] = None


    def coords(self, item):
        return self.frame_mode_canvas.coords(item)
    
    def move(self, item, dx, dy):
        self.frame_mode_canvas.move(item, dx, dy)


    def display_image(self):
        if self.selected_frame_index is not None:
            if self.pose_mode:
                frame = self.empty_video_frames[self.selected_frame_index]
            else:
                frame = self.video_frames[self.selected_frame_index]

            height, width, channel = frame.shape
            self.image_width, self.image_height = width, height
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Convert to ImageTk format and display on the canvas
            tk_img = ImageTk.PhotoImage(img)
            if (self.canvas_height-height)/2 + height +self.frame_strip_height < self.canvas_height:
                self.frame_mode_canvas.create_image((self.canvas_width-width)/2, (self.canvas_height-height)/2 , anchor=tk.NW, image=tk_img)
                self.frame_mode_img_start_location = ((self.canvas_width-width)/2 , (self.canvas_height-height)/2)
            else:
                if self.canvas_height - height - self.frame_strip_height > 0:
                    offset = self.canvas_height - height - self.frame_strip_height
                    self.frame_mode_canvas.create_image((self.canvas_width-width)/2, offset//2 , anchor=tk.NW, image=tk_img)
                    self.frame_mode_img_start_location = ((self.canvas_width-width)/2 , offset//2)
                else:
                    new_height = self.canvas_height-self.frame_strip_height
                    new_width = int(width * new_height / height)
                    resized_frame = cv2.resize(frame, (self.image_width, self.canvas_height-self.frame_strip_height))
                    self.frame_mode_img_start_location = ((self.canvas_width-width)/2 , offset//2)
            self.frame_mode_canvas.image = tk_img
            self.add_navigation_circles()
        

##################  ADD KEYPOINTS ##########################
    def add_keypoint(self):

        options_menu = tk.Menu(self.root, tearoff=0, bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), borderwidth= 0, activebackground= button_activate)
        options_menu.add_command(label="Pose", command=lambda: self.add_pose())
        options_menu.add_command(label="Right Hand", command=lambda: self.add_rh_pose())
        options_menu.add_command(label="Left Hand", command=lambda: self.add_lh_pose())
        options_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def add_pose(self):
        self.adding_pose_flag = True
        self.mode = 'adding keypoint'

    def add_rh_pose(self):
        self.adding_rh_flag = True
        self.mode = 'adding keypoint'

    def add_lh_pose(self):
        self.adding_lh_flag = True
        self.mode = 'adding keypoint'

    def add_new_keypoint(self, event):
        
        x, y = event.x, event.y
        color = '#{:02x}{:02x}{:02x}'.format(255, 0, 0)
        new_keypoint = self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, tags="guideline")
        self.keypoints.append(new_keypoint)
        self.getting_index(new_keypoint=new_keypoint,x= x, y= y)

    def getting_index(self, new_keypoint, x, y):
        # Open input window for keypoint index
        self.escape_new_keypoint = False
        self.input_window = tk.Toplevel(self.root)
        self.input_window.geometry(f"+{x + self.root.winfo_x()}+{y + self.root.winfo_y()}")
        self.input_window.title("Input Keypoint Index")

        ttk.Label(self.input_window, text="Enter Keypoint Index:").pack(pady=5)
        self.keypoint_index_entry = ttk.Entry(self.input_window)
        self.keypoint_index_entry.pack(pady=5)
        self.keypoint_index_entry.focus_set()
        self.input_window.bind('<Escape>', self.handle_escape,new_keypoint)
        if not self.escape_new_keypoint:
        #self.keypoint_index_entry.bind('<Return>', self.process_keypoint_index(new_keypoint,x=x,y=y))  # Bind <Return> key to the entry widget
            ttk.Button(self.input_window, text="OK", command=lambda: self.process_keypoint_index(new_keypoint,x=x,y=y)).pack(pady=5)
            self.keypoint_index_entry.bind('<Return>', lambda event: self.process_keypoint_index(new_keypoint,x=x,y=y))  # Bind <Return> key to the entry widget


    def handle_escape(self,new_keypoint):
        # Define the actions to perform on Escape key press
        print("Escape key pressed. Closing the input window.")
        self.escape_new_keypoint = True
        self.input_window.destroy()
        #self.keypoints.remove(new_keypoint)
        self.canvas.delete(new_keypoint)

    def process_keypoint_index(self, new_keypoint,x,y):
        # Process the keypoint index input
        try:
            index = int(self.keypoint_index_entry.get().strip())
            if (self.skeleton[self.selected_frame_index])[0,index] ==0 and (self.skeleton[self.selected_frame_index])[0,index] ==0:
                flag = True
            else:
                flag = self.confirm_overwrite()
                if flag:
                    pass
                else:
                    self.input_window.destroy()
                    self.getting_index(new_keypoint, x, y)
                    
            if flag:
                if self.adding_pose_flag:
                    if index in range(0,6):
                        normal_x,normal_y = _pixel_coordinates_to_normalized(x, y, self.image_width, self.image_height)
                        (self.skeleton[self.selected_frame_index])[0,index],(self.skeleton[self.selected_frame_index])[1,index] = normal_x, normal_y
                        self.input_window.destroy()
                    else:
                        messagebox.showinfo("Input Correct Index","Pose Keypoint Index Should be in the 0-7")
                        raise ValueError("Invalid index")

                elif self.adding_rh_flag:
                    if index in range(6,27):
                        normal_x,normal_y = _pixel_coordinates_to_normalized(x, y, self.image_width, self.image_height)
                        (self.skeleton[self.selected_frame_index])[0,index],(self.skeleton[self.selected_frame_index])[1,index] = normal_x, normal_y
                        self.input_window.destroy()
                    else:
                        messagebox.showinfo("Input Correct Index","Right Hand Keypoint Index Should be in the 8-28")
                        raise ValueError("Invalid index")
                elif self.adding_lh_flag:
                    if index in range(27,48):
                        normal_x,normal_y = _pixel_coordinates_to_normalized(x, y, self.image_width, self.image_height)
                        (self.skeleton[self.selected_frame_index])[0,index],(self.skeleton[self.selected_frame_index])[1,index] = normal_x, normal_y
                        self.input_window.destroy()
                    else:
                        messagebox.showinfo("Input Correct Index","Right Hand Keypoint Index Should be in the 29-49 \n Please try again.")
                        raise ValueError("Invalid index")

        except ValueError as e:
            #messagebox.showerror("Invalid input", f"{e}. Please try again.")
            self.input_window.destroy()
            self.getting_index(new_keypoint, x, y)  # Re-open the input window for another attempt
        
        self.clear_lines()
        self.clear_keypoints()
        self.pose_display_edit()
    
    def confirm_overwrite(self):
        return messagebox.askyesno("Overwrite Existing Data", "This keypoint already has data. Do you want to overwrite it?")
    
    def cancel_action(self,event):
        # Cancel the current action
        if self.input_window:
            self.input_window.destroy()

        self.mode = 'replacing keypoint'
        # Additional actions to cancel other operations can be added here
        print("Action cancelled")

################## POSE REFERENCE ##########################
    def pose_reference(self):
        self.pose_visible = not self.pose_visible

        if self.pose_visible:

            pose_reference_img_address = "D:/uzh_project/interface/images/pose_reference.png"  # Replace with your right arrow image path
            self.pose_reference_image = Image.open(pose_reference_img_address)
            self.pose_reference_img = ImageTk.PhotoImage(self.pose_reference_image)
            self.pose_reference_canvas = self.frame_mode_canvas.create_image(0, 0, image=self.pose_reference_img, anchor=tk.NW)
            
        else:
            
            self.ref_clean()
    
    def ref_clean(self):

        self.frame_mode_canvas.delete(self.pose_reference_canvas)

################# RIGHT HAND REFERENCE ##################
    def right_hand_reference(self):
        # Implement right hand reference functionality
        self.right_hand_ref_visible = not self.right_hand_ref_visible
        if self.right_hand_ref_visible:

            right_hand_reference_img_address = "D:/uzh_project/interface/images/right_hand_reference.png"  # Replace with your right arrow image path
            self.right_hand_reference_image = Image.open(right_hand_reference_img_address)
            width, height = self.right_hand_reference_image.size
            self.right_hand_reference_img = ImageTk.PhotoImage(self.right_hand_reference_image)
            self.right_hand_reference_canvas = self.frame_mode_canvas.create_image(self.canvas_width - width*2, 0, image=self.right_hand_reference_img, anchor=tk.NW)
            
        else:
            
            self.ref_right_hand_clean()

    def ref_right_hand_clean(self):
        self.frame_mode_canvas.delete(self.right_hand_reference_canvas)

################# LEFT HAND REFERENCE ####################
    def left_hand_reference(self):
        # Implement left hand reference functionality
        self.left_hand_ref_visible = not self.left_hand_ref_visible
        if self.left_hand_ref_visible:

            left_hand_reference_img_address = "D:/uzh_project/interface/images/left_hand_reference.png"  
            self.left_hand_reference_image = Image.open(left_hand_reference_img_address)
            width, height = self.left_hand_reference_image.size
            self.left_hand_reference_img = ImageTk.PhotoImage(self.left_hand_reference_image)
            self.left_hand_reference_canvas = self.frame_mode_canvas.create_image(self.canvas_width - width, 0, image=self.left_hand_reference_img, anchor=tk.NW)
            
        else:
            
            self.ref_left_hand_clean()

    def ref_left_hand_clean(self):
        self.frame_mode_canvas.delete(self.left_hand_reference_canvas)

    ################### SEGMENT MODE #####################

    def edit_segment_sidebar(self):

        self.segment_mode_sidebar_buttons = []
        self.segment_mode_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.segment_mode_sidebar.pack_propagate(False)

        load_logo = Image.open("D:/uzh_project/interface/images/sidebar_logo.png")  # Replace with your image path
        self.logo_img = ImageTk.PhotoImage(load_logo)
        self.logo_label = tk.Label(self.segment_mode_sidebar, image=self.logo_img, bg=sidebar_color)
        self.logo_label.pack(side=tk.TOP, pady=10)  # Adjust x and y to position the logo correctly


        segment_mode_button2 = tk.Button(self.segment_mode_sidebar, text="Mode", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.button_mode2)
        segment_mode_button2.pack(side=tk.TOP, pady=5)
        segment_mode_button2.bind("<Enter>", lambda event: segment_mode_button2.config(bg=button_activate))
        segment_mode_button2.bind("<Leave>", lambda event: segment_mode_button2.config(bg=sidebar_color))
        self.segment_mode_sidebar_buttons.append(segment_mode_button2)

        segment_mode_button0 = tk.Button(self.segment_mode_sidebar, text="New", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.segment_edit_new_video)
        segment_mode_button0.pack(side=tk.TOP, pady=5)
        segment_mode_button0.bind("<Enter>", lambda event: segment_mode_button0.config(bg=button_activate))
        segment_mode_button0.bind("<Leave>", lambda event: segment_mode_button0.config(bg=sidebar_color))
        self.segment_mode_sidebar_buttons.append(segment_mode_button0)

        segment_mode_button1 = tk.Button(self.segment_mode_sidebar, text="save", width=8, height=1, borderwidth=0, relief='flat', font=('Calibri Light', 12), bg=sidebar_color, fg=button_color, activebackground=button_activate, activeforeground=button_color, command=self.save_video)
        segment_mode_button1.pack(side=tk.TOP, pady=5)
        segment_mode_button1.bind("<Enter>", lambda event: segment_mode_button1.config(bg=button_activate))
        segment_mode_button1.bind("<Leave>", lambda event: segment_mode_button1.config(bg=sidebar_color))
        self.segment_mode_sidebar_buttons.append(segment_mode_button1)

    def button_mode2(self):
        file_menu = tk.Menu(self.root, tearoff=0, bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), activebackground= button_activate, relief=tk.FLAT, borderwidth=0)
        file_menu.add_command(label="Edit Frames", command=lambda: self.change_mode_from_segment2frame())
        file_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def change_mode_from_segment2frame(self):

        self.general_mode = 'frame_level'
        self.segment_mode_canvas.pack_forget()
        self.segment_mode_sidebar.pack_forget()
        self.delete_forget_segment_mode_features()
        self.frame_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.edit_frame_sidebar()
        if self.pose_mode:
            empty_inserted_pose_path = 'empty_video.mp4'
            self.create_empty_video(frame_number = len(self.skeleton), des_path= empty_inserted_pose_path,width= self.video_width, height= self.video_height, fps= self.fps)
            self.empty_video_frames = []
            success, frame = self.empty_video.read()
            while True:
                success, frame = self.empty_video.read()
                if not success:
                    break
                self.empty_video_frames.append(frame)
        #self.load_frames()
        self.display_frames()
        self.add_navigation_circles()
        if self.pose_estimator and not self.pose_mode:
            self.skeleton, self.face_skeleton = mediapipe(self.file_path)


    def segment_edit_new_video(self):
        file_path = filedialog.askopenfilename()

        if not self.pose_mode:
            self.file_path = file_path
            self.video = cv2.VideoCapture(file_path)

        else:
            
            self.pose_file_path = file_path
            saving_path = self.config['pose_video_path']
            empty_video_path = 'empty_video.mp4'
            if file_path:

                self.skeleton, self.face_skeleton = np.load(self.pose_file_path,allow_pickle=True)
                self.pose_to_video(self.skeleton, self.face_skeleton,saving_path,flag= False)
                self.create_empty_video(frame_number = len(self.skeleton), des_path= empty_video_path,width= 800, height= 600, fps= 25)

    
    def save_video(self):
        if not self.pose_mode:

            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("*.mp4")])
        
            if save_path:
                video = self.video
                np.save(save_path, video)
                print(f"Video saved to {save_path}")

        else:
            save_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy array", "*.npy")])
            
            if save_path:
                skeleton = [self.skeleton, self.face_skeleton]
                np.save(save_path, skeleton)
                print(f"Pose data saved to {save_path}")


    def setup_layout(self):
        
        self.segment_mode_canvas.bind("<Button-1>", self.toggle_play_pause)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale", background=background_color, troughcolor="#bfbfbf", sliderlength=20)

        self.slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video, style="Horizontal.TScale")

        self.frame_label = tk.Label(self.root, text="Frame: 0", font=("Calibri Light", 12), background=background_color, fg= button_color)

        self.time_label = tk.Label(self.root, text="Time: 0.00", font=("Calibri Light", 12),background=background_color, fg= button_color)

        if not self.video.isOpened():
            print("The video file could not be opened or is empty.")
            if self.file_path:
                self.video = cv2.VideoCapture(self.file_path)

        if len(self.video_frames) == 0:
            while True:
                success, frame = self.video.read()
                if not success:
                    break
                self.video_frames.append(frame)

        else:
            print("The video file opened successfully.")
        self.fig_height_inch = 1.2
        # Desired DPI (dots per inch)
        self.dpi = 100
        self.load_button_images()
        canvas_height = self.video_height + self.segment_mode_button_width+ self.fig_height_inch * self.dpi + 20 + 20 + 20 + 20
        self.segment_mode_canvas.config(width=self.canvas_width, height = canvas_height)# Place the time frame bar
        self.slider.place(x = self.canvas_width * 0.545, y = 2 * 20 + self.video_height, anchor=tk.CENTER, width=800)
        # Place the frame label
        self.frame_label.place(x=0.58 * self.canvas_width, y = 3 * 20 + self.video_height, anchor=tk.CENTER)
        # Place the time label
        self.time_label.place(x=0.47 * self.canvas_width, y = 3 * 20 + self.video_height, anchor=tk.CENTER)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.slider.config(to=self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.display_first_frame()

        self.create_buttons()
        self.movement_graph(self.video.get(cv2.CAP_PROP_FPS))

    def load_button_images(self):
        self.button_images = []
        button_files = ["mark.png", "insert_video.png", "recording.png", "insert_pose.png"]  # Replace with your image paths
        for file in button_files:
            image = Image.open(os.path.join(self.config['images_root'],file))
            photo = ImageTk.PhotoImage(image)
            self.button_images.append(photo)
        self.segment_mode_button_width, self.segment_mode_button_height = image.size
    def create_buttons(self):
        button_commands = [self.mark, self.insert_uploaded_video, self.toggle_recording, self.insert_pose]  # Define your button commands
        for i, image in enumerate(self.button_images):
            button = tk.Button(self.root, image=image, command=button_commands[i], borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
            button.place(x=int(self.canvas_width*(0.07 * (i + 1)+0.34)), y = self.video_height + 4*20 + self.fig_height_inch * self.dpi, anchor=tk.CENTER)  # Adjust positioning as needed
            self.segment_mode_buttons.append(button)

    def moving_average(self,data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def movement_graph(self, fps):
    
        right_hand_speeds, right_hand_accelerations, left_hand_speeds, left_hand_accelerations = movement(self.skeleton, fps)
        # Apply smoothing to the speed data
        window_size = 5  # Adjust this window size for more or less smoothing
        smooth_right_hand_speeds = self.moving_average(right_hand_speeds, window_size)
        smooth_left_hand_speeds = self.moving_average(left_hand_speeds, window_size)
        # Time or frame indices (assuming constant time intervals)
        time_indices = list(range(len(smooth_right_hand_speeds)))
        # Desired DPI (dots per inch)
        self.dpi = 100
        # Calculate the figure width in inches to match the video width
        self.fig_width_inch = self.video_width*1.3 / self.dpi
        self.fig_height_inch = 1.2
        fig, ax = plt.subplots(figsize=(self.fig_width_inch, self.fig_height_inch), dpi=self.dpi)

        # Set the background colors
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor('gray')

        # Plot all lines on the same graph
        ax.plot(time_indices, smooth_right_hand_speeds, label='Right Hand Speed', color='b', linestyle='-',linewidth='0.5')
        ax.plot(time_indices, smooth_left_hand_speeds, label='Left Hand Speed', color='r', linestyle='-',linewidth='0.5')
        #ax.plot(time_indices, right_hand_accelerations, label='Right Hand Acceleration', color='g', marker='s')
        #ax.plot(time_indices, left_hand_accelerations, label='Left Hand Acceleration', color='m', marker='^')

        # ax.set_title('Hand Speed Over Time')
        # ax.set_xlabel('Frame')
        # ax.set_ylabel('Value')
        ax.set_xlim(0, len(time_indices) - 1)  # Ensure the x-axis starts at 0 and ends at the last index
        #ax.grid(True, which='both', axis='both', color='white', linestyle='--', linewidth=0.5, alpha=0.7)
        # Customize major and minor grid lines
        #ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        #ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        # ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_xlabel('Frame')
        #ax.set_ylabel('Value')
        #ax.tick_params(axis='x', labelsize='small', color = 'gray')  # You can change 'small' to 'x-small', 'medium', or a numeric value

        plt.tight_layout()
        #ax.legend(fontsize='small', labelcolor='gray')
        self.speed_graph = FigureCanvasTkAgg(fig, master=self.segment_mode_canvas)
        self.speed_graph.draw()

        graph_widget = self.speed_graph.get_tk_widget()

        # Set the location and size of the canvas widget on the tk canvas
        #(self.canvas_width-self.video_width)/2, (self.canvas_height-self.video_height)- self.canvas_height*0.15-20
        x = (self.canvas_width-self.video_width)/2
        y = self.video_height + 20 + 20 + 20 
        width = self.video_width
        height = self.fig_height_inch * self.dpi
        graph_widget.place(x=x, y=y, width=width, height=height)

        # Bind the resize event to update the canvas
        self.root.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        # Update the size of the figure canvas based on the window size
        self.speed_graph.get_tk_widget().config(width=event.width, height=event.height)


    def delete_forget_segment_mode_features(self):
        self.frame_label.destroy()
        self.time_label.destroy()
        self.slider.destroy()
        self.logo_label.pack_forget()
        for button in self.segment_mode_buttons:
            button.destroy()
        for sidebar_button in self.segment_mode_sidebar_buttons:
            sidebar_button.pack_forget()

    def mark(self):
        # Get the current frame and time
        frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_number / self.video.get(cv2.CAP_PROP_FPS)
        
        # Store the marked position
        self.marked_positions.append((frame_number, current_time))
        c = len(self.marked_positions)

        # Choose a color from the list based on the mark index
        color_index = (c - 1) % len(mark_colors)  # Ensure it wraps around if there are more than 20 marks
        mark_color = mark_colors[color_index]
        
        # Display the time and frame on the canvas
        button_text = f"Mark{c}: Frame: {frame_number}, Time: {current_time:.2f}"
        button = tk.Button(
            #self.root,
            self.segment_mode_canvas,
            text=button_text, 
            bg=sidebar_color, 
            fg=mark_color, 
            activebackground=button_activate, 
            font=("Helvetica", 9),
            relief=tk.FLAT,
            borderwidth=0,
            width=25,  # Set the width of the button
            height=1   # Set the height of the button
        )
        
        # Assign the command after button initialization
        button.config(command=lambda btn=button, idx=c - 1: self.select_mark(idx, btn))
        button.bind("<Button-3>", lambda event, idx=c-1: self.show_context_menu(event, idx))
        button.initial_color = mark_color
        
        button.place(x=50, y=350 + 25 * c, anchor=tk.NW)
        # Store the button reference
        self.mark_buttons.append(button)
        # Draw a red line on the slider
        self.add_line(frame_number, mark_color)
    
    def add_line(self, frame_number, color):
        slider_length = self.slider.winfo_width()
        slider_position = int((frame_number / self.video.get(cv2.CAP_PROP_FRAME_COUNT)) * slider_length)
        start_y = self.video_height + 20
        end_y = self.video_height + 20 + 20 
        line = self.segment_mode_canvas.create_line(
            slider_position + (self.canvas_width - slider_length) /2, start_y, slider_position + (self.canvas_width - slider_length) /2, end_y,
            fill=color, width=2
        )
        #line.initial_color = color
        self.mark_lines.append(line)
        self.mark_line_ids[line] = slider_position

    def show_context_menu(self, event, index):
        # Create a context menu
        context_menu = tk.Menu(self.root, tearoff=0,bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), activebackground= button_activate, relief=tk.FLAT, borderwidth=0)
        context_menu.add_command(label="Remove Mark", command=lambda: self.remove_mark(index))
        context_menu.post(event.x_root, event.y_root)  # Display the menu at the pointer location

    def remove_mark(self, index):
        # Remove the mark button
        self.mark_buttons[index].destroy()
        # Remove the line
        self.segment_mode_canvas.delete(self.mark_lines[index])
        # Remove the corresponding entries
        del self.mark_buttons[index]
        del self.marked_positions[index]
        del self.mark_lines[index]
        # Check and remove from selected_marks if present
        if index in self.selected_marks:
            self.selected_marks.remove(index)

        # Update positions of subsequent marks and lines if needed
        self.update_mark_positions()


    def update_mark_positions(self):
        # Re-position the remaining buttons
        for i, button in enumerate(self.mark_buttons):
            button.place(x=50, y=350 + 25 * (i + 1), anchor=tk.NW)
            # Update the button command to reflect new index
            button.config(command=lambda btn=button, idx=i: self.select_mark(idx, btn))
            button.bind("<Button-3>", lambda event, idx=i: self.show_context_menu(event, idx))

        # Update the text of the buttons to reflect new indices
        for i, (frame_number, current_time) in enumerate(self.marked_positions):
            self.mark_buttons[i].config(text=f"Mark{i+1}: Frame: {frame_number}, Time: {current_time:.2f}")

    def update_marks_after_insertion(self, old_frame_count, new_frame_count):
        """Updates marks and lines after a video segment is inserted."""
        del_list = []
        frame_difference = new_frame_count - old_frame_count
        start_index, end_index = self.selected_marks

        # Retrieve the frame numbers for the selected marks
        start_frame = self.marked_positions[start_index][0]
        end_frame = self.marked_positions[end_index][0]
        transition_frame = 10

        # Update marked positions
        counter = 0
        for i, (frame_number, _) in enumerate(self.marked_positions):

            if frame_number <= start_frame or frame_number >= end_frame:
                if frame_number <= start_frame:
                    new_frame_number = frame_number
            
                elif frame_number >= end_frame:
                    if self.pose_mode:
                        new_frame_number = frame_number + frame_difference + transition_frame * 2
                    else:
                        new_frame_number = frame_number + frame_difference
                #new_time = new_frame_number / self.video.get(cv2.CAP_PROP_FPS)
                new_time = new_frame_number / self.fps
                self.marked_positions[i] = (new_frame_number, new_time)
                # Update the button text
                self.mark_buttons[i].config(
                    text=f"Mark{counter + 1}: Frame: {new_frame_number}, Time: {new_time:.2f}",
                    bg=sidebar_color
                )

                color = self.mark_buttons[i].initial_color
                # Update the red line position
                self.update_line_position(i, new_frame_number,new_frame_count,color)
                counter += 1

            else:
                self.mark_buttons[i].destroy()
                self.segment_mode_canvas.delete(self.mark_lines[i])
                del_list.append(i)

        del_list.sort(reverse=True)
        for idx in del_list:
            del self.marked_positions[idx]
            del self.mark_buttons[idx]
            del self.mark_lines[idx]

        # Adjust the position of remaining buttons
        for i, button in enumerate(self.mark_buttons):
            button.place(x=50, y=350 + 25 * (i + 1), anchor=tk.NW)

        self.selected_marks = []

    def update_line_position(self, index, frame_number,new_frame_count,color):
        # color_index = (index) % len(mark_colors)  # Ensure it wraps around if there are more than 20 marks
        # line_color = mark_colors[color_index]
        #slider_length = self.slider.winfo_width()
        slider_length = self.video_width
        #print('slider_length:{}'.format(slider_length))
        #slider_position = int((frame_number / self.video.get(cv2.CAP_PROP_FRAME_COUNT)) * slider_length)
        slider_position = int((frame_number / new_frame_count) * slider_length)
        #print('frame_number:{}, slider_length:{},slider_position'.format(frame_number, slider_length,slider_position))
        # Delete the old line and draw a new one
        start_y = self.video_height + 20
        end_y = self.video_height + 20 + 20
        self.segment_mode_canvas.delete(self.mark_lines[index])
        line = self.segment_mode_canvas.create_line(
            slider_position + (self.canvas_width - slider_length) /2, start_y, slider_position + (self.canvas_width - slider_length) /2, end_y,
            fill=color, width=2
        )
        self.mark_lines[index] = line

    def select_mark(self, index, button):

        # Reset the color of previously selected buttons and lines if more than two are selected
        if len(self.selected_marks) >= 2:
            # Reset the background color of the oldest selected mark
            oldest_index = self.selected_marks.pop(0)
            self.mark_buttons[oldest_index].config(bg=sidebar_color)

            color = self.mark_buttons[oldest_index].initial_color

            # Change the oldest line color back to red
            self.segment_mode_canvas.itemconfig(self.mark_lines[oldest_index], fill=color)

        # Add the current index to the selected marks and change its color
        self.selected_marks.append(index)
        self.mark_buttons[index].config(bg=button_activate)

        # Change the color of the selected mark's line to green
        self.segment_mode_canvas.itemconfig(self.mark_lines[index], fill=gold_color)

    def calculate_recording_duration(self):
        if len(self.marked_positions) >= 2:
            start_time = self.marked_positions[-2][1]
            end_time = self.marked_positions[-1][1]
            duration = (end_time - start_time)   # convert milliseconds to seconds
            return duration
        return 0

    def insert_uploaded_video(self):

        self.uploaded_frames = []
        uploaded_video_path = filedialog.askopenfilename()
        self.uploaded_video_path = uploaded_video_path
        self.uploaded_video = cv2.VideoCapture(self.uploaded_video_path)
        while True:
            success, frame = self.uploaded_video.read()
            if not success:
                break
            self.uploaded_frames.append(frame)

        start_farme, end_frame = self.get_marked_frame_range()
        self.insert_video_segment(start_farme, end_frame, 'uploaded')

    def insert_pose(self):

        self.uploaded_frames = []
        inserted_pose_path = filedialog.askopenfilename()
        self.inserted_pose_path = inserted_pose_path
        saving_path = 'inserted_video.mp4'
        
        if inserted_pose_path:
            #self.skeleton = np.load(self.pose_file_path,allow_pickle=True).item()
            self.inserted_skeleton, self.inserted_face_skeleton = np.load(self.inserted_pose_path,allow_pickle=True)
            self.pose_to_video(self.inserted_skeleton, self.inserted_face_skeleton,saving_path,flag=True)
            while True:
                success, frame = self.uploaded_video.read()
                if not success:
                    break
                self.uploaded_frames.append(frame)
            start_farme, end_frame = self.get_marked_frame_range()
            best_start_frame, best_end_frame = self.find_best_start_end_frame(self.skeleton, start_farme, end_frame)
            best_start_frame_, best_end_frame_ = self.find_best_start_end_frame(self.inserted_skeleton, 0, len(self.inserted_skeleton) - 1)
            print('st1: {}, st2: {}, ef1: {}, ef2: {}'.format(best_start_frame,best_start_frame_,best_end_frame,best_end_frame_))
            self.concatenate_poses(best_start_frame, best_end_frame, best_start_frame_, best_end_frame_, transition_frames=10)
            #self.insert_video_segment(best_start_frame, best_end_frame, 'uploaded')


    def concatenate_poses(self,start_farme, end_frame, start_frame_, end_frame_,transition_frames=10):
        
        skeleton = {}
        face_skeleton = {}
        counter = 0
        counter_ = 0
        last_frame_pose1 = self.skeleton[start_farme]
        first_frame_pose2 = self.inserted_skeleton[start_frame_]
        last_frame_pose1_ = self.inserted_skeleton[end_frame_]
        first_frame_pose2_ = self.skeleton[end_frame]
        transition = {}
        for i in range(transition_frames):
            alpha = (i + 1) / (transition_frames + 1)  # Linear interpolation factor
            transition[i] = (1 - alpha) * last_frame_pose1 + alpha * first_frame_pose2
        transition_ = {}
        for i in range(transition_frames):
            alpha = (i + 1) / (transition_frames + 1)  # Linear interpolation factor
            transition_[i] = (1 - alpha) * last_frame_pose1_ + alpha * first_frame_pose2_

        face_last_frame_pose1 = self.face_skeleton[start_farme]
        face_first_frame_pose2 = self.inserted_face_skeleton[start_frame_]
        face_last_frame_pose1_ = self.inserted_face_skeleton[end_frame_]
        face_first_frame_pose2_ = self.face_skeleton[end_frame]
        
        face_transition = {}
        for i in range(transition_frames):
            alpha = (i + 1) / (transition_frames + 1)  # Linear interpolation factor
            face_transition[i] = (1 - alpha) * face_last_frame_pose1 + alpha * face_first_frame_pose2
        face_transition_ = {}
        for i in range(transition_frames):
            alpha = (i + 1) / (transition_frames + 1)  # Linear interpolation factor
            face_transition_[i] = (1 - alpha) * face_last_frame_pose1_ + alpha * face_first_frame_pose2_

        new_video_frame_num = len(self.skeleton) + len(self.inserted_skeleton) - (end_frame - start_farme) + 2 * transition_frames

        for idx in range(new_video_frame_num):


            if idx < start_farme:
                skeleton[idx] = self.skeleton[idx]
                face_skeleton[idx] = self.face_skeleton[idx]


            elif idx >= start_farme and idx < (start_farme + transition_frames):
                skeleton[idx] = transition[idx-start_farme]
                face_skeleton[idx] = face_transition[idx-start_farme]


            elif idx >= (start_farme + transition_frames) and idx < (start_farme + len(self.inserted_skeleton)+transition_frames):

                skeleton[idx] = self.inserted_skeleton[counter]
                face_skeleton[idx] = self.inserted_face_skeleton[counter]
                counter += 1

            elif idx >= len(self.inserted_skeleton) + start_farme + transition_frames and idx < len(self.inserted_skeleton) +  start_farme + 2*transition_frames:
                skeleton[idx] = transition_[counter_]
                face_skeleton[idx] = face_transition_[counter_]
                counter_ += 1


            elif idx >= len(self.inserted_skeleton) + start_farme + 2 * transition_frames:
                skeleton[idx] = self.skeleton[idx - len(self.inserted_skeleton) + (end_frame - start_farme) - 2*transition_frames]
                face_skeleton[idx] = self.face_skeleton[idx - len(self.inserted_skeleton) + (end_frame - start_farme)- 2*transition_frames]

        old_video_frame_num = len(skeleton)
        self.skeleton  = skeleton
        self.face_skeleton = face_skeleton
        self.file_path = 'new_pose_video.mp4'
        self.pose_to_video(self.skeleton, self.face_skeleton,self.file_path,flag=False)
        self.display_first_frame()
        self.after_replacing_new_video(old_video_frame_num, new_video_frame_num)

    def find_best_start_end_frame(self, pose, start_frame, end_frame):

        """
        Find the best start and end frames for a given pose sequence.
        
        Parameters:
        pose (list): A list of frames, each frame being a numpy array with required dimensions.
        start_frame (int): Initial guess for the starting frame.
        end_frame (int): Initial guess for the ending frame.
        
        Returns:
        tuple: The best start and end frame indices.
        """
        num_frames = len(pose)
        counter = 0
        while True:
            # Check if the incremented index is within bounds
            if start_frame + counter < num_frames:
                if (pose[start_frame + counter][0,8] != 0 and pose[start_frame + counter][0,30] != 0):
                    best_start_frame = start_frame + counter
                    break

            if start_frame - counter >= 0:
                if (pose[start_frame - counter][0,8] != 0 and pose[start_frame - counter][0,30] != 0):
                    best_start_frame = start_frame - counter
                    break

            counter += 1

        counter = 0
        while True:
            if end_frame + counter < num_frames:
                if (pose[end_frame + counter][0,7] != 0 and pose[end_frame + counter][0,30] != 0):
                    best_end_frame = end_frame + counter
                    break

            if end_frame - counter >= 0:
                if (pose[end_frame - counter][0,7] != 0 and pose[end_frame - counter][0,30] != 0):
                    best_end_frame = end_frame - counter
                    break

            counter += 1

        return best_start_frame, best_end_frame

    def toggle_recording(self, event=None):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        self.recording_thread = threading.Thread(target=self.record_video)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        # self.recording_thread.join()
        # self.insert_recorded_video()
        if self.recording_thread is not None:
            self.recording_thread.join()

    def record_video(self):

        cap = cv2.VideoCapture(0)

        # Set camera width and height to 800x600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width != self.video_width or height != self.video_height:
            print(f"Warning: The requested resolution {self.video_width}x{self.video_height} is not supported.")
            #print(f"Camera resolution is set to {width}x{height} instead.")
            print(f"frames resize to {self.video_width}x{self.video_height}.")
        else:
            print(f"Camera resolution set to {width}x{height}.")

        self.recorded_frames = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #self.out = cv2.VideoWriter(self.recorded_video_path, fourcc, 25.0, (width, height))
        self.out = cv2.VideoWriter(self.recorded_video_path, fourcc, self.fps, (self.video_width, self.video_height))
        
        while self.recording:
        #while self.recording and (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if ret:
                if width != self.video_width or height != self.video_height:
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                self.recorded_frames.append(frame)
                self.out.write(frame)
                cv2.imshow('Recording', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.recording = False
                    break
            else:
                break

        cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.insert_recorded_video()

    def insert_recorded_video(self):
        start_frame, end_frame = self.get_marked_frame_range()
        self.insert_video_segment(start_frame, end_frame,'recorded')

    def get_marked_frame_range(self):
        # Ensure that exactly two marks are selected0
        if len(self.selected_marks) != 2:
            print("Exactly two marks must be selected to determine order.")
            return

        # Retrieve the indices of the selected marks
        start_index, end_index = self.selected_marks

        # Retrieve the frame numbers for the selected marks
        start_frame = self.marked_positions[start_index][0]
        end_frame = self.marked_positions[end_index][0]

        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        
        return start_frame, end_frame

    def insert_video_segment(self,start_frame, end_frame, flag):
        if start_frame is None or end_frame is None:
            return
        
        if len(self.marked_positions) < 2:
            return

        old_video_frame_num = len(self.video_frames)
        if flag == 'recorded':
            if self.video_synthesis_flag:
                video_synthesis.run_remote(self.recorded_video_path,self.ref_img_path,self.synthetic_video_path,(self.video_width,self.video_height))
                self.synthetic_video = cv2.VideoCapture(self.synthetic_video_path)
                self.synthetic_video_frames = []
                while True:
                    success, frame = self.synthetic_video.read()
                    if not success:
                        break
                    self.synthetic_video_frames.append(frame)
                 
                self.video_frames = (self.video_frames[:start_frame] + self.synthetic_video_frames + self.video_frames[end_frame:])
            else:
                self.video_frames = (self.video_frames[:start_frame] + self.recorded_frames + self.video_frames[end_frame:])
        
        elif flag == 'uploaded':
            if self.video_synthesis_flag:

                video_synthesis.run_remote(self.uploaded_video_path,self.ref_img_path,self.synthetic_video_path,(self.video_width,self.video_height))
                self.synthetic_video = cv2.VideoCapture(self.synthetic_video_path)
                self.synthetic_video_frames = []
                while True:
                    success, frame = self.synthetic_video.read()
                    if not success:
                        break
                    self.synthetic_video_frames.append(frame)
                 
                self.video_frames = (self.video_frames[:start_frame] + self.synthetic_video_frames + self.video_frames[end_frame:])
            else:
                self.video_frames = (self.video_frames[:start_frame] + self.uploaded_frames + self.video_frames[end_frame:])
            
        new_video_frame_num = len(self.video_frames)
        # Check frame size consistency
        for i, frame in enumerate(self.video_frames):
            if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
                print(f"Frame {i} has incorrect size: {frame.shape}. Resizing to ({self.video_height}, {self.video_width})")
                self.video_frames[i] = cv2.resize(frame, (self.video_width, self.video_height))


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.video_width, self.video_height))

        for frame in self.video_frames:
            new_video.write(frame)

        new_video.release()
        # Reinitialize the cv2.VideoCapture object with the new video file
        self.video = cv2.VideoCapture(self.output_video_path)
        self.file_path = self.output_video_path
        self.display_first_frame()
        self.after_replacing_new_video(old_video_frame_num, new_video_frame_num)
        self.pose_estimator = True

    def after_replacing_new_video(self,old_video_frame_num, new_video_frame_num):
        self.seek_video(0)
        self.slider.destroy()
        self.slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video, style="Horizontal.TScale")
        self.slider.place(x = self.canvas_width * 0.545, y = 2 * 20 + self.video_height, anchor=tk.CENTER, width=self.video_width)
        self.slider.config(to=self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.update_marks_after_insertion(old_video_frame_num, new_video_frame_num)
        self.skeleton, self.face_skeleton = mediapipe(self.file_path)
        self.destroy_speed_graph()
        self.movement_graph(self.video.get(cv2.CAP_PROP_FPS))


    def destroy_speed_graph(self):

        if self.speed_graph is not None:
            # Unplace the widget from the Tkinter canvas
            self.speed_graph.get_tk_widget().place_forget()
            
            # Destroy the widget
            self.speed_graph.get_tk_widget().destroy()
            
            # Optionally, clear the figure
            self.speed_graph.figure.clf()
            
            # Set self.speed_graph to None to indicate it has been destroyed
            self.speed_graph = None

    def display_first_frame(self):
        # Check if there are preloaded frames
        if self.video_frames:
            # Use the first frame in the preloaded list
            self.current_frame = self.video_frames[0]
            self.display_frame(self.current_frame)
            self.pause_video()  # Ensure the video is paused initially


    def play_video(self):
        if self.video:
            self.is_playing = True
            self.play_next_frame()

    def pause_video(self):
        self.is_playing = False

    def toggle_play_pause(self, event):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def stop_video(self):
        self.is_playing = False
        self.current_frame = None

    def play_next_frame(self):
        if self.video and self.is_playing:
            ret, frame = self.video.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                self.slider.set(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                self.root.after(10, self.play_next_frame)  # Update frame every 10 milliseconds (100 fps)
            else:
                self.stop_video()

    def update_segment_mode_canvas(self,event):
        # Calculate the new position based on the canvas size
        canvas_width = event.width
        canvas_height = event.height

        # Determine the position you want, e.g., centering
        x = (canvas_width-self.video_width)/2
        y =   self.video_height + 20 # Adjusted to have some margin from the top
        # Update the image position on the canvas
        self.segment_mode_canvas.coords(self.segment_mode_video, x, y)

        for line in self.mark_lines:

            slider_length = self.video_width
            slider_position = self.mark_line_ids[line]
            start_y = self.video_height + 20
            end_y = self.video_height + 20 + 20
            self.segment_mode_canvas.coords(line, slider_position + (canvas_width - slider_length) /2, start_y, slider_position + (canvas_width - slider_length) /2, end_y)
        
    def display_frame(self, frame):
        if frame is not None:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)

            self.segment_mode_video = self.segment_mode_canvas.create_image((self.canvas_width-self.video_width)/2, 20, anchor=tk.NW, image=photo)
            self.segment_mode_canvas.image = photo
            self.frame_label.config(text=f"Frame: {int(self.video.get(cv2.CAP_PROP_POS_FRAMES))}")
            current_time = int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) / self.fps
            self.time_label.config(text=f"Time: {current_time:.2f}")

    def seek_video(self, value):
        if self.video:
            frame_number = float(value)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseEditor(root)
    root.minsize(min_canvas_width, min_canvas_height)
    root.mainloop()
