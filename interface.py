import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, font
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import yaml
from datetime import datetime
import threading
import tempfile
from remote_controlnext import ControlNeXt
from face_swapper import facefusion
from frame_interpolation import frame_interpolate
from reading_video import read_video
import queue
import time

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

gray_color = '#3c3c3c'
background_color = '#464a4d'
sidebar_color = '#333739'
button_color = '#bebebe'
button_activate = '#646464'
gold_color = "#FFC733"

min_canvas_width = 1300
min_canvas_height = 750

def save_first_and_last_frames(video_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # Read the first frame
    # Skip to the second frame (index 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame.")
    first_frame_path = os.path.join(output_dir, 'first_frame.png')
    cv2.imwrite(first_frame_path, first_frame)

    # Get the total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the capture to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the last frame.")
    last_frame_path = os.path.join(output_dir, 'last_frame.png')
    cv2.imwrite(last_frame_path, last_frame)

    cap.release()

def next_divisible_by_64(n):
    return int(((n // 64) + 1) * 64 if n % 64 != 0 else n)

def hex_to_bgr(hex_color):
    """Convert a hex color string to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr

def resize_video(input_path, output_path, target_width, target_height):
    # Open input video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Preserve original FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when frames are done
        
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap.release()
    out.release()

# Get the current time in a suitable format for a folder name
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

class PoseEditor(tk.Tk):
    def __init__(self, root):
        self.root = root
        self.root.title("Post-Editing Interface")
        self.root.configure(bg = background_color)  # Set the background color to gray

        self.style = ttk.Style()
        self.style.configure("TButton",
                padding=0,
                relief="flat",
                background = background_color,   # Button background color
                foreground = "#000000",   # Button text color
                bordercolor = gray_color,  # Button border color
                highlightthickness=0)          # Highlight border thickness)     # Button font)   # Focus border color

        # Load the YAML file
        with open('config.yaml', 'r') as file:
            self.interface_config = yaml.safe_load(file)
        # Set up a canvas with the desired resolution
        self.canvas_width = self.interface_config['canvas_width']
        self.canvas_height = self.interface_config['canvas_height']
        self.sidebar_width = 60
        self.offset = self.interface_config['offset']
        self.fps = self.interface_config['fps']
        self.video = None
        self.recording = False
        self.out = None
        self.recording_thread = None
        self.working_directory = os.path.join(self.interface_config['working_directory'],current_time)
        os.makedirs(self.working_directory, exist_ok=True)
        self.output_path = self.working_directory + '/output.mp4'
        self.is_playing = False
        self.slider = None
        self.frame_time_frame = None
        self.frame_label = None
        self.time_label = None
        self.video_frames = []
        self.pose_video_frames = []
        self.pose_video = None
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

        self.segment_mode_canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg = background_color, highlightbackground='#555555', highlightthickness=2)
        self.segment_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.segment_mode_canvas.pack_forget()
        #self.segment_mode_canvas.bind("<Configure>", self.update_segment_mode_canvas)

        self.sidebar = tk.Frame(self.main_frame, width=self.sidebar_width, bg=sidebar_color)
        self.sidebar.pack_propagate(False)  # Prevent sidebar from resizing
        self.segment_mode_sidebar = tk.Frame(self.main_frame, width=self.sidebar_width, bg=sidebar_color)

        self.recording_queue = queue.Queue()  # To pass frames from capture thread -> main thread
        self.recording_window = None          # We create a separate Toplevel for preview
        self.recording_canvas = None
        self.recording_canvas_id = None
        #self.root.bind_all("<KeyPress-q>", self.on_q_pressed)


        self.load_logo()
        self.load_image_button()

    def load_logo(self):
        load_logo = Image.open(self.interface_config['images_directory']+"/chair.png")  # Replace with your image path
        _, img_height = load_logo.size
        self.logo = ImageTk.PhotoImage(load_logo)
        self.logo_first = self.canvas.create_image(self.canvas_width/2, self.canvas_height-img_height, image=self.logo, anchor=tk.CENTER)

    def load_image_button(self):
        # Load the image
        load_img1 = Image.open(self.interface_config['images_directory']+"/upload_video.png")  # Replace with your image path
        self.img1 = ImageTk.PhotoImage(load_img1)
        button_width, button_height = load_img1.size
        # Create a canvas item for the image
        self.img_button1 = self.canvas.create_image(self.canvas_width/2, self.canvas_height/2, image=self.img1, anchor=tk.CENTER)
        # Bind the click event to the canvas item
        self.canvas.tag_bind(self.img_button1, '<Button-1>', self.load_video)
        self.canvas.tag_bind(self.img_button1,"<Enter>", self.on_enter)
        self.canvas.tag_bind(self.img_button1,"<Leave>", self.on_leave)

        # ==== Place Configuration UI Directly on the First Canvas ====

        # ==== Initialize Default Configuration Values ====
        self.interface_config['method'] = 'Segment'  # Default method
        self.interface_config['sample_stride'] = 1  # Default sample stride
        self.interface_config['face_swapp'] = False  # Default face swap

        # Define font
        custom_font = font.Font(family="Calibri Light", size=12)

        # Measure text width
        method_text_width = custom_font.measure("Method:")
        sample_stride_text_width = custom_font.measure("Sample Stride:")

        # Base positions
        base_x = self.canvas_width // 2 +30
        base_y = self.canvas_height // 2 + button_height

        # Face Swap Checkbox
        self.face_swapp_var = tk.BooleanVar(value=False)
        self.face_swapp_check = tk.Checkbutton(
            self.root, text="Face Swap", variable=self.face_swapp_var,
            bg=background_color, fg=button_color, activebackground=background_color)
        self.face_swapp_check.place(x=base_x, y=base_y, anchor=tk.CENTER)

        self.frame_interpolation_var = tk.BooleanVar(value=True)
        self.frame_interpolation_check = tk.Checkbutton(
            self.root, text="Frame Interpolattion", variable=self.frame_interpolation_var,
            bg=background_color, fg=button_color, activebackground=background_color)
        self.frame_interpolation_check.place(x=base_x, y=base_y+30, anchor=tk.CENTER)

        # Method Selection Dropdown
        self.method_var = tk.StringVar(value='Segment')
        self.method_label = tk.Label(self.root, text="Method:", bg=background_color, fg=button_color)
        self.method_label.place(x=base_x - method_text_width, y=base_y + 30*2, anchor=tk.CENTER)

        self.method_combo = ttk.Combobox(self.root, textvariable=self.method_var, values=["Segment", "Full Video"], state="readonly", width=15)
        self.method_combo.place(x=base_x + 15 + 10 , y=base_y + 30*2, anchor=tk.CENTER)
        self.method_combo.bind("<<ComboboxSelected>>", self.update_settings_based_on_method)  # Bind event

        # Sample Stride Input (Spinbox)
        self.sample_stride_var = tk.IntVar(value=1)
        self.sample_stride_label = tk.Label(self.root, text="Sample Stride:", bg=background_color, fg=button_color)
        self.sample_stride_label.place(x=base_x - sample_stride_text_width + 15, y=base_y + 30*3, anchor=tk.CENTER)

        self.sample_stride_spinbox = tk.Spinbox(self.root, from_=1, to=9999, textvariable=self.sample_stride_var, width=15)
        self.sample_stride_spinbox.place(x=base_x + 15 + 10, y=base_y + 30*3, anchor=tk.CENTER)

    def update_settings_based_on_method(self, event=None):
        """Updates sample_stride and face_swapp when method changes"""
        selected_method = self.method_var.get()  # Get selected method

        if selected_method == "Segment":
            self.sample_stride_var.set(1)  # Set sample stride to 1
            self.face_swapp_var.set(False)  # Disable face swap
        elif selected_method == "Full Video":
            self.sample_stride_var.set(2)  # Set sample stride to 2
            self.face_swapp_var.set(True)  # Enable face swap


    def update_canvas(self, event):
        load_logo = Image.open(self.interface_config['images_directory']+"/chair.png")  # Replace with your image path
        _, img_height = load_logo.size
        canvas_width = event.width
        canvas_height = event.height

        self.canvas.coords(self.logo_first, canvas_width/2, canvas_height- img_height)
        self.canvas.coords(self.img_button1, canvas_width/2 , canvas_height/2)

    def load_video(self, event=None):

        file_path = filedialog.askopenfilename()
        self.file_path = file_path

        # Save user settings into config
        self.interface_config['face_swap'] = self.face_swapp_var.get()
        self.interface_config['frame_interpolation'] = self.frame_interpolation_var
        selected_method = self.method_var.get()
        # Map user-friendly labels to internal method names
        method_map = {"Segment": "part", "Full Video": "entire"}
        self.interface_config['method'] = method_map[selected_method]
        #self.interface_config['method'] = self.method_var.get()
        self.interface_config['sample_stride'] = self.sample_stride_var.get()

        # Hide UI elements after selecting a video
        self.face_swapp_check.place_forget()
        self.frame_interpolation_check.place_forget()
        self.method_label.place_forget()
        self.method_combo.place_forget()
        self.sample_stride_label.place_forget()
        self.sample_stride_spinbox.place_forget()

        if file_path:
            self.video = cv2.VideoCapture(file_path)
            if self.video.get(cv2.CAP_PROP_FRAME_WIDTH)%64 or self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)%64:
                width = next_divisible_by_64(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = next_divisible_by_64(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resize_video(self.file_path,self.file_path[:-4]+'_resized.mp4',width, height)
                self.file_path = self.file_path[:-4]+'_resized.mp4'
                self.video.release()
                self.video = cv2.VideoCapture(self.file_path)

            self.video_width , self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not self.video.isOpened():
                print("Error: Could not open video.")
                return

            ret, frame = self.video.read()
            if not ret:
                print("Error: Could not read the first frame from video.")
                return
            
            _, frame = self.video.read()
            cv2.imwrite(self.working_directory + '/ref_img.png', frame)

            self.canvas.delete(self.img_button1)
            self.canvas.delete(self.logo_first)
            self.edit_segments()

    def on_enter(self, event):
        event.widget.config(cursor="hand2")  # Change the cursor to a hand

    def on_leave(self, event):
        event.widget.config(cursor="")  # Revert the cursor to the default

    def edit_segments(self):
        self.canvas.pack_forget()
        self.segment_mode_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.segment_mode_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.edit_segment_sidebar()
        self.setup_layout()
        
    def button_file(self):
        file_menu = tk.Menu(self.root, tearoff=0, bg = sidebar_color, fg= button_color, font=("Calibri Light", 12), activebackground= button_activate, borderwidth= 0)
        file_menu.add_command(label="Save Keypoints", command=lambda: self.save_keypoints())
        file_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    ################### SEGMENT MODE #####################

    def edit_segment_sidebar(self):

        self.segment_mode_sidebar_buttons = []
        self.segment_mode_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.segment_mode_sidebar.pack_propagate(False)

        load_logo = Image.open(self.interface_config['images_directory']+"/sidebar_logo.png")  # Replace with your image path
        self.logo_img = ImageTk.PhotoImage(load_logo)
        self.logo_label = tk.Label(self.segment_mode_sidebar, image=self.logo_img, bg=sidebar_color)
        self.logo_label.pack(side=tk.TOP, pady=10)  # Adjust x and y to position the logo correctly

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

    def segment_edit_new_video(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.file_path = file_path
            
            # 🔹 Ensure the previous video is properly closed before loading a new one
            if self.video is not None:
                self.video.release()  # Release previous video resources

            self.video_frames = []
            # 🔹 Load the new video
            self.video = cv2.VideoCapture(file_path)
            
            if not self.video.isOpened():
                print("Error: Could not load video.")
                return  # Exit the function if the video fails to load

            # 🔹 Clear all old marks (buttons + lines)
            for button in self.mark_buttons:
                button.destroy()
            self.mark_buttons.clear()
            
            for line in self.mark_lines:
                self.segment_mode_canvas.delete(line)
            self.mark_lines.clear()
            self.marked_positions.clear()
            self.selected_marks.clear()

            # 🔹 Reset inserted video path (IMPORTANT)
            self.inserted_video_path = None

            # 🔹 Force refresh UI elements
            self.segment_mode_canvas.delete("all")  # Clears all drawings from canvas
            self.canvas.delete("all")  # Clears main canvas
            
            # 🔹 Reinitialize video display
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to first frame
            self.setup_layout()  # Refresh UI elements

            print("New video loaded successfully!")
    
    def save_video(self):

        save_path = filedialog.asksaveasfilename(defaultextension=".mp4",filetypes=[("MP4 video files", "*.mp4"), ("All files", "*.*")])
    
        if save_path:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Get width, height, and FPS
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            while True:
                success, frame = self.video.read()
                if not success:
                    break
                writer.write(frame)

            self.video.release()
            writer.release()
            print(f"Video saved to {save_path}")


    def setup_layout(self):
        
        self.segment_mode_canvas.bind("<Button-1>", self.toggle_play_pause)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale", background=background_color, troughcolor="#bfbfbf", sliderlength=20)

        self.slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video, style="Horizontal.TScale")

        self.frame_label = tk.Label(self.root, text="Frame: 0", font=("Calibri Light", 12), background=background_color, fg= button_color)

        self.time_label = tk.Label(self.root, text="Time: 0.00", font=("Calibri Light", 12),background=background_color, fg= button_color)


        if len(self.video_frames) == 0:
            while True:
                success, frame = self.video.read()
                if not success:
                    break
                
                self.video_frames.append(frame)

        self.fig_height_inch = 1.2
        # Desired DPI (dots per inch)
        self.dpi = 100
        self.load_button_images()
        self.segment_mode_canvas.config(width=self.canvas_width, height = self.canvas_height)# Place the time frame bar
        #self.slider.place(x = self.canvas_width * 0.545, y = 2 * 20 + self.video_height, anchor=tk.CENTER, width=self.video_width)
        self.slider.place(x = self.canvas_width/2 + self.sidebar_width, y = (self.canvas_height+self.video_height)/2 + 20 - self.offset, anchor=tk.CENTER, width=self.video_width)
        # Place the frame label
        self.frame_label.place(x = 0.60 * self.canvas_width + self.sidebar_width , y = 2 * 20 + (self.canvas_height+self.video_height)/2 - self.offset, anchor=tk.CENTER)
        # Place the time label
        self.time_label.place(x= 0.40 * self.canvas_width + self.sidebar_width, y = 2 * 20 + (self.canvas_height+self.video_height)/2 - self.offset, anchor=tk.CENTER)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.slider.config(to=self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.display_first_frame()
        self.create_buttons()

    def load_button_images(self):
        self.button_images = []
        button_files = ["mark.png", "insert_video.png", "recording.png"]  # Replace with your image paths
        for file in button_files:
            image = Image.open(os.path.join(self.interface_config['images_directory'], file))
            photo = ImageTk.PhotoImage(image)
            self.button_images.append(photo)
        self.button_img_width, _ = image.size
        self.segment_mode_button_width, self.segment_mode_button_height = image.size

    def create_buttons(self):
        button_commands = [self.mark, self.insert_uploaded_video, self.toggle_recording]  # Define your button commands
        for i, image in enumerate(self.button_images):
            button = tk.Button(self.root, image=image, command=button_commands[i], borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
            button.place(x=int(self.canvas_width/2 + (i - 1) * self.button_img_width*1.5) + self.sidebar_width , y = (self.canvas_height+self.video_height)/2 + 2*20 + self.fig_height_inch * self.dpi/2 - self.offset, anchor=tk.CENTER)  # Adjust positioning as needed
            self.segment_mode_buttons.append(button)

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
        button_text = f"Flag{c}: Frame: {frame_number}, Time: {current_time:.2f}"
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
        
        button.place(x = 50, y = int(self.canvas_height/2 - self.video_height/4) + 25 * c - self.offset, anchor=tk.NW)
        # Store the button reference
        self.mark_buttons.append(button)
        # Draw a red line on the slider
        self.add_line(frame_number, mark_color)
    
    def add_line(self, frame_number, color):
        slider_length = self.slider.winfo_width()
        total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

        if total_frames <= 0:
            return  # Prevent division errors
        
        # 🔹 Corrected: Calculate precise x position without shifting it too far
        slider_position = (frame_number / total_frames) * slider_length

        # 🔹 Ensure alignment with slider placement
        x = (self.canvas_width - slider_length) / 2 + slider_position 

        start_y = (self.video_height + self.canvas_height) / 2
        end_y = (self.video_height + self.canvas_height) / 2 + 20
        
        line = self.segment_mode_canvas.create_line(
            x, start_y - self.offset, x, end_y - self.offset,
            fill=color, width=2
        )
        
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
            button.place(x = 50, y = int(self.canvas_height/2 - self.video_height/4) + 25 * (i + 1) - self.offset, anchor=tk.NW)
            # Update the button command to reflect new index
            button.config(command=lambda btn=button, idx=i: self.select_mark(idx, btn))
            button.bind("<Button-3>", lambda event, idx=i: self.show_context_menu(event, idx))

        # Update the text of the buttons to reflect new indices
        for i, (frame_number, current_time) in enumerate(self.marked_positions):
            self.mark_buttons[i].config(text=f"Flag{i+1}: Frame: {frame_number}, Time: {current_time:.2f}")

    def update_marks_after_insertion(self, old_frame_count, new_frame_count):
        """Updates marks and lines after a video segment is inserted."""
        del_list = []
        frame_difference = new_frame_count - old_frame_count
        start_index, end_index = self.selected_marks

        # Retrieve the frame numbers for the selected marks
        start_frame = self.marked_positions[start_index][0]
        end_frame = self.marked_positions[end_index][0]

        # Update marked positions
        counter = 0
        for i, (frame_number, _) in enumerate(self.marked_positions):

            if frame_number == start_frame or frame_number == end_frame:

                if frame_number <= start_frame:
                    new_frame_number = frame_number
            
                else:
                    new_frame_number = frame_number + frame_difference

                new_time = new_frame_number / self.fps
                self.marked_positions[i] = (new_frame_number, new_time)
                # Update the button text
                self.mark_buttons[i].config(
                    text=f"Flag{counter + 1}: Frame: {new_frame_number}, Time: {new_time:.2f}",
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
            button.place(x = 50, y = int(self.canvas_height/2 - self.video_height/4) + 25 * (i + 1), anchor=tk.NW)

        self.selected_marks = []

    def update_line_position(self, index, frame_number,new_frame_count,color):
        slider_length = self.video_width
        #slider_position = int((frame_number / self.video.get(cv2.CAP_PROP_FRAME_COUNT)) * slider_length)
        slider_position = int((frame_number / new_frame_count) * slider_length)
        #print('frame_number:{}, slider_length:{},slider_position'.format(frame_number, slider_length,slider_position))
        # Delete the old line and draw a new one
        self.segment_mode_canvas.delete(self.mark_lines[index])
        x = (self.canvas_width - slider_length) / 2 + slider_position 
        start_y = (self.video_height + self.canvas_height) / 2 
        end_y = (self.video_height + self.canvas_height) / 2 + 20 
        line = self.segment_mode_canvas.create_line(
            x, start_y- self.offset, x, end_y- self.offset,
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
        uploaded_video_path = filedialog.askopenfilename()
        if not uploaded_video_path:
            print("No inserted video selected.")
            return
        
        # Open trimming window
        trim_window = VideoTrimmerWindow(self.root, uploaded_video_path)
        self.root.wait_window(trim_window)  # Wait for trimming to complete

        if hasattr(trim_window, 'trimmed_video'):
            self.inserted_video_path = trim_window.trimmed_video
            # print('self.inserted_video_path', self.inserted_video_path)
            self.insert_video_segment()


    def toggle_recording(self, event=None):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            # === NEW: Create the preview Toplevel and canvas
            self.setup_recording_preview()

            # Spawn background thread to capture frames
            self.recording_thread = threading.Thread(target=self.record_video, daemon=True)
            self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None

        # === NEW: Close preview window
        if self.recording_window is not None:
            self.recording_window.destroy()
            self.recording_window = None

                # Open trimming window
        trim_window = VideoTrimmerWindow(self.root, self.working_directory+'/recorded_video.mp4')
        self.root.wait_window(trim_window)  # Wait for trimming to complete

        if hasattr(trim_window, 'trimmed_video'):
            self.inserted_video_path = trim_window.trimmed_video
            # print('self.inserted_video_path', self.inserted_video_path)
            self.insert_video_segment()

    def setup_recording_preview(self):
        """Create a Toplevel with a Canvas to show live frames."""
        if self.recording_window is not None:
            return  # Already open

        self.recording_window = tk.Toplevel(self.root)
        self.recording_window.title("Recording Preview")

        # You could set a geometry if you like:
        # self.recording_window.geometry("650x550")

            # Force the Toplevel to receive key events
        self.recording_window.focus_set()

        # Bind “q” so pressing q in the Toplevel calls on_q_pressed
        self.recording_window.bind("<KeyPress-q>", self.on_q_pressed)

        # Create a canvas for live video frames
        self.recording_canvas = tk.Canvas(
            self.recording_window, width=self.video_width, height=self.video_height, bg="black"
        )
        self.recording_canvas.pack()

        # Start polling the queue to update frames
        self.update_recording_preview()

    def update_recording_preview(self):
        """
        Runs on the main thread every ~10ms:
        - Checks the queue for any new frame,
        - Displays it on the new preview canvas.
        """
        if not self.recording_queue.empty():
            frame = self.recording_queue.get()

            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            tk_img = ImageTk.PhotoImage(pil_img)

            if not self.recording_canvas_id:
                # First-time creation
                self.recording_canvas_id = self.recording_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            else:
                # Update existing image
                self.recording_canvas.itemconfig(self.recording_canvas_id, image=tk_img)

            # Keep a reference to avoid garbage-collection
            self.recording_canvas.image = tk_img

        # Schedule again if the window still exists
        if self.recording_window is not None and self.recording_window.winfo_exists():
            self.root.after(10, self.update_recording_preview)


    def record_video(self):
        """
        Background thread: captures from camera, saves to file,
        passes frames to main thread via self.recording_queue.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width != self.video_width or height != self.video_height:
            print(f"Warning: The requested resolution {self.video_width}x{self.video_height} is not supported.")
            print(f"Frames will be resized to {self.video_width}x{self.video_height}.")
        else:
            print(f"Camera resolution set to {width}x{height}.")

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.working_directory, 'recorded_video.mp4')
        self.out = cv2.VideoWriter(out_path, fourcc, self.fps, (self.video_width, self.video_height))

        print("Recording started... Press 'q' to end.")
        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Resize if actual camera size differs
            if (width, height) != (self.video_width, self.video_height):
                frame = cv2.resize(frame, (self.video_width, self.video_height))

            # Write to file
            self.out.write(frame)

            # === CHANGED: Instead of cv2.imshow() in background thread,
            #     we push frames into a queue for the main thread to display
            self.recording_queue.put(frame)

            time.sleep(0.01)  # Tiny delay to reduce CPU usage

        # Cleanup
        cap.release()
        self.out.release()
        print(f"Recording finished. Video saved to: {out_path}")
        

    def on_q_pressed(self, event):
        """Stop recording if user presses 'q'."""
        if self.recording:
            self.stop_recording()
        

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

    def insert_video_segment(self):
        
        if len(self.marked_positions) < 2:
            return

        old_video_frame_num = len(self.video_frames)
        s_f , e_f = self.get_marked_frame_range()

        start_frame_address = self.working_directory+'/s_f.png'
        end_frame_address = self.working_directory+'/e_f.png'

        cv2.imwrite(start_frame_address, self.video_frames[s_f])
        cv2.imwrite(end_frame_address, self.video_frames[e_f])

        print('method:',self.interface_config['method'])
        print('stride:',self.interface_config['sample_stride'])
        print('face swapp:',self.interface_config['face_swap'])

        if self.interface_config['method'] == 'part':
            ControlNeXt(self.file_path, self.inserted_video_path, (s_f, e_f), start_frame_address, end_frame_address, self.working_directory + '/ref_img.png', self.working_directory + '/generated_video.mp4', (self.video_width, self.video_height), self.interface_config['method'], self.interface_config['sample_stride'], self.interface_config['s3it'])

        elif self.interface_config['method'] == 'entire':
            ControlNeXt(self.file_path, self.inserted_video_path, (s_f, e_f), start_frame_address, end_frame_address, self.working_directory + '/ref_img.png', self.working_directory + '/generated_video.mp4', (self.video_width, self.video_height), self.interface_config['method'], self.interface_config['sample_stride'], self.interface_config['s3it'])

        if self.interface_config['frame_interpolation']:
            save_first_and_last_frames(video_path= self.working_directory + '/generated_video.mp4',output_dir=self.working_directory)
            fgf = self.working_directory + '/first_frame.png'
            lgf = self.working_directory + '/last_frame.png'
            frame_interpolate(fgf, lgf, start_frame_address, end_frame_address, self.working_directory,self.interface_config['num_frame'], self.interface_config, (self.video_width,self.video_height))
            self.first_trans_frames, _ = read_video(self.working_directory + '/f_trans.mp4')
            self.second_trans_frames, _ = read_video(self.working_directory + '/s_trans.mp4')

        if self.interface_config['face_swap']:
            facefusion(self.working_directory + '/generated_video.mp4', self.working_directory + '/ref_img.png', self.working_directory + '/generated_video.mp4', self.interface_config)

        self.generated_video_frames, fps_ = read_video(self.working_directory + '/generated_video.mp4')

        if self.interface_config['method'] == 'part':
            #self.video_frames = (self.video_frames[:s_f:int(self.fps//fps_)] + self.generated_video_frames + self.video_frames[e_f::int(self.fps//fps_)])
            if self.interface_config['frame_interpolation']:
                self.video_frames = (self.video_frames[:s_f] + self.first_trans_frames +self.generated_video_frames + self.second_trans_frames+ self.video_frames[e_f:])
            else:
                self.video_frames = (self.video_frames[:s_f] + self.generated_video_frames + self.video_frames[e_f:])

            
        else:
            self.video_frames = self.generated_video_frames

        self.fps = min(self.fps,fps_)
 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if self.interface_config['method'] == 'part':
            new_video = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.video_width, self.video_height))
        else:
            new_video = cv2.VideoWriter(self.output_path, fourcc, fps_, (self.video_width, self.video_height))
        
        for frame in self.video_frames:
            new_video.write(frame)

        new_video.release()
        # Reinitialize the cv2.VideoCapture object with the new video file
        self.video = cv2.VideoCapture(self.output_path)
        self.file_path = self.output_path

        self.display_first_frame()
        self.after_replacing_new_video(old_video_frame_num, len(self.video_frames))

    def after_replacing_new_video(self,old_video_frame_num, new_video_frame_num):
        self.seek_video(0)
        self.slider.destroy()
        self.slider = ttk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video, style="Horizontal.TScale")
        self.slider.place(x = self.canvas_width/2 + self.sidebar_width, y = (self.canvas_height+self.video_height)/2 + 20 - self.offset, anchor=tk.CENTER, width=self.video_width)
        self.slider.config(to=self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.update_marks_after_insertion(old_video_frame_num, new_video_frame_num)

    def display_first_frame(self):
        # Check if there are preloaded frames
        if self.video_frames:
            # Use the first frame in the preloaded list
            self.current_frame = self.video_frames[0]
            self.display_frame(self.current_frame)
            self.pause_video()


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
                self.display_frame(frame)  # Display only main video if pose video fails
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
            start_y = (self.canvas_height-self.video_height)/2
            end_y = (self.canvas_height-self.video_height)/2 +20
            self.segment_mode_canvas.coords(line, slider_position + (canvas_width - slider_length) /2, start_y - self.offset, slider_position + (canvas_width - slider_length) /2, end_y- self.offset)
        
    def display_frame(self, frame):
        if frame is not None:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            
            self.segment_mode_video = self.segment_mode_canvas.create_image((self.canvas_width-self.video_width)/2, (self.canvas_height-self.video_height)/2- self.offset, anchor=tk.NW, image=photo)
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

# Add this new class within the interface.py file
class VideoTrimmerWindow(tk.Toplevel):
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.parent = parent
        self.title("Video Trimmer")

        with open('config.yaml', 'r') as file:
            self.interface_config = yaml.safe_load(file)
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.sidebar_width = 60

        # 🔹 Load all frames for easy display & seeking
        self.video_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.video_frames.append(frame)
        self.cap.release()

        # We'll track the current frame index & playing state
        self.current_index = 0
        self.is_playing = False
        self.mark_lines = []

        # Load button images
        flag_img = Image.open(os.path.join(self.interface_config['images_directory'], "flag.png"))
        self.button_width, self.button_height = flag_img.size

        self.canvas_width = self.video_width
        self.canvas_height = self.video_height + self.button_height + 40
        
        self.geometry(f"{self.canvas_width}x{self.canvas_height}")
        self.configure(bg=background_color)
        # Video display
        self.canvas = tk.Canvas(self, bg=background_color, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        
        # 🔹 Bind left-click to toggle play/pause
        self.canvas.bind("<Button-1>", self.toggle_play_pause)
        
        self.flag_img = ImageTk.PhotoImage(flag_img)
        mark_btn = tk.Button(self.canvas, image=self.flag_img, command=self.add_mark, borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
        mark_btn.place(x= int(self.canvas_width/2 - 1.5 * self.button_width*1.5) , y = self.video_height + 3*20, anchor=tk.CENTER)

        self.cut_img = ImageTk.PhotoImage(Image.open(os.path.join(self.interface_config['images_directory'], "scissors.png")))
        cut_btn = tk.Button(self.canvas, image=self.cut_img, command=self.cut_segment, borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
        cut_btn.place(x= int(self.canvas_width/2 - 0.5 * self.button_width*1.5) , y = self.video_height + 3*20, anchor=tk.CENTER)

        self.approve_img = ImageTk.PhotoImage(Image.open(os.path.join(self.interface_config['images_directory'], "save_file.png")))
        approve_btn = tk.Button(self.canvas, image=self.approve_img, command=self.approve, borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
        approve_btn.place(x=int(self.canvas_width/2 + 0.5 * self.button_width*1.5) , y = self.video_height + 3*20, anchor=tk.CENTER)

        self.redo_img = ImageTk.PhotoImage(Image.open(os.path.join(self.interface_config['images_directory'], "redo.png")))
        redo_btn = tk.Button(self.canvas, image=self.redo_img, command=self.redo, borderwidth=0, background= background_color, activebackground= background_color, activeforeground= background_color)
        redo_btn.place(x=int(self.canvas_width/2 + 1.5 *self.button_width*1.5) , y = self.video_height + 3*20, anchor=tk.CENTER)
    
        # Slider
        self.slider = ttk.Scale(self.canvas, from_=0, to=len(self.video_frames)-1, command=self.seek_video, orient=tk.HORIZONTAL, style="Horizontal.TScale")
        self.slider.place(x = self.video_width/2, y = self.video_height + 20, anchor=tk.CENTER, width=self.video_width)
        
        # Show the first frame right away
        self.display_first_frame()

    def add_mark(self):

        # 1) If we already have two lines, do nothing
        if len(self.mark_lines) >= 2:
            print("Already have two marks. No more marks allowed.")
            return

        frame_number = self.current_index
        total_frames = len(self.video_frames)
        if total_frames == 0:
            return

        line_x = frame_number / float(total_frames) * self.video_width

        # Draw a short line near the slider
        start_y = self.video_height
        end_y   = self.video_height + 20

        line_id = self.canvas.create_line(
            line_x, start_y,
            line_x, end_y,
            fill=gold_color, width=2
        )
        self.mark_lines.append(line_id)

        
    def toggle_play_pause(self, event):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        self.is_playing = True
        self.play_next_frame()

    def pause_video(self):
        self.is_playing = False

    def play_next_frame(self):
        if not self.is_playing:
            return
        self.current_index += 1
        if self.current_index >= len(self.video_frames):
            self.is_playing = False
            return
        frame = self.video_frames[self.current_index]
        self.display_frame(frame)
        self.slider.set(self.current_index)
        # Schedule next frame
        self.after(33, self.play_next_frame)  # ~30 fps

    def display_first_frame(self):
        if self.video_frames:
            self.current_index = 0
            self.display_frame(self.video_frames[self.current_index])

    def display_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def seek_video(self, value):
        """Called whenever user drags the slider."""
        self.current_index = int(float(value))
        if 0 <= self.current_index < len(self.video_frames):
            self.display_frame(self.video_frames[self.current_index])

    def setup_preview_window(self):
        """Creates a separate Tkinter Toplevel for video preview."""
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Camera Preview")

        # We'll store the video frames here
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack()

        # If you want to resize or adjust geometry:
        # self.preview_window.geometry("800x600")
        
    def cut_segment(self):
        """
        Removes the frames between the two drawn lines in self.mark_lines,
        saves the result as a new '_cleaned' video, then reloads it for more edits.
        """
        # 1) Need exactly two lines
        if len(self.mark_lines) < 2:
            print("Error: You must have two lines before cutting.")
            return

        # 2) Retrieve the x-coords of each line
        #    We'll interpret them as fractions of self.video_width => frame indices
        line_ids = self.mark_lines[:2]  # just the first two
        coords_1 = self.canvas.coords(line_ids[0])  # (x1, y1, x2, y2)
        coords_2 = self.canvas.coords(line_ids[1])

        # Because you drew short vertical lines, x1 == x2 for each
        x_line1 = coords_1[0]  # x of first line
        x_line2 = coords_2[0]  # x of second line

        # 3) Convert x -> frame indices
        #    fraction = x / self.video_width
        #    frame_number = fraction * total_frames
        total_frames = len(self.video_frames)
        if total_frames == 0:
            print("No frames to cut.")
            return

        fraction1 = x_line1 / float(self.video_width)
        fraction2 = x_line2 / float(self.video_width)

        frame1 = int(round(fraction1 * (total_frames - 1)))
        frame2 = int(round(fraction2 * (total_frames - 1)))

        # sort them so start < end
        start_frame, end_frame = sorted([frame1, frame2])
        print(f"Cutting frames from {start_frame} to {end_frame} of total {total_frames}")

        if start_frame == end_frame:
            print("Warning: Both lines indicate the same frame. Nothing to cut.")
            return

        # 4) Keep only frames in [start_frame, end_frame]
        remain_frames = self.video_frames[start_frame : end_frame + 1]

        # 5) Close the original cap if still open
        if self.cap is not None:
            self.cap.release()

        # 6) Write the remain_frames to disk
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, self.fps,
                            (self.video_width, self.video_height))
        for f in remain_frames:
            out.write(f)
        out.release()

        # 7) Update internal references so we can continue editing
        self.video_frames = remain_frames
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = len(remain_frames)
        self.slider.config(to=self.total_frames - 1)

        # 8) Remove existing lines
        for line_id in self.mark_lines:
            self.canvas.delete(line_id)
        self.mark_lines.clear()
        print("Segment removed. Reloaded cleaned video for further edits.")

        # 9) Show the first frame of the cleaned clip
        self.display_first_frame()

        
    def approve(self):
        self.trimmed_video = self.video_path
        self.destroy()
        
    def redo(self):
        """
        When user clicks "Redo", remove all lines (marks),
        so user can start over.
        """
        for line_id in self.mark_lines:
            self.canvas.delete(line_id)
        self.mark_lines.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseEditor(root)
    root.minsize(min_canvas_width, min_canvas_height)
    root.mainloop()
