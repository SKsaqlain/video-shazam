import time
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import threading

class VideoPlayer:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.btn_play = ttk.Button(window, text="Play", command=self.play_video)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_pause = ttk.Button(window, text="Pause", command=self.pause_video)
        self.btn_pause.pack(side=tk.LEFT)

        self.btn_rewind = ttk.Button(window, text="Rewind", command=self.rewind_video)
        self.btn_rewind.pack(side=tk.LEFT)

        # Delay between frames in milliseconds. Adjust this based on the actual video FPS
        self.delay = int(1000 / self.fps)
        self.playing = False
        self.paused = False

        self.window.mainloop()

    def play_video(self):
        if not self.playing:
            self.playing = True
            self.paused = False
            threading.Thread(target=self.stream).start()

    def pause_video(self):
        self.paused = not self.paused

    def rewind_video(self):
        if self.vid.isOpened():
            time_sec = simpledialog.askfloat("Rewind", "Enter time in seconds to rewind to:", parent=self.window)
            if time_sec is not None:
                frame_no = time_sec * self.fps
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                self.update()

    def stream(self):
        while self.playing:
            if not self.paused:
                self.update()
                time.sleep(self.delay / 1000.0)

    def update(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.playing = False  # Stop playing if the video has ended

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the VideoPlayer class
root = tk.Tk()
VideoPlayer(root, "Tkinter Video Player", "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos/video1.mp4")
