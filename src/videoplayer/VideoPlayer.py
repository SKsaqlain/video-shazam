import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
import pygame

class VideoPlayer:
    def __init__(self, window, window_title, video_source, audio_source):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.audio_source = audio_source
        self.vid = cv2.VideoCapture(video_source)
        self.running = True

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Initialize pygame for audio
        pygame.init()
        pygame.mixer.init()

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_play = ttk.Button(window, text="Play", command=self.play_video)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_pause = ttk.Button(window, text="Pause", command=self.pause_video)
        self.btn_pause.pack(side=tk.LEFT)

        self.btn_rewind = ttk.Button(window, text="Rewind", command=self.rewind_video)
        self.btn_rewind.pack(side=tk.LEFT)

        self.delay = int(1000 / self.vid.get(cv2.CAP_PROP_FPS))
        self.playing = False
        self.paused = False

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close event

        self.window.mainloop()

    def play_video(self):
        if not self.playing:
            self.playing = True
            self.paused = False
            pygame.mixer.music.load(self.audio_source)
            pygame.mixer.music.play()
            threading.Thread(target=self.stream).start()

    def pause_video(self):
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    def rewind_video(self):
        if self.vid.isOpened():
            time_sec = simpledialog.askfloat("Rewind", "Enter time in seconds to rewind to:", parent=self.window)
            if time_sec is not None:
                frame_no = time_sec * self.vid.get(cv2.CAP_PROP_FPS)
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                pygame.mixer.music.load(self.audio_source)
                pygame.mixer.music.play(start=time_sec)
                self.update()

    def stream(self):
        while self.running and self.playing:
            if not self.paused:
                if not self.update():
                    break
                threading.Event().wait(self.delay / 1000.0)

    def update(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()

            if ret:
                try:
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                except tk.TclError:
                    return False  # Window has been closed
            else:
                self.playing = False
                pygame.mixer.music.stop()
        return True

    def on_close(self):
        """Handle window close event."""
        self.running = False  # Signal the thread to stop
        self.playing = False
        if self.vid.isOpened():
            self.vid.release()
        pygame.mixer.music.stop()
        pygame.quit()
        self.window.destroy()

# Create a window and pass it to the VideoPlayer class
root = tk.Tk()
VideoPlayer(root, "Tkinter Video Player", "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos/video1.mp4", "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Audios/video1.wav")
