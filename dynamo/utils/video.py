import imageio
import os


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs):
        if self.enabled:
            self.frames.append(obs)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
