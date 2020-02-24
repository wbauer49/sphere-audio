import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import scipy.io.wavfile as wav


display_graph = False
gravity = -0.01
default_k = 0.001
height = 10
width = 10


class Sphere:

    def __init__(self, pos, r=1, m=1, k=0.001):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros((3,), dtype=np.float32)
        self.acc = np.zeros((3,), dtype=np.float32)
        self.r = r
        self.m = m
        self.k = k

    def collide(self, other):
        disp = self.pos - other.pos
        overlap = self.r + other.r - la.norm(disp)
        if overlap > 0:
            self.acc += (disp / la.norm(disp)) * (overlap * self.k) / self.m

    def move(self):
        self.vel += self.acc
        self.pos += self.vel
        for i in range(3):
            if self.pos[i] + self.r > 10:
                self.vel[i] *= -1
                self.pos[i] = 2*(10-self.r) - self.pos[i]
            elif self.pos[i] - self.r < -10:
                self.vel[i] *= -1
                self.pos[i] = -2*(10-self.r) - self.pos[i]


def do_step(spheres):
    for s in spheres:
        s.acc = [0,-0.001,0]
    for s in spheres:
        for s2 in spheres:
            if s != s2:
                s.collide(s2)
    for s in spheres:
        s.move()

def make_plot(spheres):
    fig, ax = plt.subplots()
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    for s in spheres:
        circle = plt.Circle((s.pos[0], s.pos[1]), s.r, fill=False)
        ax.add_artist(circle)
    clear_output(wait=True)
    display(fig)
    plt.close()

def run_steps(spheres, num_steps, skip=1):
    audio = np.zeros((num_steps+1, 2), dtype=np.float32)
    for step in range(0, num_steps+1):
        for s in spheres:
            #audio[step] += 0.01*s.pos[1]
            audio[step,0] += 0.01*s.pos[1]*((10+s.pos[0])/20)
            audio[step,1] += 0.01*s.pos[1]*((10-s.pos[0])/20)
        if(step % skip == 0):
            if display_graph:
                make_plot(spheres)
            print(step)
        do_step(spheres)
    clear_output(wait=False)
    print(audio)
    wav.write("test1.wav", 44100, audio)


spheres = []
for x in range(3):
    for y in range(2):
        spheres.append(Sphere([x,y,0]))

run_steps(spheres, 1000000, skip=10000)
