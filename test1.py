import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


size = 10
num_spheres = 4
num_steps = 1000000
num_skip = 100000
default_g = 0.01
default_k = 0.001
sample_rate = 44100

display_graph = False


class Sphere:

    def __init__(self, pos, r=1, m=1, g=default_g, k=default_k):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros((3,), dtype=np.float32)
        self.acc = np.zeros((3,), dtype=np.float32)
        self.r = r
        self.m = m
        self.g = g
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
            if self.pos[i] + self.r > size:
                self.vel[i] *= -1
                self.pos[i] = 2 * (size - self.r) - self.pos[i]
            elif self.pos[i] - self.r < -size:
                self.vel[i] *= -1
                self.pos[i] = -2 * (size - self.r) - self.pos[i]


def do_step(spheres):
    for s in spheres:
        s.acc = [0, -s.g, 0]
    for s in spheres:
        for s2 in spheres:
            if s != s2:
                s.collide(s2)
    for s in spheres:
        s.move()

def make_plot(spheres):
    fig, ax = plt.subplots()
    ax.set_xlim((-size, size))
    ax.set_ylim((-size, size))
    for s in spheres:
        circle = plt.Circle((s.pos[0], s.pos[1]), s.r, fill=False)
        ax.add_artist(circle)
    plt.show()
    plt.close()

def run_steps(spheres, num_steps, skip=num_skip):
    audio = np.zeros((num_steps+1, 2), dtype=np.float32)
    for step in range(0, num_steps+1):
        for s in spheres:
            dists = np.array([size + s.pos[0], size - s.pos[0]])
            audio[step,:] += (0.01 * s.pos[1] * dists) / (2*size)
        if(step % skip == 0):
            if display_graph:
                make_plot(spheres)
            print(step)
        do_step(spheres)
    print(audio)
    filename = "wavs/test1_s{0}_n{1}_g{2}_k{3}.wav".format(size, num_spheres, '{:.0e}'.format(default_g), '{:.0e}'.format(default_k))
    wav.write(filename, 44100, audio)
    print("saved to", filename)

def run_simulation():
    spheres = []
    for i in range(num_spheres):
        spheres.append(Sphere([i, i, 0]))
    run_steps(spheres, num_steps)


for kp in range(-9, 0):
    default_k = 10 ** (kp)
    run_simulation()
