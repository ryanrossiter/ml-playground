from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

init_tiles = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

test_tiles = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

test2_tiles = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

class ConwaysGOL:
    def __init__(self, tiles):
        self.tiles = tiles
        self.h = len(tiles)
        self.w = len(tiles[0])

    def show(self):
        plt.imshow(self.tiles)
        plt.show()

    def _getTile(self, x, y):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return 0

        return self.tiles[y][x]

    def _countNeighbours(self, x, y):
        n = 0
        for xx in range(-1, 2):
            for yy in range(-1, 2):
                if xx == 0 and yy == 0:
                    continue
                n += self._getTile(x + xx, y + yy)

        return n

    def getTileFrame(self, x, y):
        frame = [[0 for i in range(5)] for ii in range(5)]
        for xx in range(5):
            for yy in range(5):
                frame[yy][xx] = self._getTile(x + xx - 2, y + yy - 2)

        return frame

    def step(self):
        next_tiles = [[0 for i in range(self.w)] for ii in range(self.h)]
        for y in range(self.h):
            for x in range(self.w):
                me = self._getTile(x, y)
                n = self._countNeighbours(x, y)
                v = 0
                if me:
                    if n == 2 or n == 3:
                        v = 1
                elif n == 3:
                    v = 1

                next_tiles[y][x] = v

        return ConwaysGOL(next_tiles)

def build_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(5, 5, 1)))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def prepare_data():
    frames = np.array([])
    states = np.array([])
    game = next_game = ConwaysGOL(init_tiles)

    steps = 10
    for i in range(steps):
        game = next_game
        next_game = game.step()

        states = np.append(states, next_game.tiles)
        for y in range(game.h):
            for x in range(game.w):
                frames = np.append(frames, game.getTileFrame(x, y))

    frames = frames.reshape(steps * game.w * game.h, 5, 5, 1)
    states = states.reshape(steps * game.w * game.h, 1)
    return (frames, states)

fig = plt.figure()
ims = []
def show_gif(frames):
    print(len(frames))
    for i in range(len(frames)):
        im = plt.imshow(frames[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    plt.show()

(x_train, y_train) = prepare_data()

# print(x_train.shape, y_train.shape)
model = build_model()
model.fit(x_train, y_train, epochs=25)

game = ConwaysGOL(init_tiles)
next_tiles = game.tiles
frames = []
for i in range(50):
    frames.append(next_tiles)

    next_tiles = [[0 for i in range(game.w)] for ii in range(game.h)]
    for y in range(game.h):
        for x in range(game.w):
            frame = np.array(game.getTileFrame(x, y)).reshape(1, 5, 5, 1)
            next_tiles[y][x] = model.predict(frame)[0][0]

    game.tiles = next_tiles

show_gif(frames)

# game = ConwaysGOL(init_tiles)
# frame = game.getTileFrame(5, 6)
# next_frame = game.step().getTileFrame(5, 6)
# print(frame)
# print(next_frame)
# print(model.predict(np.array(frame).reshape(1, 5, 5, 1)))

# print(y_train[0])
# print(model.predict(np.array(x_train[0]).reshape(1, 5, 5, 1)))

# game = ConwaysGOL(init_tiles)
# game.show()

# for i in range(10):
#     game = game.step()
#     game.show()
