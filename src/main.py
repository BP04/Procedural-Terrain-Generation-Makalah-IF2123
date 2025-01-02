# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# COLOR WEIGHT
WEIGHTS = [42, 38, 27, 18, 22, 20, 16, 24, 26]
WEIGHTS = np.array(WEIGHTS) / sum(WEIGHTS)
thresholds = np.cumsum(WEIGHTS)

COLORS = [
    [0, 0, 0.5],     # Deep ocean
    [0, 0, 0.75],    # Shallow ocean
    [0, 0, 1],       # Ocean
    [0.8, 0.7, 0.5], # Beach
    [0, 0.8, 0],     # Light grassland
    [0, 0.6, 0],     # Medium grassland
    [0, 0.4, 0],     # Dark grassland
    [0.5, 0.3, 0.1], # Mountain
    [0.3, 0.2, 0.1], # Higher moutain
]

# Permutation
permutation_size = 2**10

def shuffle(array_to_shuffle):
    np.random.shuffle(array_to_shuffle)

def make_permutation():
    P = np.arange(permutation_size, dtype=int)
    shuffle(P)
    return np.concatenate((P, P))

Perm = make_permutation()

# Perlin Noise and Map Calculation
def get_constant_vector(v):
    h = v & 3
    if h == 0:
        return np.array([1.0, 1.0])
    elif h == 1:
        return np.array([-1.0, 1.0])
    elif h == 2:
        return np.array([-1.0, -1.0])
    else:
        return np.array([1.0, -1.0])

def fade(t):
    return ((6 * t - 15) * t + 10) * t * t * t

def calculate_map(row1, col1, row2, col2, lacunarity):
    global Perm, permutation_size

    row_length = row2 - row1 + 1
    col_length = col2 - col1 + 1

    TR = np.zeros(row_length * col_length)
    TL = np.zeros(row_length * col_length)
    BR = np.zeros(row_length * col_length)
    BL = np.zeros(row_length * col_length)
    U1 = np.zeros(row_length * col_length)
    V1 = np.zeros(row_length * col_length)

    for i in range(row_length):
        for j in range(col_length):
            curx = (col1 + j) * lacunarity
            cury = (row1 + i) * lacunarity

            x_wrapped = int(np.floor(curx)) & (permutation_size - 1)
            y_wrapped = int(np.floor(cury)) & (permutation_size - 1)

            x_floor = curx - np.floor(curx)
            y_floor = cury - np.floor(cury)

            TRV = np.array([x_floor - 1.0, y_floor - 1.0])
            TLV = np.array([x_floor,       y_floor - 1.0])
            BRV = np.array([x_floor - 1.0, y_floor])
            BLV = np.array([x_floor,       y_floor])

            TRCV = get_constant_vector(Perm[Perm[x_wrapped + 1] + y_wrapped + 1])
            TLCV = get_constant_vector(Perm[Perm[x_wrapped]     + y_wrapped + 1])
            BRCV = get_constant_vector(Perm[Perm[x_wrapped + 1] + y_wrapped])
            BLCV = get_constant_vector(Perm[Perm[x_wrapped]     + y_wrapped])

            TR[i * col_length + j] = np.dot(TRV, TRCV)
            TL[i * col_length + j] = np.dot(TLV, TLCV)
            BR[i * col_length + j] = np.dot(BRV, BRCV)
            BL[i * col_length + j] = np.dot(BLV, BLCV)

            U1[i * col_length + j] = fade(x_floor)
            V1[i * col_length + j] = fade(y_floor)

    U2 = 1 - U1
    V2 = 1 - V1

    alpha = (U2 * ((V2 * BL) + (V1 * TL))) + (U1 * ((V2 * BR) + (V1 * TR)))

    result = np.zeros((row_length, col_length))

    for i in range(row_length):
        for j in range(col_length):
            result[i, j] = alpha[i * col_length + j]
    
    return result

global_minimum  = -0.85
global_maximum  = 1.15
first_iteration = True
TANH_K = 2.5

def apply_tanh_transform(value_array, k = TANH_K):
    # Shift [0,1] to [-1,1], apply k, then tanh, then shift back
    return 0.5 * (np.tanh(TANH_K * (2.0 * value_array - 1.0)) + 1.0)

def calculate_map_with_fbm(row1, col1, row2, col2, numOctaves):
    global global_minimum, global_maximum, first_iteration

    persistence = 0.9
    lacunarity  = 0.015

    result = np.zeros((row2 - row1 + 1, col2 - col1 + 1))

    for _ in range(numOctaves):
        result = result + persistence * calculate_map(row1, col1, row2, col2, lacunarity)

        persistence *= 0.5
        lacunarity  *= 2.0

    result = (result - global_minimum) / (global_maximum - global_minimum)

    result = apply_tanh_transform(result)

    return result

# binary search to find correct color
def upper_bound_threshold(height):
    left = 0
    right = len(thresholds) - 1
    pos = -1

    while left <= right:
        middle = (left + right) // 2
        if height < thresholds[middle]:
            pos = middle
            right = middle - 1
        else:
            left = middle + 1

    return pos

# Procedural Generation
upper_left_row = 0
upper_left_col = 0
ORIGINAL = 150
SIZE = 150
step = 10
OCTAVES = 7

MAP = [[] for _ in range(ORIGINAL)]

def init_map():
    global MAP, ORIGINAL, OCTAVES, COLORS

    fbm_map = calculate_map_with_fbm(0, 0, ORIGINAL - 1, ORIGINAL - 1, OCTAVES)

    for i in range(ORIGINAL):
        for j in range(ORIGINAL):
            k = upper_bound_threshold(fbm_map[i, j])
            MAP[i].append(COLORS[k])

def show_map():
    global MAP, ORIGINAL, upper_left_row, upper_left_col

    colored_map = []

    for i in range(upper_left_row, upper_left_row + ORIGINAL):
        current_row = []
        for j in range(upper_left_col, upper_left_col + ORIGINAL):
            current_row.append(MAP[i][j])
        colored_map.append(current_row)

    to_show_map = np.array(colored_map)

    plt.clf()
    plt.imshow(to_show_map, origin='upper')
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

def calculate_new_map():
    global MAP, SIZE, step

    fbm_map = calculate_map_with_fbm(SIZE, 0, SIZE + step - 1, SIZE - 1, OCTAVES)

    for i in range(SIZE, SIZE + step):
        MAP.append([])

        for j in range(SIZE):
            k = upper_bound_threshold(fbm_map[i - SIZE, j])
            MAP[i].append(COLORS[k])

    fbm_map = calculate_map_with_fbm(0, SIZE, SIZE + step - 1, SIZE + step - 1, OCTAVES)

    for i in range(SIZE + step):
        for j in range(step):
            k = upper_bound_threshold(fbm_map[i, j])
            MAP[i].append(COLORS[k])
    
    SIZE += step

def update_map(event):
    global upper_left_row, upper_left_col, ORIGINAL, step

    if event.key == 'up':
        if upper_left_row - step >= 0:
            upper_left_row -= step

    elif event.key == 'down':
        if upper_left_row + ORIGINAL + step >= SIZE:
            calculate_new_map()
        upper_left_row += step

    elif event.key == 'left':
        if upper_left_col - step >= 0:
            upper_left_col -= step

    elif event.key == 'right':
        if upper_left_col + ORIGINAL + step >= SIZE:
            calculate_new_map()
        upper_left_col += step

    show_map()

init_map()
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', update_map)
show_map()
plt.show()