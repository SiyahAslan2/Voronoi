import math
import random
import collections
import os
import sys
import functools
import operator as op
import numpy as np
import warnings

from scipy.spatial import cKDTree as KDTree
from skimage.filters.rank import entropy, otsu
from skimage.morphology import disk, dilation, erosion, diamond
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import sobel, gaussian, scharr
from skimage.restoration import denoise_tv_bregman
from skimage.transform import downscale_local_mean

"""def rand(x):
    r = x
    while r == x:
        r = random.uniform(0, x)
    return r"""


def uniform(a, b):
    return a + (b - a) * random.random()


def poisson_disc(img, n, k=30):
    h, w = img.shape[:2]

    nimg = denoise_tv_bregman(img, weight=1)

    def rgb2gray(rgb):
        return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])

    img_gray = rgb2gray(nimg)
    img_lab = rgb2lab(nimg)

    """entropy_weight = 2**(entropy(img_as_ubyte(img_gray), disk(15))) # da Entropy log zur Basis 2
    entropy_weight /= np.amax(entropy_weight) # max. des arrays
    entropy_weight = gaussian(dilation(entropy_weight, disk(15)), 5)"""

    """otsu_weight = otsu(img_as_ubyte(img_gray), disk(15))
    otsu_weight = gaussian(dilation(otsu_weight, disk(15)), 5)"""

    # color = [scharr(img_lab[:, :, i])**2 for i in range(1, 3)]
    color = []
    for i in range(1, 3):
        color.append(scharr(img_lab[:, :, i]) ** 2)
    edge_weight = functools.reduce(lambda x, y: x + y, color) ** (1 / 2) / 50
    edge_weight = dilation(edge_weight, disk(5))

    weight = edge_weight
    weight /= np.mean(weight)

    max_dist = avg_dist * 8  # min(h, w) / 4
    avg_dist = math.sqrt(w * h / (n * math.pi * 0.5) ** (1.05))
    min_dist = avg_dist / 4

    dists = np.clip(avg_dist / weight, min_dist, max_dist)

    # Generate first point randomly.
    first_point = (uniform(0, h), uniform(0, w))
    to_process = [first_point]
    sample_points = [first_point]
    tree = KDTree(sample_points)

    def gen_rand_point_around(point):
        radius = uniform(dists[int(point[0])][int(point[1])], max_dist)
        angle = uniform(0, 2 * math.pi)
        offset = np.array([radius * math.sin(angle), radius * math.cos(angle)])
        return tuple(point + offset)

    def has_neighbours(point):
        point_dist = dists[int(point[0])][int(point[1])]  # Output: int
        distances, idxs = tree.query(point,
                                     len(sample_points) + 1,
                                     distance_upper_bound=max_dist)

        if len(distances) == 0:
            return True

        for dist, idx in zip(distances, idxs):
            if np.isinf(dist):
                break

            if dist < point_dist and dist < dists[int((tuple(tree.data[idx]))[0])][int((tuple(tree.data[idx]))[1])]:
                return True

        return False

    while to_process:
        # Pop a random point.
        point = to_process.pop(random.randrange(len(to_process)))

        for _ in range(k):
            new_point = gen_rand_point_around(point)

            if (0 <= new_point[0] < h and 0 <= new_point[1] < w
                    and not has_neighbours(new_point)):
                to_process.append(new_point)
                sample_points.append(new_point)
                tree = KDTree(sample_points)
                if len(sample_points) % 1000 == 0:
                    print("Generated {} points.".format(len(sample_points)))

    print("Generated {} points.".format(len(sample_points)))

    return sample_points


def sample_colors(img, sample_points, n):
    h, w = img.shape[:2]

    print("Sampling colors...")
    tree = KDTree(np.array(sample_points))  # divide into voronoi segments
    color_samples = collections.defaultdict(list)
    img_lab = rgb2lab(img)
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]
    nearest = tree.query(pixel_coords)[1]

    i = 0
    for pixel_coord in pixel_coords:
        color_samples[tuple(tree.data[nearest[i]])].append(
            img_lab[tuple(pixel_coord)])
        i += 1

    print("Computing color means...")
    samples = []
    for point, colors in color_samples.items():
        avg_color = np.sum(colors, axis=0) / len(colors)
        samples.append(np.append(point, avg_color))

    if len(samples) > n:
        print("Downsampling {} to {} points...".format(len(samples), n))

    while len(samples) > n:
        tree = KDTree(np.array(samples))
        dists, neighbours = tree.query(np.array(samples), 2)
        dists = dists[:, 1]
        worst_idx = min(range(len(samples)), key=lambda i: dists[i])
        samples[neighbours[worst_idx][1]] += samples[neighbours[worst_idx][0]]
        samples[neighbours[worst_idx][1]] /= 2
        samples.pop(neighbours[worst_idx][0])

    color_samples = []
    for sample in samples:
        color = lab2rgb([[sample[2:]]])[0][0]
        color_samples.append(tuple(sample[:2][::-1]) + tuple(color))

    return color_samples


def render(img, color_samples):
    print("Rendering...")
    h, w = [2 * x for x in img.shape[:2]]
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    pixel_coords = np.c_[xx.ravel(), yy.ravel()]

    colors = np.empty([h, w, 3])
    coords = []
    for color_sample in color_samples:
        coord = tuple(x * 2 for x in color_sample[:2][::-1])
        colors[int(coord[0])][int(coord[1])] = color_sample[2:]
        coords.append(coord)

    tree = KDTree(coords)
    idxs = tree.query(pixel_coords)[1]
    data = colors[tuple(tree.data[idxs].astype(int).T)].reshape((w, h, 3))
    data = np.transpose(data, (1, 0, 2))

    return downscale_local_mean(data, (2, 2, 1))


if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    img = imread("autorenders/Bielefeld.jpg")
    print("Calibrating")
    mult = 1.02 * 500 / len(poisson_disc(img, 500))

# n = int(input("Geben Sie ein n an:"))
n = 1000
sample_points = poisson_disc(img, mult * n)
samples = sample_colors(img, sample_points, n)
base = os.path.basename("Bielefeld.jpg")
"""with open("{}-{}.txt".format(os.path.splitext(base)[0], n), "w") as f:
    for sample in samples:
        f.write(" ".join("{:.3f}".format(x) for x in sample) + "\n")"""
imsave("autorenders/Voronoi/{}-{}.png".format(os.path.splitext(base)[0], n),
       render(img, samples))

print("Done!")
