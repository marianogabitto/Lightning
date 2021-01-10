# coding=utf-8
import os
import csv
import copy
import pickle
import numpy as np
from scipy import io as sio

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# #################################### IMPORT DATASETS FROM FILES ####################################
def data_import(dataset=None, pl=0):

    # ######################################################################
    # ######################################################################
    # This dataset is composed of the three fluorophores attached to the
    # DNA Origami Scaffold.
    if dataset == 'Origami-AF647':
        file = 'mat'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if file == 'pkl':
            new = list()
            with open(os.path.join(dir_path, "DNA-Origami-DNA-AF647-Lakadamyali.pkl"), "rb") as f:
                new.append(pickle.load(f))
                new.append(pickle.load(f))
                new.append(pickle.load(f))
                new.append(pickle.load(f))

        elif file == 'mat':
            data = sio.loadmat(os.path.join(dir_path, "DNA-Origami-DNA-AF647-Lakadamyali.mat"))

            # Observations of the three fluorophores
            new = list()
            new.append(np.zeros((data['obs'].shape[0], 5)))
            new[0][:, :3] = data['obs'][:, 0:3]
            new[0][:, 3:5] = data['obs'][:, 8][:, None] ** 2 / 10

            # Observations Fluorophore 1
            new.append(np.zeros((data['fluo1'].shape[0], 5)))
            new[1][:, :3] = data['fluo1'][:, 0:3]
            new[1][:, 3:5] = data['fluo1'][:, 8][:, None] ** 2 / 10

            # Observations Fluorophore 2
            new.append(np.zeros((data['fluo2'].shape[0], 5)))
            new[2][:, :3] = data['fluo2'][:, 0:3]
            new[2][:, 3:5] = data['fluo2'][:, 8][:, None] ** 2 / 10

            # Observations Fluorophore 3
            new.append(np.zeros((data['fluo3'].shape[0], 5)))
            new[3][:, :3] = data['fluo3'][:, 0:3]
            new[3][:, 3:5] = data['fluo3'][:, 8][:, None] ** 2 / 10

        # Plot if value passed
        if pl == 1:
            fig, ax = plt.subplots()
            ax.plot(new[0][:, 1], new[0][:, 2], 'r.', ms=0.5)
            for n_ in np.arange(new[0].shape[0]):
                ax.add_artist(plt.Circle((new[0][n_, 1], new[0][n_, 2]), np.sqrt(new[0][n_, 3]), fill=False))
            ax.axis([-150, 150, -250, 100])
            plt.show()

    # This dataset is composed of the Dynein Motors attached to the
    # DNA Origami Scaffold.
    elif dataset == 'Origami-Dynein-AF647':
        data = sio.loadmat('Datasets/DNA-Origami-Dynein-AB-AF647-Lakadamyali.mat')

        # Observations of the three fluorophores
        new = list()
        new.append(np.zeros((data['obs'].shape[0], 5)))
        new[0][:, :3] = data['obs'][:, 1:4]
        new[0][:, 3:5] = data['obs'][:, 9][:, None] ** 2

        # Observations Fluorophore 1
        new.append(np.zeros((data['obs_fluo1'].shape[0], 5)))
        new[1][:, :3] = data['obs_fluo1'][:, 1:4]
        new[1][:, 3:5] = data['obs_fluo1'][:, 9][:, None] ** 2

        # Observations Fluorophore 2
        new.append(np.zeros((data['obs_fluo2'].shape[0], 5)))
        new[2][:, :3] = data['obs_fluo2'][:, 1:4]
        new[2][:, 3:5] = data['obs_fluo2'][:, 9][:, None] ** 2

        # Observations Fluorophore 3
        new.append(np.zeros((data['obs_fluo3'].shape[0], 5)))
        new[3][:, :3] = data['obs_fluo3'][:, 1:4]
        new[3][:, 3:5] = data['obs_fluo3'][:, 9][:, None] ** 2

        # Plot if value passed
        if pl == 1:
            fig, ax = plt.subplots()
            ax.plot(new[0][:, 1], new[0][:, 2], 'r.', ms=0.5)
            ax.plot(new[1][:, 1], new[1][:, 2], 'b+', ms=2)
            ax.plot(new[2][:, 1], new[2][:, 2], 'c+', ms=2)
            ax.plot(new[3][:, 1], new[3][:, 2], 'y+', ms=2)
            for n_ in np.arange(new[0].shape[0]):
                ax.add_artist(plt.Circle((new[0][n_, 1], new[0][n_, 2]), np.sqrt(new[0][n_, 3]), fill=False))
            # ax.axis([-150, 150, -250, 100])
        plt.show()

    # This dataset is composed of a list of isolated Nuclear Pore Complexes
    # in which NUP-107-SNAP-TAG is labelled with AF-647.
    elif dataset == 'NPC-Nup107-Reis':
        f = open('Datasets/npc-Nup107-AF647-Reis.pkl', 'rb')
        new = list()
        try:
            new.append(pickle.load(f))  # Cropped Images
            new.append(pickle.load(f))  # Localizations in every Images
            new.append(pickle.load(f))
            new.append(pickle.load(f))
        except:
            f = open('Datasets/npc-Nup107-AF647-Reis.pkl', 'rb')
            new = list()
            new.append(pickle.load(f, encoding='latin1'))  # Cropped Images
            new.append(pickle.load(f, encoding='latin1'))  # Localizations in every Images
            new.append(pickle.load(f, encoding='latin1'))
            new.append(pickle.load(f, encoding='latin1'))

        # Plot if value passed
        if pl == 1:
            fig, ax = plt.subplots()
            ax.imshow(new[0][0])
            plt.show()

    # This dataset is composed of a list of isolated Nuclear Pore Complexes
    # in which NUP-210 is immuno-stainned and labelled with AF-647.
    # Imaged at the Reichter Lab
    elif dataset == 'NPC-Nup210-Reichter':
        f = open('Datasets/npc-Nup210-AB-AF647-Reichter.pkl', 'rb')
        new = list()
        new.append(pickle.load(f))  # Cropped Images
        new.append(pickle.load(f))  # Localizations in every Images
        new.append(pickle.load(f))
        new.append(pickle.load(f))

        # Plot if value passed
        if pl == 1:
            fig, ax = plt.subplots()
            ax.imshow(new[0][0])
            plt.show()

    # This dataset is composed of a list of isolated Nuclear Pore Complexes
    # in which NUP-210 is immuno-stainned and labelled with AF-647.
    # Imaged at the Sandrine Lab
    elif dataset == 'NPC-Nup210-Sandrine':
        dirpath = os.getcwd()
        filename = dirpath + '/Datasets/npc-Nup210-AB-AF647-Sandrine.pkl'
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
        new = list()
        for l_ in np.arange(len(data)):
            temp_dat = data[l_]
            temp_dat[:, 3:] = temp_dat[:, 3:] ** 2
            new.append(temp_dat)

    # ######################################################################
    # ######################################################################

    else:
        new = np.array([])
        print('Incorrect Selection. possible options are:')
        print(' Origami-AF647, Origami-Dynein-AF647, NPC-Nup107-Reis, NPC-Nup210-Reichter, NPC-Nup210-Sandrine')
        quit()

    return new


# ####################################     CREATE FLUOROPHORES    ####################################
def create_twofluo(fluos, time=0, dist=0, noise=0, pl=0, plcircles=0, seed=-1):

    if seed > -1:
        np.random.seed(seed)

    mean = list()
    mean.append([])
    mean_model_position1 = np.sum(fluos[1][:, 1] * fluos[1][:, 3]**-1, axis=0) * (np.sum(fluos[1][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[1][:, 2] * fluos[1][:, 4]**-1, axis=0) * (np.sum(fluos[1][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[1][:, 1], axis=0)
    m2 = np.mean(fluos[1][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])
    mean_model_position1 = np.sum(fluos[2][:, 1] * fluos[2][:, 3]**-1, axis=0) * (np.sum(fluos[2][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[2][:, 2] * fluos[2][:, 4]**-1, axis=0) * (np.sum(fluos[2][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[2][:, 1], axis=0)
    m2 = np.mean(fluos[2][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])
    mean_model_position1 = np.sum(fluos[3][:, 1] * fluos[3][:, 3]**-1, axis=0) * (np.sum(fluos[3][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[3][:, 2] * fluos[3][:, 4]**-1, axis=0) * (np.sum(fluos[3][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[3][:, 1], axis=0)
    m2 = np.mean(fluos[3][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])

    # Select Fluos
    idx1 = np.random.choice([1, 2, 3])
    idx2 = idx1
    while idx2 == idx1:
        idx2 = np.random.choice([1, 2, 3])

    # Create Fluos
    fluo1 = fluos[idx1]
    fluo1[:, 1:3] = fluo1[:, 1:3] - mean[idx1]
    fluo1[:, 0] = fluo1[:, 0] - fluo1[0, 0]

    fluo2 = fluos[idx2]
    fluo2[:, 1:3] = fluo2[:, 1:3] + [dist, 0] - mean[idx2]
    fluo2[:, 0] = fluo2[:, 0] - fluo2[0, 0] + time

    out_data = np.concatenate([fluo1, fluo2])
    out_idx = np.zeros((fluo1.shape[0] + fluo2.shape[0], 3))
    out_idx[0:fluo1.shape[0], 1] = 1
    out_idx[fluo1.shape[0]:, 2] = 1

    # Corrupt Image with Noise
    if noise > 0:
        # Number of Points Parameters
        n_total = np.int(out_data.shape[0])
        n_noise = np.int(np.round(n_total * noise))

        # Rounding box parameters
        l_x = np.max(out_data[:, 1]) - np.min(out_data[:, 1])
        l_y = np.max(out_data[:, 2]) - np.min(out_data[:, 2])
        xmi = np.min(out_data[:, 1]) - 0.5 * l_x
        xma = np.max(out_data[:, 1]) + 0.5 * l_x
        ymi = np.min(out_data[:, 2]) - 0.5 * l_y
        yma = np.max(out_data[:, 2]) + 0.5 * l_y

        # Add noise
        noise_point = copy.copy(out_data[np.random.choice(n_total, n_noise), :])
        noise_point[:, 1:3] = np.array([np.random.uniform(xmi, xma, n_noise), np.random.uniform(ymi, yma, n_noise)]).T

        out_data = np.concatenate([out_data, noise_point])
        out_idx = np.concatenate([out_idx, np.array([[1, 0, 0] for _ in np.arange(noise_point.shape[0])])])

    if pl == 1:
        fig, ax = plt.subplots()
        ax.plot(out_data[:, 1], out_data[:, 2], '+', ms=0.5)
        if plcircles == 1:
            for n_ in np.arange(out_data.shape[0]):
                ax.add_artist(plt.Circle((out_data[n_, 1], out_data[n_, 2]), np.sqrt(out_data[n_, 3]), fill=False))
        plt.show()
        ax.axis([-100, dist + 100, -100, 100])

    return out_data, out_idx


def create_nfluo(fluos, dist=50, n=1, noise=0, pl=0, plcircles=0, seed=-1):

    if seed > -1:
        np.random.seed(seed)

    mean = list()
    mean.append([])
    mean_model_position1 = np.sum(fluos[1][:, 1] * fluos[1][:, 3]**-1, axis=0) * (np.sum(fluos[1][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[1][:, 2] * fluos[1][:, 4]**-1, axis=0) * (np.sum(fluos[1][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[1][:, 1], axis=0)
    m2 = np.mean(fluos[1][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])
    mean_model_position1 = np.sum(fluos[2][:, 1] * fluos[2][:, 3]**-1, axis=0) * (np.sum(fluos[2][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[2][:, 2] * fluos[2][:, 4]**-1, axis=0) * (np.sum(fluos[2][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[2][:, 1], axis=0)
    m2 = np.mean(fluos[2][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])
    mean_model_position1 = np.sum(fluos[3][:, 1] * fluos[3][:, 3]**-1, axis=0) * (np.sum(fluos[3][:, 3]**-1, axis=0)**-1)
    mean_model_position2 = np.sum(fluos[3][:, 2] * fluos[3][:, 4]**-1, axis=0) * (np.sum(fluos[3][:, 4]**-1, axis=0)**-1)
    m1 = np.mean(fluos[3][:, 1], axis=0)
    m2 = np.mean(fluos[3][:, 2], axis=0)
    # mean.append([m1, m2])
    mean.append([mean_model_position1, mean_model_position2])

    # Create Fluos In the grid
    out_data = []
    out_idx = []
    out_loc = np.zeros((n*n, 2))
    for i_ in np.arange(n):
        for j_ in np.arange(n):
            idx = np.random.choice([1, 2, 3])
            fluo2 = copy.deepcopy(fluos[idx])
            fluo2[:, 1:3] = fluo2[:, 1:3] - mean[idx] + [i_ * dist, j_ * dist]
            fluo2[:, 0] = fluo2[:, 0] - fluo2[0, 0]
            out_loc[i_*n + j_, :] = [i_ * dist, j_ * dist]

            if (i_ == 0) and (j_ == 0):
                out_data = fluo2
                out_idx = np.zeros((len(fluo2), 2))
                out_idx[:, 1] = 1
            else:
                out_data = np.concatenate([out_data, fluo2])
                nf = len(out_idx)
                out_idx = np.concatenate([out_idx, np.zeros((nf, 1))], axis=1)
                out_idx = np.concatenate([out_idx, np.zeros((len(fluo2), out_idx.shape[1]))], axis=0)
                out_idx[-len(fluo2):, -1] = 1

    # Corrupt Image with Noise
    if noise > 0:
        # Number of Points Parameters
        n_total = np.int(out_data.shape[0])
        n_noise = np.int(np.round(n_total * noise))

        # Rounding box parameters
        xmi = np.min(out_data[:, 1]) - dist / 2.
        xma = np.max(out_data[:, 1]) + dist / 2.
        ymi = np.min(out_data[:, 2]) - dist / 2.
        yma = np.max(out_data[:, 2]) + dist / 2.

        # Add noise
        noise_point = copy.copy(out_data[np.random.choice(n_total, n_noise), :])
        noise_point[:, 1:3] = np.array([np.random.uniform(xmi, xma, n_noise), np.random.uniform(ymi, yma, n_noise)]).T

        out_data = np.concatenate([out_data, noise_point])

        noise_vector = np.zeros((noise_point.shape[0], out_idx.shape[1]))
        noise_vector[:, 0] = 1
        out_idx = np.concatenate([out_idx, noise_vector], axis=0)

    if pl == 1:
        # colormap = np.array(['r', 'g', 'b', 'o', 'c', 'm'])
        a = np.mod(np.argmax(out_idx, axis=1), 4)
        fig, ax = plt.subplots()
        ax.scatter(out_data[:, 1], out_data[:, 2], c=a, marker="+")

        if plcircles == 1:
            for n_ in np.arange(out_data.shape[0]):
                ax.add_artist(plt.Circle((out_data[n_, 1], out_data[n_, 2]), np.sqrt(out_data[n_, 3]), fill=False))
        plt.show()
        # ax.axis([-100, dist + 100, -100, 100])

    return out_data, out_idx, out_loc


# ####################################     EVALUATE PERFORMANCE   ####################################
def metric_distance(mus, true_position):

    distance_centers = np.zeros(len(mus))
    for mi_, m_ in enumerate(mus):
        distance_centers[mi_] = np.min(np.linalg.norm(m_ - true_position, axis=1), axis=0)

    return np.mean(distance_centers), np.std(distance_centers)


# #################################### IMPORT DATASETS FROM FILES ####################################
def csv_import(filename, delimiter=',', columns=[0]):
    output = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            output.append([row[x] for x in columns])

    return np.array(output)
