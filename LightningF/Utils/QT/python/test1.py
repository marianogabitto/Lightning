#!/usr/bin/env python

import LightC
import numpy as np
import time


def load_points(filename):
    f = open(filename, 'r')
    N = int(f.readline())
    AP = np.loadtxt(f)
    return AP


def load_centers(filename, AP):
    f = open(filename, 'r')
    N = int(f.readline())
    AIX = np.loadtxt(f, dtype=np.int64)
    AC = AP[AIX,:]
    return AC

    
def test1():

    x = LightC.LightC()

    print("Loading points from file")
    t0 = time.time()
    AP = load_points("../../data/points.txt")
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))
    print("    Number of points: {}".format(AP.shape[0]))

    print("Adding points to C object")
    t0 = time.time()
    x.load_points(AP)
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))

    print("Loading centers from file")
    t0 = time.time()
    AC = load_centers("../../data/centers.txt", AP)
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))
    print("    Number of centers: {}".format(AC.shape[0]))

    print("Adding centers to C object")
    t0 = time.time()
    x.load_centers(AC)
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))

    print("Building tree of centers")
    t0 = time.time()
    x.build_tree_centers()
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))
    x.print_qtree_leaf_stats()
    
    print("Looping over points assigning them to centers")
    t0 = time.time()
    x.points_to_centers()
    print("    Elapsed: {:.3f} sec".format(time.time() - t0))

    point_ix, center_ix, logNnk, rnk = x.get_points_to_centers()
    print("    Total number of center-point assignments: {}".format(len(point_ix)))
    
    for ix in range(100):
        pix = point_ix[ix]
        cix = center_ix[ix]
        #print(AP[pix], AC[cix], AP[pix, 0:2] - AC[cix, 0:2]
        d = np.linalg.norm(AP[pix, 0:2] - AC[cix, 0:2])
        print("{:8d} {:8d} {:8.2f}  {:g}  {:g}".format(pix, cix, d, logNnk[ix], rnk[ix]))
        
    
test1()
