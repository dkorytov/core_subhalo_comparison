#!/usr/bin/env python2.7

from __future__ import print_function, division 
import numpy as np
import matplotlib
import os
#checks if there is a display to use.
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import sys
import time
import numpy.random
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree


from matplotlib import rc
rc('text', usetex=True)
rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], })
rc('font', size=18)

def load_gio_into_dict(fname, dic, var):
    dic[var] = dtk.gio_read(fname, var)

def load_core_catalog(fname):
    result = {}
    dtk.gio_inspect(fname)
    load_gio_into_dict(fname, result, 'x')
    load_gio_into_dict(fname, result, 'y')
    load_gio_into_dict(fname, result, 'z')
    load_gio_into_dict(fname, result, 'vx')
    load_gio_into_dict(fname, result, 'vy')
    load_gio_into_dict(fname, result, 'vz')   
    load_gio_into_dict(fname, result, 'infall_mass')
    load_gio_into_dict(fname, result, 'infall_step')
    load_gio_into_dict(fname, result, 'radius')
    load_gio_into_dict(fname, result, 'fof_halo_tag')
    result['central'] = result['infall_step']==499
    srt = np.argsort(result['x'])
    dtk.reorder_dict(result, srt)
    return result

def load_subhalo_catalog(fname):
    result = {}
    dtk.gio_inspect(fname)
    load_gio_into_dict(fname, result, 'subhalo_mean_x')
    load_gio_into_dict(fname, result, 'subhalo_mean_y')
    load_gio_into_dict(fname, result, 'subhalo_mean_z')
    load_gio_into_dict(fname, result, 'subhalo_mean_vx')
    load_gio_into_dict(fname, result, 'subhalo_mean_vy')
    load_gio_into_dict(fname, result, 'subhalo_mean_vz')
    load_gio_into_dict(fname, result, 'subhalo_mass')
    load_gio_into_dict(fname, result, 'subhalo_count')
    load_gio_into_dict(fname, result, 'subhalo_tag')
    load_gio_into_dict(fname, result, 'subhalo_vel_disp')
    load_gio_into_dict(fname, result, 'fof_halo_count')
    load_gio_into_dict(fname, result, 'fof_halo_tag')
    srt = np.argsort(result['subhalo_mean_x'])
    dtk.reorder_dict(result, srt)
    return result

def load_bighalo_particle_catalog(fname):
    result = {}
    dtk.gio_inspect(fname)
    print("loading big halo particles")
    eta = dtk.ETA()
    load_gio_into_dict(fname, result, 'x')
    eta.print_progress(1,3)
    load_gio_into_dict(fname, result, 'y')
    eta.print_progress(2,3)
    load_gio_into_dict(fname, result, 'z')
    eta.print_done()
    return result

def load_fof_halo_catalog(fname):
    result = {}
    dtk.gio_inspect(fname)
    load_gio_into_dict(fname, result, 'fof_halo_center_x')
    load_gio_into_dict(fname, result, 'fof_halo_center_y')
    load_gio_into_dict(fname, result, 'fof_halo_center_z')
    load_gio_into_dict(fname, result, 'fof_halo_mean_vx')
    load_gio_into_dict(fname, result, 'fof_halo_mean_vy')
    load_gio_into_dict(fname, result, 'fof_halo_mean_vz')
    load_gio_into_dict(fname, result, 'fof_halo_tag')
    srt = np.argsort(result['fof_halo_tag'])
    dtk.reorder_dict(result, srt)
    return result

def load_accum_particle_catalog(fname):
    result = {}
    dtk.gio_inspect(fname)
    print("Loading accum particles")
    eta = dtk.ETA()
    load_gio_into_dict(fname, result, 'x')
    eta.print_progress(1, 3)
    load_gio_into_dict(fname, result, 'y')
    eta.print_progress(2, 3)
    load_gio_into_dict(fname, result, 'z')
    eta.print_done()
    return result

def combine_dict(d1, d2):
    result = {}
    for k in d1.keys():
        result[k] = d1[k]
    for k in d2.keys():
        result[k] = d2[k]
    return result

def peroid_distance(x1, x2, rl):
    dx = np.abs(x1-x2)
    # if np.isscalar(dx):
    #     if dx>rl/2:
    #         dx = rl-dx
    # else:
    #     slct = dx>rl/2
    #     dx[slct] = rl-dx[slct]
    return dx
        
def velocity_magnitude(x1, y1, z1):
    return np.sqrt(x1**2 + y1**2 + z1**2)

def angle_between_vectors(x1, y1, z1, x2, y2, z2):
    mag1 = velocity_magnitude(x1, y1, z1)
    mag2 = velocity_magnitude(x2, y2, z2)
    dot = x1*x2 + y1*y2 + z1*z2
    return dot/(mag1*mag2)

def distance_between_core_and_subhalo(cores, core_index, subhalos, subhalo_index, rl):
    dx = peroid_distance(cores['x'][core_index], subhalos['subhalo_mean_x'][subhalo_index], rl)
    dy = peroid_distance(cores['y'][core_index], subhalos['subhalo_mean_y'][subhalo_index], rl)
    dz = peroid_distance(cores['z'][core_index], subhalos['subhalo_mean_z'][subhalo_index], rl)
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def distance_between_cores_sq(cores, core_indexes1, core_indexes2, rl):
    dx = peroid_distance(cores['x'][core_indexes1], cores['x'][core_indexes2], rl)
    dy = peroid_distance(cores['y'][core_indexes1], cores['y'][core_indexes2], rl)
    dz = peroid_distance(cores['z'][core_indexes1], cores['z'][core_indexes2], rl)
    # return np.sqrt(dx*dx + dy*dy + dz*dz)
    return dx*dx + dy*dy + dz*dz
    
def find_closest(cores, subhalos, r_max, rl):
    subhalo_cat_length = len(subhalos['subhalo_mean_x'])
    cores_size = len(cores['x'])
    subhalos['core_index'] = -1*np.ones(subhalo_cat_length, dtype=np.int64)
    subhalos['core_mass'] = -1*np.ones(subhalo_cat_length, dtype=np.int64)
    t0= dtk.AutoTimer()
    simple_searcher = dtk.SimpleOneDimensionalSearcher(subhalos['subhalo_mean_x'])
    print(t0)
    t0.reset()
    close_subhalos_to_cores = simple_searcher.query_unsorted_nearby(cores['x'], r_max)
    eta = dtk.ETA()
    for core_index, close_subhalos_to_core in enumerate(close_subhalos_to_cores):
        if core_index%1000 == 1:
            eta.print_progress(core_index, cores_size)
        # For each core, find the closest subhalo

        closest_subhalo_index = -1
        closest_subhalo_distance = np.inf
        for close_subhalo_index in close_subhalos_to_core:
            dr = distance_between_core_and_subhalo(cores, core_index, subhalos, close_subhalo_index, rl)
            if dr<closest_subhalo_distance and dr<r_max:
                closest_subhalo_distance = dr
                closest_subhalo_index = close_subhalo_index
        # If the closest subhalo has been assigned a core, check if
        # that core is less massive. If so, assign the this core to the subhalo
        assign_core_to_subhalo = False
        if subhalos['core_index'][closest_subhalo_index] ==-1:
            assign_core_to_subhalo = True
        else:
            if subhalos['core_mass'][closest_subhalo_index] < cores['infall_mass'][core_index]:
                assign_core_to_subhalo = True
        if assign_core_to_subhalo:
            subhalos['core_index'][closest_subhalo_index] = core_index
            subhalos['core_mass'][closest_subhalo_index] = cores['infall_mass'][core_index]
    print(t0)
    

def find_closest_ckdtree(cores, subhalos, r_max, rl, match_cores_to_subhalos=True):
    subhalo_cat_length = len(subhalos['subhalo_mean_x'])
    cores_size = len(cores['x'])
    subhalos['core_index'] = -1*np.ones(subhalo_cat_length, dtype=np.int64)
    subhalos['core_mass'] = -1*np.ones(subhalo_cat_length, dtype=np.int64)
    cores['subhalo_index'] = -1*np.ones(cores_size, dtype=np.int64)
    t0= dtk.AutoTimer()
    subhalo_xyz_mat = np.stack((subhalos['subhalo_mean_x'], subhalos['subhalo_mean_y'], subhalos['subhalo_mean_z']), axis=1)
    core_xyz_mat = np.stack((cores['x'], cores['y'], cores['z']), axis=1)
    if match_cores_to_subhalos:
        ckdtree = cKDTree(subhalo_xyz_mat, balanced_tree = False, compact_nodes = False)
        print("cKDtree construction:", t0)
        t0.reset()
        subhalo_indexes_per_core = ckdtree.query_ball_point(core_xyz_mat, r_max, n_jobs=10)
        print("query time:", t0)
        eta=dtk.ETA()
        for core_index, subhalo_indexes in enumerate(subhalo_indexes_per_core):
            if core_index %1000 == 1:
                eta.print_progress(core_index, cores_size)
            for subhalo_index in subhalo_indexes:
                assign_core_to_subhalo = False
                if subhalos['core_index'][subhalo_index] ==-1:
                    assign_core_to_subhalo = True
                else:
                    if subhalos['core_mass'][subhalo_index] < cores['infall_mass'][core_index]:
                        assign_core_to_subhalo = True
                        # un-pair the currently paired core from the subhalo since we found a heavier
                        # core for the subhalo
                        cores['subhalo_index'][subhalos['core_index'][subhalo_index]] = -1 
                if assign_core_to_subhalo:
                    subhalos['core_index'][subhalo_index] = core_index
                    subhalos['core_mass'][subhalo_index]  = cores['infall_mass'][core_index]
                    cores['subhalo_index'][core_index] = subhalo_index
    else: #match subhalos to cores
        ckdtree = cKDTree(core_xyz_mat, balanced_tree=False, compact_nodes=False)
        print("cKDtree construction:", t0)
        t0.reset()
        distances, core_index =ckdtree.query(subhalo_xyz_mat, 1, distance_upper_bound=0.1)
        slct_found_element = np.isfinite(distances)
        plt.figure()

        h1, xbins = np.histogram(distances[slct_found_element], bins=np.logspace(-3, 0, 100))

        mass_cut = 3e11
        slct_mass = subhalos['subhalo_mass']>mass_cut
        h2, xbins = np.histogram(distances[slct_found_element & slct_mass], bins=np.logspace(-3, 0, 100))
        plt.plot(dtk.bins_avg(xbins), h1, label='All subhalos')
        plt.plot(dtk.bins_avg(xbins), h2, label='>{:.2e}h $^{{-1}}$ Msun'.format(mass_cut))
        plt.xscale('log')
        plt.legend(loc='best')
        plt.xlabel('Distance from Subhalo to Core')
        plt.ylabel('Count')
        dtk.save_figs("figs/"+__file__+"/")
        plt.close()

        plt.figure()
        plt.plot(dtk.bins_avg(xbins), h1, label='All subhalos')
        plt.plot(dtk.bins_avg(xbins), h2, label='>{:.2e}h $^{{-1}}$ Msun'.format(mass_cut))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel('Distance from Subhalo to Core')
        plt.ylabel('Count')
        dtk.save_figs("figs/"+__file__+"/")
        plt.close()

        print(np.max(core_index), np.min(core_index))
        print(np.min(distances), np.max(distances), np.sum(~np.isfinite(distances)))
        print(distances[~np.isfinite(distances)])
        subhalos['core_index'] = core_index

        subhalos['core_mass'][slct_found_element] = cores['infall_mass'][core_index[slct_found_element]]
        subhalos['core_index'][~slct_found_element] = -1
        subhalos['distance_to_core'] = distances
        subhalo_indexes = np.arange(len(subhalos['core_index']), dtype=int)
        cores['subhalo_index'] = -1*np.ones(len(cores['x']), dtype=int)
        cores['subhalo_mass'] = -1*np.ones(len(cores['x']), dtype=float)
        cores['subhalo_index'][subhalos['core_index'][slct_found_element]] = subhalo_indexes[slct_found_element]
        cores['subhalo_mass'][subhalos['core_index'][slct_found_element]] = subhalos['subhalo_mass'][slct_found_element]

def fof_merge_colors(colors, i, j):
    c1 = colors[i]
    c2 = colors[j]
    if c1 == c2:
        return
    if c1 < c2:
        colors[colors == c2] = c1
    else:
        colors[colors == c1] = c2

def link_cores_fof(cores, link_length):
    link_length_sq = link_length**2
    cores_size = len(cores['x'])
    fof_colors = np.arange(0, cores_size)
    t0=dtk.AutoTimer()
    simple_searcher = dtk.SimpleOneDimensionalSearcher(cores['x'])
    print("Init Simple Searcher:", t0)
    t0.reset()
    core_indexes_close_to_cores = simple_searcher.query_unsorted_nearby(cores['x'], link_length)
    print("Search Simple Searcher:", t0)
    eta = dtk.ETA()
    for core_index, core_indexes_close_to_core in enumerate(core_indexes_close_to_cores):
        if core_index % 1000 ==1:
            eta.print_progress(core_index, cores_size)
        for core_index_close_to_core in core_indexes_close_to_core:
            dr_sq = distance_between_cores_sq(cores, core_index, core_index_close_to_core, 256.0)
            if dr_sq<link_length_sq:
                fof_merge_colors(fof_colors, core_index, core_index_close_to_core)
    return fof_colors

def link_cores_fof_ckdtree(cores, link_length):
    cores_size = len(cores['x'])
    fof_colors = np.arange(0, cores_size)
    core_xyz_mat = np.stack((cores['x'], cores['y'], cores['z']), axis=1)
    t0 = dtk.AutoTimer()    
    ckdtree = cKDTree(core_xyz_mat, balanced_tree = False, compact_nodes = False)
    print("Tree Construction time: ", t0)
    t0.reset()    
    neighbor_indexes_per_core = ckdtree.query_ball_point(core_xyz_mat, link_length, n_jobs=10)
    print("Query time:", t0)
    eta = dtk.ETA()
    for core_index, neighbor_indexes in enumerate(neighbor_indexes_per_core):
        if core_index%1000 ==1:
            eta.print_progress(core_index, cores_size)
        for neighbor_index in neighbor_indexes:
            fof_merge_colors(fof_colors, core_index, neighbor_index)
    return fof_colors

def init_fof_merged_cores(size):
    result = {}
    result['x'] = np.zeros(size)
    result['y'] = np.zeros(size)
    result['z'] = np.zeros(size)
    result['vx'] = np.zeros(size)
    result['vy'] = np.zeros(size)
    result['vz'] = np.zeros(size)
    result['infall_mass'] = np.zeros(size)
    result['infall_mass_avg'] = np.zeros(size)
    result['infall_step'] = np.zeros(size)
    result['radius'] = np.zeros(size)
    result['central'] = np.zeros(size, dtype='bool')
    result['halo_relative_r'] = np.zeros(size)
    result['halo_relative_vx'] = np.zeros(size)
    result['halo_relative_vy'] = np.zeros(size)
    result['halo_relative_vz'] = np.zeros(size)
    result['halo_relative_radial_vx'] = np.zeros(size)
    result['halo_relative_radial_vy'] = np.zeros(size)
    result['halo_relative_radial_vz'] = np.zeros(size)
    result['halo_relative_tan_vx'] = np.zeros(size)
    result['halo_relative_tan_vy'] = np.zeros(size)
    result['halo_relative_tan_vz'] = np.zeros(size)
    result['fof_halo_tag'] = np.zeros(size, dtype=np.int64)
    result['core_indexes'] = []
    return result

def construct_fof_merged_cores(cores, fof_colors):
    print("construct fof merged cores")
    unique_colors = np.unique(fof_colors)
    fof_merged_cores = init_fof_merged_cores(len(unique_colors))
    eta = dtk.ETA()
    for merged_core_index, unique_color in enumerate(unique_colors):
        if merged_core_index%1000 == 1:
            eta.print_progress(merged_core_index, len(unique_colors))
        cores_to_merge_indexes = np.where(fof_colors==unique_color)
        fof_merged_cores['x'][merged_core_index] = np.average(cores['x'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['y'][merged_core_index] = np.average(cores['y'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['z'][merged_core_index] = np.average(cores['z'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['vx'][merged_core_index] = np.average(cores['vx'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['vy'][merged_core_index] = np.average(cores['vy'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['vz'][merged_core_index] = np.average(cores['vz'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_r'][merged_core_index] = np.average(cores['halo_relative_r'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_vx'][merged_core_index] = np.average(cores['halo_relative_vx'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_vy'][merged_core_index] = np.average(cores['halo_relative_vy'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_vz'][merged_core_index] = np.average(cores['halo_relative_vz'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_radial_vx'][merged_core_index] = np.average(cores['halo_relative_radial_vx'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_radial_vy'][merged_core_index] = np.average(cores['halo_relative_radial_vy'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_radial_vz'][merged_core_index] = np.average(cores['halo_relative_radial_vz'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_tan_vx'][merged_core_index] = np.average(cores['halo_relative_tan_vx'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_tan_vy'][merged_core_index] = np.average(cores['halo_relative_tan_vy'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['halo_relative_tan_vz'][merged_core_index] = np.average(cores['halo_relative_tan_vz'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['fof_halo_tag'][merged_core_index] = np.min(cores['fof_halo_tag'][cores_to_merge_indexes])
        fof_merged_cores['radius'][merged_core_index] = np.average(cores['radius'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['infall_mass_avg'][merged_core_index] = np.average(cores['infall_mass'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['infall_mass'][merged_core_index] = np.sum(cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['infall_step'][merged_core_index] = np.average(cores['infall_step'][cores_to_merge_indexes], weights=cores['infall_mass'][cores_to_merge_indexes])
        fof_merged_cores['central'][merged_core_index] = np.max(cores['central'][cores_to_merge_indexes])
        # fof_merged_cores['core_indexes'].append(cores_to_merge_indexes)
    eta.print_done()
    return fof_merged_cores

def fof_core_and_halo_merging(link_length, cores_fname):
    print("\n\nFoF Merging of cores\n\n")
    halos = load_fof_halo_catalog('/home/dkorytov/data/AlphaQ/fof/m000-499.fofproperties')
    cores = load_core_catalog(cores_fname)
    combine_cores_with_fof_halos(cores, halos)
    if link_length != 0:
        # slct = (cores['z']<10)
        # cores = dtk.select_dict(cores, slct)
        fof_colors = link_cores_fof_ckdtree(cores, link_length)
        fof_merged_cores = construct_fof_merged_cores(cores, fof_colors)
    else:
        fof_merged_cores = cores
    dtk.save_dict_hdf5("cache/fof_merged_cores.ll={:0.3f}.hdf5".format(link_length), fof_merged_cores)
    
def get_unit_vector(dx, dy, dz):
    dr = np.sqrt(dx*dx + dy*dy + dz*dz)
    unit_x, unit_y, unit_z = dx/dr, dy/dr, dz/dr
    slct_zero = dr==0
    unit_x[slct_zero] = 1.0
    unit_y[slct_zero] = 0
    unit_z[slct_zero] = 0
    return unit_x, unit_y, unit_z

def combine_cores_with_fof_halos(cores, halos):
    matched_indexes = dtk.search_sorted(halos['fof_halo_tag'], cores['fof_halo_tag'])
    assert np.sum(matched_indexes == -1)==0, "hmm every core should be in a fof halo"
    dvx = cores['vx'] - halos['fof_halo_mean_vx'][matched_indexes]
    dvy = cores['vy'] - halos['fof_halo_mean_vy'][matched_indexes]
    dvz = cores['vz'] - halos['fof_halo_mean_vz'][matched_indexes]
    cores['halo_relative_vx'] = dvx
    cores['halo_relative_vy'] = dvy
    cores['halo_relative_vz'] = dvz
    dx  = cores['x'] - halos['fof_halo_center_x'][matched_indexes]
    dy  = cores['y'] - halos['fof_halo_center_y'][matched_indexes]
    dz  = cores['z'] - halos['fof_halo_center_z'][matched_indexes]
    dx_unit, dy_unit, dz_unit = get_unit_vector(dx, dy, dz)
    dvx_radial = dvx*dx_unit
    dvy_radial = dvy*dy_unit
    dvz_radial = dvz*dz_unit
    cores['halo_relative_radial_vx'] = dvx_radial
    cores['halo_relative_radial_vy'] = dvy_radial
    cores['halo_relative_radial_vz'] = dvz_radial
    cores['halo_relative_tan_vx']    = dvx-dvx_radial
    cores['halo_relative_tan_vy']    = dvy-dvy_radial
    cores['halo_relative_tan_vz']    = dvz-dvz_radial
    cores['halo_relative_r']         = np.sqrt(dx*dx + dy*dy + dz*dz )

def combine_subhalos_with_fof_halos(subhalos, halos):
    matched_indexes = dtk.search_sorted(halos['fof_halo_tag'], subhalos['fof_halo_tag'])
    assert np.sum(matched_indexes == -1) == 0, "hmm every subhalo should be in a fof halo"
    dvx = subhalos['subhalo_mean_vx'] - halos['fof_halo_mean_vx'][matched_indexes]
    dvy = subhalos['subhalo_mean_vy'] - halos['fof_halo_mean_vy'][matched_indexes]
    dvz = subhalos['subhalo_mean_vz'] - halos['fof_halo_mean_vz'][matched_indexes]
    subhalos['halo_relative_vx'] = dvx
    subhalos['halo_relative_vy'] = dvy
    subhalos['halo_relative_vz'] = dvz
    dx  = subhalos['subhalo_mean_x'] - halos['fof_halo_center_x'][matched_indexes]
    dy  = subhalos['subhalo_mean_y'] - halos['fof_halo_center_y'][matched_indexes]
    dz  = subhalos['subhalo_mean_z'] - halos['fof_halo_center_z'][matched_indexes]
    dx_unit, dy_unit, dz_unit = get_unit_vector(dx, dy, dz)
    dvx_radial = dvx*dx_unit
    dvy_radial = dvy*dy_unit
    dvz_radial = dvz*dz_unit
    subhalos['halo_relative_radial_vx'] = dvx_radial
    subhalos['halo_relative_radial_vy'] = dvy_radial
    subhalos['halo_relative_radial_vz'] = dvz_radial
    subhalos['halo_relative_tan_vx']    = dvx-dvx_radial
    subhalos['halo_relative_tan_vy']    = dvy-dvy_radial
    subhalos['halo_relative_tan_vz']    = dvz-dvz_radial
    subhalos['halo_relative_r']         = np.sqrt(dx*dx + dy*dy + dz*dz)

def plot_unmatched_cores_subhalos(cores, subhalos, particles, nth_largest, plot_radius, unmatched_cores=True, accum_particles=None):
    print("plotting unmatched core")
    if unmatched_cores:
        slct = cores['subhalo_index'] == -1
        core_unmatched_mass = cores['infall_mass']*slct #if mass == 0, it's a matched core        
        target_core_index = np.argsort(-core_unmatched_mass)[nth_largest]
        target_x, target_y, target_z = cores['x'][target_core_index], cores['y'][target_core_index], cores['z'][target_core_index]
    else:
        # plot only objects far from the boundary condition
        slct_inner_objects = select_inner_objects(subhalos['subhalo_mean_x'],
                                                  subhalos['subhalo_mean_y'],
                                                  subhalos['subhalo_mean_z'],
                                                  256.0,
                                                  2)
        
        # if it's a matched subhalo, record the mass as zero for the
        # sorting. Same thing for central/main body subhalos        
        slct = (subhalos['core_index'] == -1) & (subhalos['subhalo_tag'] != 0) & slct_inner_objects
        subhalo_unmatched_mass = subhalos['subhalo_mass']*slct
        target_subhalo_index = np.argsort(-subhalo_unmatched_mass)[nth_largest]
        target_x, target_y, target_z = subhalos['subhalo_mean_x'][target_subhalo_index], subhalos['subhalo_mean_y'][target_subhalo_index], subhalos['subhalo_mean_z'][target_subhalo_index]
    dx = cores['x']-target_x
    dy = cores['y']-target_y
    dz = cores['z']-target_z
    core_distance_to_target = dx*dx + dy*dy + dz*dz
    slct_cores_near_target = core_distance_to_target < plot_radius**2

    dx = subhalos['subhalo_mean_x']-target_x
    dy = subhalos['subhalo_mean_y']-target_y
    dz = subhalos['subhalo_mean_z']-target_z
    subhalo_distance_to_target = dx*dx + dy*dy + dz*dz
    slct_subhalos_near_target = subhalo_distance_to_target < plot_radius**2
    slct_massive_subhalos = subhalos['subhalo_mass']>3e11

    dx = particles['x']-target_x
    dy = particles['y']-target_y
    dz = particles['z']-target_z
    particle_distance_to_target = dx*dx + dy*dy + dz*dz
    slct_particles_near_target = particle_distance_to_target < plot_radius**2

    
    plt.figure()
    h, xbins, ybins = np.histogram2d(particles['x'][slct_particles_near_target],
                                     particles['y'][slct_particles_near_target], 
                                     bins = 100)
    plt.pcolor(xbins, ybins, h.T, cmap='Greys', norm=clr.LogNorm())
    if accum_particles is not None:
        dx = accum_particles['x']-target_x
        dy = accum_particles['y']-target_y
        dz = accum_particles['z']-target_z
        particle_distance_to_target = dx*dx + dy*dy + dz*dz
        slct_particles_near_target = particle_distance_to_target < plot_radius**2



        h, xbins, ybins = np.histogram2d(accum_particles['x'][slct_particles_near_target],
                                         accum_particles['y'][slct_particles_near_target], 
                                         bins = 100)
        plt.pcolor(xbins, ybins, h.T, cmap='Greens', norm=clr.LogNorm(), alpha=0.6)

    plt.scatter(subhalos['subhalo_mean_x'][slct_subhalos_near_target & slct_massive_subhalos],
                subhalos['subhalo_mean_y'][slct_subhalos_near_target & slct_massive_subhalos],
                marker='o', facecolors='none', edgecolors='b', s=50,
                label='Subhalos')
    plt.plot(cores['x'][slct_cores_near_target],
             cores['y'][slct_cores_near_target],
             'xr', label='Cores', ms=8)
    for core_index in np.where(slct_cores_near_target)[0]:
        if cores['subhalo_index'][core_index] == -1: # if this is an unmatched core, add a '+' to it
            plt.plot(cores['x'][core_index], cores['y'][core_index], 'r+', ms=8)
        else:
            paired_subhalo_index = cores['subhalo_index'][core_index]
            plt.plot([cores['x'][core_index], subhalos['subhalo_mean_x'][paired_subhalo_index]], 
                     [cores['y'][core_index], subhalos['subhalo_mean_y'][paired_subhalo_index]], 
                     'r-', lw=2.0)
    plt.axvline(target_x, color='k', ls=':')
    plt.axhline(target_y, color='k', ls=':')
    plt.plot([], [], ':k', label='Unmatched core')
    plt.plot([], [], '-r', lw=2.0, label='Core-Subhalo Pair')
    if unmatched_cores:
        plt.title("Unmatched Core Mass: {:.2e}".format(cores['infall_mass'][target_core_index]))
    else:
        plt.title("Unmatched subhalo Mass: {:.2e} subhalo tag:{}\nhalo tag: {}".format(
            subhalos['subhalo_mass'][target_subhalo_index],
            subhalos['subhalo_tag'][target_subhalo_index],
            subhalos['fof_halo_tag'][target_subhalo_index]))
    plt.legend(loc='best')
    
    dtk.save_figs("./figs/unmatched_cores/")
    plt.close()
        
def select_inner_objects(x, y, z, rL, exclusion_length):
    slct_x = (x > exclusion_length) & (x < rL-exclusion_length)
    slct_y = (y > exclusion_length) & (y < rL-exclusion_length)
    slct_z = (z > exclusion_length) & (z < rL-exclusion_length)
    return slct_x & slct_y & slct_z

def subhalo_comparison(load_cache=False, link_length=0.1, core_fname='/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties'):
    print("Subhalo Comparison")
    ALPHAQ_RL = 256.0
    # if not load_cache:
        # cores = load_core_catalog('/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties')
        # exit()
    cores = dtk.load_dict_hdf5('cache/fof_merged_cores.ll={:1.3f}.hdf5'.format(link_length))
    cores = dtk.select_dict(cores, cores['infall_mass']>1.1e11)
    subhalos = load_subhalo_catalog('/home/dkorytov/data/AlphaQ/subhalos/m000-499.subhaloproperties')
    particles = load_bighalo_particle_catalog('/media/luna1/dkorytov/data/AlphaQ/big_halo_prtcls2/m000-499.bighaloparticles')
    accum_particles = load_accum_particle_catalog('/media/luna1/dkorytov/data/AlphaQ/accum/m000-499.accumulatedcores')
    halos = load_fof_halo_catalog('/home/dkorytov/data/AlphaQ/fof/m000-499.fofproperties')
    combine_subhalos_with_fof_halos(subhalos, halos)

    # subhalos = dtk.select_dict(subhalos, subhalos['subhalo_tag'] != 0)
    # del cores['core_indexes']
    # cores  = dtk.select_dict(cores,    cores['central'] == 0)
    find_closest_ckdtree(cores, subhalos, 0.1, ALPHAQ_RL, match_cores_to_subhalos=False)
        # dtk.save_dict_hdf5('cache/cores.search=0.1.hdf5', cores)
        # dtk.save_dict_hdf5('cache/subhalos.search=0.1hdf5', subhalos)
    # else:#load cache
        # cores = dtk.load_dict_hdf5('cache/cores.search=0.1.hdf5')
        # subhalos = dtk.load_dict_hdf5('cache/subhalos.search=0.1.hdf5')
        # pass
    # if True:
    #     cores2 = load_core_catalog('/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties')
    #     subhalos2 = load_subhalo_catalog('/home/dkorytov/data/AlphaQ/subhalos/m000-499.subhaloproperties')
    #     cores = combine_dict(cores, cores2)
    #     subhalos = combine_dict(subhalos, subhalos2)

    for i in range(0, 300):
        plot_unmatched_cores_subhalos(cores, subhalos, particles, i, 2.0, unmatched_cores=False, accum_particles=accum_particles)
    edge_buffer = 4
    far_from_edge_x = ( subhalos['subhalo_mean_x'] > edge_buffer) & (subhalos['subhalo_mean_x'] < ALPHAQ_RL-edge_buffer)
    far_from_edge_y = ( subhalos['subhalo_mean_y'] > edge_buffer) & (subhalos['subhalo_mean_y'] < ALPHAQ_RL-edge_buffer)
    far_from_edge_z = ( subhalos['subhalo_mean_z'] > edge_buffer) & (subhalos['subhalo_mean_z'] < ALPHAQ_RL-edge_buffer)
    far_from_edge = far_from_edge_x & far_from_edge_y & far_from_edge_z
    far_from_edge = far_from_edge == far_from_edge
    slct = (subhalos['core_index'] != -1) & (subhalos['subhalo_tag'] !=0) & far_from_edge #& (subhalos['subhalo_count'] > 100)
    slct_coreless = (subhalos['core_index'] == -1) & (subhalos['subhalo_tag'] !=0) & far_from_edge
    print("Satellite Subhalos: ", np.sum(subhalos['subhalo_tag'] !=0))
    print("Satellite Subhalos with cores: ", np.sum(slct))
    print("fraction: ", np.float(np.sum(slct))/np.sum((subhalos['subhalo_tag'] !=0))) #& (subhalos['subhalo_count'] > 100)))

    

    plt.figure()
    h_with, xbins = np.histogram(subhalos['subhalo_mass'][slct], bins=np.logspace(10,15,100))
    plt.plot(dtk.bins_center(xbins), h_with, label='Subhalos w/ Cores')
    h_without, xbins = np.histogram(subhalos['subhalo_mass'][slct_coreless], bins=np.logspace(10,15,100))
    plt.plot(dtk.bins_center(xbins), h_without, label='Subhalos w/o Cores')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.xlabel('Subhalo Mass [$h^{-1}$ M$_\odot$]')
    plt.legend(loc='best')
    plt.axvline(1.6e11, ls='--', color='k', label='100 particles')
    plt.legend(loc='best', framealpha=0.0)

    plt.tight_layout()

    plt.figure()
    h_tot = h_with+h_without
    plt.semilogx(dtk.bins_center(xbins), h_with/h_tot)
    plt.ylabel('Fraction with Core')
    plt.axvline(1.6e11, ls='--', color='k', label='100 particles')
    plt.xlabel('Subhalo Mass [$h^{-1}$ M$_\odot$]')
    plt.legend(loc='best', framealpha=0.0)
    plt.tight_layout()

    plt.figure()
    slct_core_with_subhalo = cores['subhalo_index']!=-1
    h_with, xbins = np.histogram(cores['infall_mass'][slct_core_with_subhalo], bins=np.logspace(11, 16, 100))
    h_tot, xbins = np.histogram(cores['infall_mass'], bins=np.logspace(11, 16, 100))
    plt.semilogx(dtk.bins_center(xbins), h_with/h_tot)
    plt.ylabel('Fraction w/ Subhalo')
    plt.xlabel('Merged Core Infall Mass Sum [$h^{-1}$ M$_\odot$]')
    plt.ylim([0,1])
    plt.axvline(1.6e11, ls='--', color='k', label='100 particles')
    plt.legend(loc='best')
    plt.tight_layout()

    for lim in np.linspace(100, 1000, 25):
        mass_cut = lim * 1.6e9
        slct_satellite = subhalos['subhalo_tag'] != 0
        slct_with_core = (subhalos['core_index'] != -1) & slct_satellite
        slct_above_cut = (subhalos['subhalo_mass'] > mass_cut) & slct_satellite
        print("Mass cut: ", mass_cut)
        print("\t with core: ", np.sum(slct_with_core & slct_above_cut))
        print("\t above cut: ", np.sum(slct_above_cut))
        print("\tpercentage: ", np.float(np.sum(slct_with_core & slct_above_cut)/np.sum(slct_above_cut)))
        print("")

    plt.figure()
    slct_subhalo_with_core = subhalos['core_index']!=-1
    h_with, xbins = np.histogram(cores['subhalo_index'][slct_core_with_subhalo], bins=np.logspace(11, 16, 100))
    h_tot, xbins = np.histogram(cores['infall_mass'], bins=np.logspace(11, 16, 100))
    plt.semilogx(dtk.bins_center(xbins), h_with/h_tot)
    plt.ylabel('Fraction with Core')
    plt.xlabel('[$h^{-1}$ M$_\odot$]')
    plt.ylim([0,1])
    plt.axvline(1.6e11, ls='--', color='k', label='100 particles')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.figure()
    slct_core_with_subhalo = cores['subhalo_index']!=-1
    h_with, xbins = np.histogram(cores['halo_relative_r'][slct_core_with_subhalo], bins=np.logspace(0, 1, 100))
    h_tot, xbins = np.histogram(cores['halo_relative_r'], bins=np.logspace(0, 1, 100))
    plt.semilogx(dtk.bins_center(xbins), h_with/h_tot)
    plt.ylabel('Fraction w/ Subhalo')
    plt.xlabel('Distance to Halo Center [h$^{-1}$ Mpc]')
    plt.ylim([0,1])
    plt.tight_layout()

    plt.figure()
    h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], subhalos['core_mass'][slct], bins=np.logspace(10,15,100))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(plt.xlim(), plt.xlim(), '--k')
    plt.ylabel("Assigned Core Infall Mass [$h^{-1}$ M$_\odot$]")
    plt.xlabel('Subhalo Mass [$h^{-1}$ M$_\odot$]')
    plt.colorbar(label='Population Density')
    plt.tight_layout()    

    plt.figure()    
    distance = distance_between_core_and_subhalo(cores, subhalos['core_index'][slct], subhalos, np.arange(len(subhalos['subhalo_mean_x']))[slct], ALPHAQ_RL)
    print(distance)
    h, xbins = np.histogram(distance, bins=np.linspace(1e-3, .6, 100))
    xbins_cen = dtk.bins_avg(xbins)
    plt.plot(xbins_cen, h, '-')
    plt.xlabel('Core-Subhalo Separation [$h^{-1}$ Mpc]')
    plt.ylabel('Count')
    plt.tight_layout()

    plt.figure()
    mass_ratio = subhalos['subhalo_mass'][slct]/subhalos['core_mass'][slct]
    print("mass ratio greater than 1: ", np.float(np.sum(mass_ratio > 1))/mass_ratio.size)
    h, xbins = np.histogram(mass_ratio, bins=np.linspace(0, 2, 100))
    plt.plot(dtk.bins_avg(xbins), h, )
    plt.ylabel('Count')
    plt.xlabel('Subhalo Mass/Core Mass')
    plt.tight_layout()

    plt.figure()
    mass_ratio = subhalos['subhalo_mass'][slct]/subhalos['core_mass'][slct]
    h, xbins = np.histogram(mass_ratio, bins=np.logspace(-3, 1, 100))
    plt.plot(dtk.bins_avg(xbins), h, )
    plt.ylabel('Count')
    plt.xlabel('Subhalo Mass/Core Mass')
    plt.xscale('log')
    plt.tight_layout()

    plt.figure()
    h, xbins, ybins = np.histogram2d(cores['radius'][subhalos['core_index'][slct]], mass_ratio, bins=(np.logspace(-4, 0, 100), np.linspace(0, 2, 100)))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    mass_ratio_reasonable = mass_ratio<5
    plt.plot(dtk.bins_center(xbins), dtk.binned_average(cores['radius'][subhalos['core_index'][slct]][mass_ratio_reasonable], mass_ratio[mass_ratio_reasonable], xbins), 'r-', label='Average')
    plt.plot(dtk.bins_center(xbins), dtk.binned_median(cores['radius'][subhalos['core_index'][slct]], mass_ratio, xbins), 'r--', label='Median')
    plt.xlabel('Core Radius [$h^{-1}$ Mpc]')
    plt.ylabel('Subhalo Mass/Core Mass')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.colorbar(label='Population Density')
    plt.tight_layout()

    plt.figure()
    h, xbins, ybins = np.histogram2d(cores['infall_step'][subhalos['core_index'][slct]], mass_ratio, bins=(np.linspace(0, 500, 50), np.linspace(0, 2, 100)))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    mass_ratio_reasonable = mass_ratio<2
    plt.plot(dtk.bins_center(xbins), dtk.binned_average(cores['infall_step'][subhalos['core_index'][slct]][mass_ratio_reasonable], mass_ratio[mass_ratio_reasonable], xbins), 'r-', label='Average')
    plt.plot(dtk.bins_center(xbins), dtk.binned_median(cores['infall_step'][subhalos['core_index'][slct]][mass_ratio_reasonable], mass_ratio[mass_ratio_reasonable], xbins), 'r--', label='Median')
    plt.xlabel('Infall Step')
    plt.ylabel('Subhalo Mass/Core Mass')
    plt.legend(loc='best')
    plt.colorbar(label='Population Density')
    plt.tight_layout()

    plt.figure()
    h, xbins, ybins = np.histogram2d(cores['halo_relative_r'][subhalos['core_index'][slct]], mass_ratio, bins=(np.linspace(0, 2, 100), np.linspace(0, 2, 100)))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    mass_ratio_reasonable = mass_ratio<5
    plt.plot(dtk.bins_center(xbins), dtk.binned_average(cores['halo_relative_r'][subhalos['core_index'][slct]][mass_ratio_reasonable], mass_ratio[mass_ratio_reasonable], xbins), 'r-', label='Average')
    plt.plot(dtk.bins_center(xbins), dtk.binned_median(cores['halo_relative_r'][subhalos['core_index'][slct]], mass_ratio, xbins), 'r--', label='Median')
    plt.xlabel('Distance from Halo Center [h$^-1$ Mpc]')
    plt.ylabel('Subhalo Mass/Core Mass')
    plt.legend(loc='best')
    plt.colorbar(label='Population Density')
    plt.tight_layout()


    if True: # plot absolute velocities
        core_velocity_mags = velocity_magnitude(cores['vx'], cores['vy'], cores['vz'])
        subhalo_velocity_mags = velocity_magnitude(subhalos['subhalo_mean_vx'], subhalos['subhalo_mean_vy'], subhalos['subhalo_mean_vz'])
        velocity_ratio = core_velocity_mags[subhalos['core_index'][slct]]/subhalo_velocity_mags[slct]

        sh_vx, sh_vy, sh_vz = subhalos['subhalo_mean_vx'][slct],        subhalos['subhalo_mean_vy'][slct],        subhalos['subhalo_mean_vz'][slct]
        cr_vx, cr_vy, cr_vz = cores['vx'][subhalos['core_index'][slct]], cores['vy'][subhalos['core_index'][slct]], cores['vz'][subhalos['core_index'][slct]]
        core_subhalo_angle = np.arccos(angle_between_vectors(sh_vx, sh_vy, sh_vz, cr_vx, cr_vy, cr_vz))*180.0/np.pi
        
        #1d histogram of mags
        plt.figure() 
        h_core, xbins = np.histogram(core_velocity_mags[subhalos['core_index'][slct]], bins=np.logspace(0,4, 100))
        h_subhalo, _ = np.histogram(subhalo_velocity_mags[slct], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_core, label='Cores')
        plt.plot(dtk.bins_center(xbins), h_subhalo, label='Subhalo')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('count')
        plt.xlabel('Velocity Mag [km/s]')
        plt.legend(loc='best')
        plt.tight_layout()
        
        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_velocity_mags[slct], core_velocity_mags[subhalos['core_index'][slct]], bins = np.logspace(1.5, 4, 100))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_velocity_mags[slct], core_velocity_mags[subhalos['core_index'][slct]], xbins)
        plt.plot(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_velocity_mags[slct],  core_velocity_mags[subhalos['core_index'][slct]], xbins,)
        plt.plot(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Subahlo Velocity Magnitude')
        plt.ylabel("Core Velocity Magnitude")
        plt.plot(plt.ylim(), plt.ylim(), '--k')
        plt.legend(loc='best')
        plt.tight_layout()
    
        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_velocity_mags[slct], velocity_ratio, bins=(np.logspace(1, 4, 100), np.logspace(-1, 1, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_velocity_mags[slct], velocity_ratio, xbins)
        plt.loglog(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        plt.xlabel('Subhalo Velocity Magnitude [km/h]')
        plt.ylabel('Velocity Magnitude Ratio (core/Subhalo)')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], core_subhalo_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalos['subhalo_mass'][slct], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalos['subhalo_mass'][slct], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Mass")
        plt.ylabel("Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Core Infall Mass")
        plt.ylabel("Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_velocity_mags[slct], core_subhalo_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_velocity_mags[slct], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_velocity_mags[slct], core_subhalo_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Velocity Magnitude")
        plt.ylabel("Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()



    # Core Subhalo relative velocities
    if True: # plot relative velocities
        core_relative_velocity_mags = velocity_magnitude(cores['halo_relative_vx'], cores['halo_relative_vy'], cores['halo_relative_vz'])
        subhalo_relative_velocity_mags = velocity_magnitude(subhalos['halo_relative_vx'], subhalos['halo_relative_vy'], subhalos['halo_relative_vz'])
        relative_velocity_ratio = core_relative_velocity_mags[subhalos['core_index'][slct]]/subhalo_relative_velocity_mags[slct]

        sh_vx, sh_vy, sh_vz = subhalos['halo_relative_vx'][slct], subhalos['halo_relative_vy'][slct], subhalos['subhalo_mean_vz'][slct]
        cr_vx, cr_vy, cr_vz = cores['halo_relative_vx'][subhalos['core_index'][slct]], cores['halo_relative_vy'][subhalos['core_index'][slct]], cores['halo_relative_vz'][subhalos['core_index'][slct]]
        core_subhalo_relative_angle = np.arccos(angle_between_vectors(sh_vx, sh_vy, sh_vz, cr_vx, cr_vy, cr_vz))*180.0/np.pi

        plt.figure() 
        h_core, xbins = np.histogram(core_relative_velocity_mags[subhalos['core_index'][slct]], bins=np.logspace(0,4, 100))
        h_subhalo, _ = np.histogram(subhalo_relative_velocity_mags[slct], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_core, label='Cores')
        plt.plot(dtk.bins_center(xbins), h_subhalo, label='Subhalo')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('count')
        plt.xlabel('Relative Velocity Mag [km/s]')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_velocity_mags[slct], core_relative_velocity_mags[subhalos['core_index'][slct]], bins = np.logspace(1.5, 4, 100))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_velocity_mags[slct], core_relative_velocity_mags[subhalos['core_index'][slct]], xbins)
        plt.plot(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_velocity_mags[slct],  core_relative_velocity_mags[subhalos['core_index'][slct]], xbins,)
        plt.plot(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Subahlo Relative Velocity Magnitude')
        plt.ylabel("Core Relative Velocity Magnitude")
        plt.plot(plt.ylim(), plt.ylim(), '--k')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_velocity_mags[slct], relative_velocity_ratio, bins=(np.logspace(1, 4, 100), np.logspace(-1, 1, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_velocity_mags[slct], relative_velocity_ratio, xbins)
        plt.loglog(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        plt.xlabel('Subhalo Relative Velocity Magnitude [km/h]')
        plt.ylabel('Relative Velocity Magnitude Ratio (core/Subhalo)')
        plt.legend(loc='best')
        plt.tight_layout()


        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], core_subhalo_relative_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalos['subhalo_mass'][slct], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalos['subhalo_mass'][slct], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Mass")
        plt.ylabel("Relative Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Core Infall Mass")
        plt.ylabel("Relative Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_velocity_mags[slct], core_subhalo_relative_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_velocity_mags[slct], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_velocity_mags[slct], core_subhalo_relative_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Relative Velocity Magnitude")
        plt.ylabel("Relative Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

    if True: # plot radial relative velocities
        core_relative_radial_velocity_mags = velocity_magnitude(cores['halo_relative_radial_vx'], cores['halo_relative_radial_vy'], cores['halo_relative_radial_vz'])
        subhalo_relative_radial_velocity_mags = velocity_magnitude(subhalos['halo_relative_radial_vx'], subhalos['halo_relative_radial_vy'], subhalos['halo_relative_radial_vz'])
        relative_radial_velocity_ratio = core_relative_radial_velocity_mags[subhalos['core_index'][slct]]/subhalo_relative_radial_velocity_mags[slct]

        sh_vx, sh_vy, sh_vz = subhalos['halo_relative_radial_vx'][slct], subhalos['halo_relative_radial_vy'][slct], subhalos['subhalo_mean_vz'][slct]
        cr_vx, cr_vy, cr_vz = cores['halo_relative_radial_vx'][subhalos['core_index'][slct]], cores['halo_relative_radial_vy'][subhalos['core_index'][slct]], cores['halo_relative_radial_vz'][subhalos['core_index'][slct]]
        core_subhalo_relative_radial_angle = np.arccos(angle_between_vectors(sh_vx, sh_vy, sh_vz, cr_vx, cr_vy, cr_vz))*180.0/np.pi

        plt.figure() 
        print(np.sum(~np.isfinite(core_relative_radial_velocity_mags[subhalos['core_index'][slct]])))
        print(np.min(core_relative_radial_velocity_mags[subhalos['core_index'][slct]]))
        print(np.max(core_relative_radial_velocity_mags[subhalos['core_index'][slct]]))

        h_core, xbins = np.histogram(core_relative_radial_velocity_mags[subhalos['core_index'][slct]], bins=np.logspace(0,4, 100))
        h_subhalo, _ = np.histogram(subhalo_relative_radial_velocity_mags[slct], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_core, label='Cores')
        plt.plot(dtk.bins_center(xbins), h_subhalo, label='Subhalo')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('count')
        plt.xlabel('Relative Radial Velocity Mag [km/s]')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], core_relative_radial_velocity_mags[subhalos['core_index'][slct]], bins = np.logspace(1.5, 4, 100))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], core_relative_radial_velocity_mags[subhalos['core_index'][slct]], xbins)
        plt.plot(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_radial_velocity_mags[slct],  core_relative_radial_velocity_mags[subhalos['core_index'][slct]], xbins,)
        plt.plot(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Subahlo Relative Radial Velocity Magnitude')
        plt.ylabel("Core Relative Radial Velocity Magnitude")
        plt.plot(plt.ylim(), plt.ylim(), '--k')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], relative_radial_velocity_ratio, bins=(np.logspace(1, 4, 100), np.logspace(-1, 1, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], relative_radial_velocity_ratio, xbins)
        plt.loglog(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        plt.xlabel('Subhalo Relative Radial Velocity Magnitude [km/h]')
        plt.ylabel('Relative Radial Velocity Magnitude Ratio (core/Subhalo)')
        plt.legend(loc='best')
        plt.tight_layout()


        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Mass")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Core Infall Mass")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Relative Radial Velocity Magnitude")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

    if True: # plot radial relative velocities
        core_relative_radial_velocity_mags = velocity_magnitude(cores['halo_relative_radial_vx'], cores['halo_relative_radial_vy'], cores['halo_relative_radial_vz'])
        subhalo_relative_radial_velocity_mags = velocity_magnitude(subhalos['halo_relative_radial_vx'], subhalos['halo_relative_radial_vy'], subhalos['halo_relative_radial_vz'])
        relative_radial_velocity_ratio = core_relative_radial_velocity_mags[subhalos['core_index'][slct]]/subhalo_relative_radial_velocity_mags[slct]

        sh_vx, sh_vy, sh_vz = subhalos['halo_relative_radial_vx'][slct], subhalos['halo_relative_radial_vy'][slct], subhalos['subhalo_mean_vz'][slct]
        cr_vx, cr_vy, cr_vz = cores['halo_relative_radial_vx'][subhalos['core_index'][slct]], cores['halo_relative_radial_vy'][subhalos['core_index'][slct]], cores['halo_relative_radial_vz'][subhalos['core_index'][slct]]
        core_subhalo_relative_radial_angle = np.arccos(angle_between_vectors(sh_vx, sh_vy, sh_vz, cr_vx, cr_vy, cr_vz))*180.0/np.pi

        plt.figure() 
        h_core, xbins = np.histogram(core_relative_radial_velocity_mags[subhalos['core_index'][slct]], bins=np.logspace(0,4, 100))
        h_subhalo, _ = np.histogram(subhalo_relative_radial_velocity_mags[slct], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_core, label='Cores')
        plt.plot(dtk.bins_center(xbins), h_subhalo, label='Subhalo')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('count')
        plt.xlabel('Relative Radial Velocity Mag [km/s]')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], core_relative_radial_velocity_mags[subhalos['core_index'][slct]], bins = np.logspace(1.5, 4, 100))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], core_relative_radial_velocity_mags[subhalos['core_index'][slct]], xbins)
        plt.plot(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_radial_velocity_mags[slct],  core_relative_radial_velocity_mags[subhalos['core_index'][slct]], xbins,)
        plt.plot(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Subahlo Relative Radial Velocity Magnitude')
        plt.ylabel("Core Relative Radial Velocity Magnitude")
        plt.plot(plt.ylim(), plt.ylim(), '--k')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], relative_radial_velocity_ratio, bins=(np.logspace(1, 4, 100), np.logspace(-1, 1, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], relative_radial_velocity_ratio, xbins)
        plt.loglog(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        plt.xlabel('Subhalo Relative Radial Velocity Magnitude [km/h]')
        plt.ylabel('Relative Radial Velocity Magnitude Ratio (core/Subhalo)')
        plt.legend(loc='best')
        plt.tight_layout()


        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalos['subhalo_mass'][slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Mass")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Core Infall Mass")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Relative Radial Velocity Magnitude")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['halo_relative_r'][subhalos['core_index'][slct]], core_subhalo_relative_radial_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_radial_velocity_mags[slct], core_subhalo_relative_radial_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Radial distance to halo center [$h^{-1}$ Mpc]")
        plt.ylabel("Relative Radial Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

    if True: # plot tan relative velocities
        core_relative_tan_velocity_mags = velocity_magnitude(cores['halo_relative_tan_vx'], cores['halo_relative_tan_vy'], cores['halo_relative_tan_vz'])
        subhalo_relative_tan_velocity_mags = velocity_magnitude(subhalos['halo_relative_tan_vx'], subhalos['halo_relative_tan_vy'], subhalos['halo_relative_tan_vz'])
        relative_tan_velocity_ratio = core_relative_tan_velocity_mags[subhalos['core_index'][slct]]/subhalo_relative_tan_velocity_mags[slct]

        sh_vx, sh_vy, sh_vz = subhalos['halo_relative_tan_vx'][slct], subhalos['halo_relative_tan_vy'][slct], subhalos['subhalo_mean_vz'][slct]
        cr_vx, cr_vy, cr_vz = cores['halo_relative_tan_vx'][subhalos['core_index'][slct]], cores['halo_relative_tan_vy'][subhalos['core_index'][slct]], cores['halo_relative_tan_vz'][subhalos['core_index'][slct]]
        core_subhalo_relative_tan_angle = np.arccos(angle_between_vectors(sh_vx, sh_vy, sh_vz, cr_vx, cr_vy, cr_vz))*180.0/np.pi

        plt.figure() 
        h_core, xbins = np.histogram(core_relative_tan_velocity_mags[subhalos['core_index'][slct]], bins=np.logspace(0,4, 100))
        h_subhalo, _ = np.histogram(subhalo_relative_tan_velocity_mags[slct], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_core, label='Cores')
        plt.plot(dtk.bins_center(xbins), h_subhalo, label='Subhalo')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('count')
        plt.xlabel('Relative Tan Velocity Mag [km/s]')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_tan_velocity_mags[slct], core_relative_tan_velocity_mags[subhalos['core_index'][slct]], bins = np.logspace(1.5, 4, 100))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_tan_velocity_mags[slct], core_relative_tan_velocity_mags[subhalos['core_index'][slct]], xbins)
        plt.plot(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_tan_velocity_mags[slct],  core_relative_tan_velocity_mags[subhalos['core_index'][slct]], xbins,)
        plt.plot(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Subahlo Relative Tan Velocity Magnitude')
        plt.ylabel("Core Relative Tan Velocity Magnitude")
        plt.plot(plt.ylim(), plt.ylim(), '--k')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_tan_velocity_mags[slct], relative_tan_velocity_ratio, bins=(np.logspace(1, 4, 100), np.logspace(-1, 1, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_tan_velocity_mags[slct], relative_tan_velocity_ratio, xbins)
        plt.loglog(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        plt.xlabel('Subhalo Relative Tan Velocity Magnitude [km/h]')
        plt.ylabel('Relative Tan Velocity Magnitude Ratio (core/Subhalo)')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalos['subhalo_mass'][slct], core_subhalo_relative_tan_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalos['subhalo_mass'][slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalos['subhalo_mass'][slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Mass")
        plt.ylabel("Relative Tan Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_tan_angle, bins=(np.logspace(11,16,100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(cores['infall_mass'][subhalos['core_index'][slct]], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Core Infall Mass")
        plt.ylabel("Relative Tan Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(subhalo_relative_tan_velocity_mags[slct], core_subhalo_relative_tan_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_tan_velocity_mags[slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_tan_velocity_mags[slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Subhalo Relative Tan Velocity Magnitude")
        plt.ylabel("Relative Tan Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

        plt.figure()
        h, xbins, ybins = np.histogram2d(cores['halo_relative_r'][subhalos['core_index'][slct]], core_subhalo_relative_tan_angle, bins=(np.logspace(1, 4, 100), np.linspace(0, 180, 100)))
        plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
        binned_average = dtk.binned_average(subhalo_relative_tan_velocity_mags[slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_average, '-r', label='Average')
        binned_median = dtk.binned_median(subhalo_relative_tan_velocity_mags[slct], core_subhalo_relative_tan_angle, xbins)
        plt.semilogx(dtk.bins_center(xbins), binned_median, '--r', label='Median')
        plt.xlabel("Radial distance to halo center [$h^{-1}$ Mpc]")
        plt.ylabel("Relative Tan Velocity Angle Separation")
        plt.legend(loc='best')
        plt.tight_layout()

    matched_cores = cores['subhalo_index'] != -1
    unmatched_cores = ~matched_cores
    
    bins_dict = {'radius': np.logspace(-3,0, 100),
                 'infall_mass': np.logspace(11, 16, 100),
                 'infall_step': np.linspace(0, 500, 50)}
    for x_axis_quant, y_axis_quant in [['infall_mass', 'radius'], ['infall_mass', 'infall_step'], ['infall_step', 'radius']]:
        f, axs = plt.subplots(1,2, figsize=(9,5))
        axs[0].set_title('Unmatched Cores')
        axs[1].set_title('Matched Cores')
        xbins = bins_dict[x_axis_quant]
        ybins = bins_dict[y_axis_quant]
        #        for ax, slct_cores in zip([axs[0], axs[1]], [unmatched_cores, matched_cores]):
        h_matched, _, _ = np.histogram2d(cores[x_axis_quant][matched_cores], cores[y_axis_quant][matched_cores], bins=(xbins, ybins))
        h_unmatched, _, _ = np.histogram2d(cores[x_axis_quant][unmatched_cores], cores[y_axis_quant][unmatched_cores], bins=(xbins, ybins))
        vmax = np.max((np.max(h_matched), np.max(h_unmatched)))
        vmin = 1
        axs[0].pcolor(xbins, ybins, h_unmatched.T, cmap='Blues', norm=clr.LogNorm(), vmax=vmax, vmin=vmin)
        axs[0].set_xlabel(x_axis_quant.replace('_', '\_'))
        axs[0].set_ylabel(y_axis_quant.replace('_', '\_'))
        # axs.plot(dtk.bins_center(xbins), 
        if x_axis_quant != 'infall_step':
            axs[0].set_xscale('log')
        if y_axis_quant != 'infall_step':
            axs[0].set_yscale('log')
            # plt.tight_layout()
        # axs[0].colorbar(label='Population Density')

        axs[1].pcolor(xbins, ybins, h_matched.T, cmap='Blues', norm=clr.LogNorm(), vmax=vmax, vmin=vmin)
        axs[1].set_xlabel(x_axis_quant.replace('_', '\_'))
        axs[1].set_ylabel(y_axis_quant.replace('_', '\_'))
        # axs.plot(dtk.bins_center(xbins), 
        if x_axis_quant != 'infall_step':
            axs[1].set_xscale('log')
        if y_axis_quant != 'infall_step':
            axs[1].set_yscale('log')
            # plt.colorbar(label='Population Density (Both Panels)')
        # plt.tight_layout()


    for x_axis_quant in ['infall_mass', 'radius', 'infall_step']:
        plt.figure()
        xbins = bins_dict[x_axis_quant]
        h_with, _ = np.histogram(cores[x_axis_quant][matched_cores], bins=xbins)
        h_without, _ = np.histogram(cores[x_axis_quant][unmatched_cores], bins=xbins)
        plt.plot(dtk.bins_center(xbins), h_with, label='Merged Cores with Subhalo')
        plt.plot(dtk.bins_center(xbins), h_without, label='Merged Cores without Subhalo')
        plt.plot(dtk.bins_center(xbins), h_with+h_without, '-k', label="All Merged Cores")
        if x_axis_quant != 'infall_step':
            plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.ylabel('count')
        plt.xlabel(x_axis_quant.replace('_', '\_'))
        plt.tight_layout()

    print("subhalos w/ cores", np.float(np.sum(slct))/slct.size)

    # plt.show()

if __name__ == "__main__":
    if len(sys.argv)>1:
        load_cache = sys.argv[1]=='True' or sys.argv[1]=='true' or sys.argv[1]=='t'
    else:
        load_cache = True
    core_link_length=0.# 00
    # core_fname='/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties'
    # core_fname='/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties'
    # core_fname='/home/dkorytov/data/AlphaQ/core_catalog5/07_13_17.AlphaQ.499.coreproperties'
    cores_fname='/media/luna1/dkorytov/data/AlphaQ/core_catalog6_0.1/09_03_2019.AQ.499.coreproperties'
    fof_core_and_halo_merging(link_length=core_link_length, cores_fname=cores_fname)
    subhalo_comparison(load_cache=load_cache, link_length=core_link_length)
    dtk.save_figs("figs/")
