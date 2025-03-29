# -*- coding: utf-8 -*-
# @Time        : 12/07/2024 17:10 PM
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com
import numpy as np
import plyfile
from glob import glob
import os
import h5py


def write_ply_points(filename, points):
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats = 'f8, f8, f8')
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el]).write(filename)
    

def load_points_candidates(file_name):
    f = h5py.File(file_name, 'r')
    points = f['points'][:]
    faces = f['faces'][:]
    labels = f['labels'][:]
    f.close()
        
    return points,faces,labels


def paserH52ply(data_dir,save_dir):

    # Start loading
    mesh_names = sorted(glob(data_dir+'/*.'+'h5'))
    start = len(data_dir)
    num_mesh = len(mesh_names)
    for meshname in mesh_names:
        points,faces,labels = load_points_candidates(meshname)
        baseName = meshname[start:-2]+'ply'
        out_filename = os.path.join(save_dir,baseName)
        write_ply_points(out_filename,points)
        

if __name__ == '__main__':
    input_dir='/user38/data/mesh/quadrilateral/20240630/'
    output_dir='/user38/data/mesh/quadrilateral/20240630-pc/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paserH52ply(input_dir,output_dir)