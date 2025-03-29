import numpy as np 
from glob import glob
import os
import h5py

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance[:,np.newaxis]
    return input, centroid,furthest_distance

def read_points_candidates(points_filename, candidates_filename):
    points = []
    with open(points_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            points.append([float(oneline[2]),float(oneline[3]),float(oneline[4])])
    faces = []
    labels = []
    with open(candidates_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            if oneline[0]=='Face':
                faces.append([int(oneline[2]),int(oneline[3]),int(oneline[4]),int(oneline[5])])
                labels.append(int(oneline[6]))
    #points, faces, labels = np.array(points),np.array(faces),np.array(labels)
    #print('points.shape=',points.shape)
    #print('faces.shape=',faces.shape)
    #print('labels.shape=',labels.shape)
    return points, faces, labels

if __name__ == '__main__':
    input_dir='E:/data/mesh/'
    output_dir='E:/data/mesh/h5/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points = []
    faces = []
    labels = []
    point_names = sorted(glob(input_dir+'coordinate'+'/*.'+'m'))[:6]
    start = len(input_dir)+10
    face_nums = 12*1024
    for pointName in point_names:
        #print(pointName)
        candidates_filename = input_dir+'dataset' + pointName[start:-12]+'DataSet.m'
        onepoints, onefaces, onelabels = read_points_candidates(pointName, candidates_filename)
        if len(onefaces)!=face_nums or len(onelabels)!=face_nums:
            print(pointName,': the number of face or label is not equal ',face_nums)
            continue
        points.append(onepoints)
        faces.append(onefaces)
        labels.append(onelabels)
        
    points = np.array(points, dtype=np.float32)
    points,_,_ = normalize_point_cloud(points)
    faces = np.array(faces, dtype=np.int16)
    labels = np.array(labels, dtype=np.uint8)
    print('points.shape=',points.shape)
    print('faces.shape=',faces.shape)
    print('labels.shape=',labels.shape)
    out_filename = output_dir+'point1024candidate12.h5'
    #out_filename = output_dir+'test.h5'
    with h5py.File(out_filename, 'w') as f:
        f['points'] = points
        f['faces'] = faces
        f['labels'] = labels
