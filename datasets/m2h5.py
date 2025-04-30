import numpy as np 
from glob import glob
import os
import h5py

import numba as nb
#from numba import cuda
#检测一下GPU是否可用
#print(cuda.gpus)



def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def read_points_candidates(points_filename, candidates_filename):
    points = []
    with open(points_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            
            v0 = float(oneline[2])
            v1 = float(oneline[3])
            v2 = float(oneline[4])

            points.append([v0,v1,v2])
    faces = []
    labels = []
    with open(candidates_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            if oneline[0]=='Face':
                faces.append([int(oneline[2]),int(oneline[3]),int(oneline[4]),int(oneline[5])])
                labels.append(int(oneline[6]))
    return points, faces, labels

def read_points_delete_duplicate_candidates(points_filename, candidates_filename):
    points = []
    with open(points_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            if "-nan(ind)" in oneline:
                print('points_filename=',points_filename)
                return [], [], [],[]
            else:
                points.append([float(oneline[2]),float(oneline[3]),float(oneline[4])])
    faces = []
    labels = []
    face_vertex = {}
    face_features = []
    #flag=False
    with open(candidates_filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            if oneline[0]=='Face':
                id_sort = [int(oneline[2]),int(oneline[3]),int(oneline[4]),int(oneline[5])]
                id_sort.sort()
                #print(id_sort)
                key = str(id_sort[0])+','+str(id_sort[1])+','+str(id_sort[2])+','+str(id_sort[3])
                #print(key)
                if key in face_vertex:
                    face_vertex[key] += 1
                else:
                    face_vertex.update({key: 1})
                    faces.append([int(oneline[2]),int(oneline[3]),int(oneline[4]),int(oneline[5])])
                    #labels.append(int(oneline[6]))
                    labels.append(int(oneline[24]))
                    #print('oneline[24]=',oneline[24])
                    oneff = []
                    for i in range(6,24):
                        '''
                        if oneline[i] == "-nan(ind)":
                            oneff.append(-0.57735)
                            print('candidates_filename=',candidates_filename)
                            #flag=True
                            #break
                        else:
                            oneff.append(float(oneline[i]))
                        '''
                        oneff.append(float(oneline[i]))
                        
                    face_features.append(oneff)
                    #if flag:
                    #    break
    return points, faces, labels,face_features



#@nb.jit
def point_sets_distance(points, p_sets):
    n_p = points.shape[0]
    n_s = p_sets.shape[0]
    
    dist = np.zeros(n_p)
    span= 1000
    step = n_p//span
    tile_s = np.tile(p_sets[np.newaxis,:,:], (span,1,1))
    for i in range(step):
        onestep = points[i*span:(i+1)*span,:][:,np.newaxis,:]
        temp = np.tile(onestep, (1,n_s,1))
        dist_p2s = np.linalg.norm(temp-tile_s, axis=2)
        #print('dist_p2s.shape=',dist_p2s.shape)
        dist[i*span:(i+1)*span] = np.min(dist_p2s, axis=1)

    rem = n_p-span*step
    if rem>0:
        onestep = points[-rem:,:][:,np.newaxis,:]
        temp = np.tile(onestep, (1,n_s,1))
        tile_s = np.tile(p_sets[np.newaxis,:,:], (rem,1,1))
        dist_p2s = np.linalg.norm(temp-tile_s, axis=2)
        dist[-rem:] = np.min(dist_p2s, axis=1)
    return dist


def write_point_obj(file, vs, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))



if __name__ == '__main__':
    input_dir='D:/data/mesh/DataNonoise/'
    output_dir='D:/data/mesh/restart/DataNonoise-pc-obj/'
    
    if not os.path.exists(output_dir):       
        os.makedirs(output_dir)


    points_dir = os.path.join(input_dir, 'points')
    point_names = sorted(glob(points_dir+'/*.'+'m'))
    start = len(points_dir)
    
    for pointName in point_names:
        #print(pointName)
        #candidates_filename = input_dir+'dataset' + pointName[start:-12]+'DataSet.m'
        candidates_filename = input_dir+'faces/' + pointName[start:-4]+'gt.m'
        #onepoints, onefaces, onelabels = read_points_candidates(pointName, candidates_filename)
        onepoints, onefaces, onelabels, face_infos = read_points_delete_duplicate_candidates(pointName, candidates_filename)
        if len(onepoints)==0:
            continue
        points = np.array(onepoints, dtype=np.float32)
        #print('points=',points)
        points,_,_ = normalize_point_cloud(points)
        faces = np.array(onefaces, dtype=np.int32)-1
        labels = np.array(onelabels, dtype=np.int32)
        face_infos = np.array(face_infos, dtype=np.float32)

        out_filename = output_dir+ pointName[start:-5]+'.h5'
        with h5py.File(out_filename, 'w') as f:
            f['points'] = points
            f['faces'] = faces
            f['labels'] = labels
            f['face_infos'] = face_infos
