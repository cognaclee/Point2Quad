# -*- coding: utf-8 -*-
# @Time        : 27/04/2024 21:10 PM
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com
import numpy as np
import math


def dot(vec_A, vec_B):
    return np.sum(vec_A*vec_B, axis=-1)
    
def angle(vec_A, vec_B):
    len_a = np.linalg.norm(x=vec_A, ord=2,axis = -1)
    len_b = np.linalg.norm(x=vec_B, ord=2,axis = -1)
    if abs(len_a)<1e-9 or abs(len_b)<1e-9:
        return 0
    else:
        return dot(vec_A, vec_B)/(len_a*len_b)

# threshold shoud be a little larger than 1
def filter_face_with_three_commom_vertices(points, faces, probs,threshold=1.1):
    faces_sort = np.sort(faces,axis = 1)
    f_nums = faces.shape[0]
    #right = 0.5*math.pi
    flag = [True]*f_nums
    for i in range(f_nums): 
        if flag[i] == False:
            continue
        for j in range(i+1,f_nums):
            if flag[j] == False:
                continue
            if sum( faces_sort[i,:]==  faces_sort[j,:])>=3:
                '''
                ang_i = [0]*4
                ang_j = [0]*4
                for k in range(4):
                    vec_A = points[faces[i][(k-1)%4],:] - points[faces[i][k],:]
                    vec_B = points[faces[i][(k+1)%4],:] - points[faces[i][k],:]
                    ang_i[k]=angle(vec_A, vec_B)
                    vec_A = points[faces[j][(k-1)%4],:] - points[faces[j][k],:]
                    vec_B = points[faces[j][(k+1)%4],:] - points[faces[j][k],:]
                    ang_j[k]=angle(vec_A, vec_B)
                #ang_i_sum = ang_i[0]+ang_i[1]+ang_i[2]+ang_i[3]
                ang_i_err = (ang_i[0])**2+(ang_i[1])**2+(ang_i[2])**2+(ang_i[3])**2
                
                #ang_j_sum = ang_j[0]+ang_j[1]+ang_j[2]+ang_j[3]
                ang_j_err = (ang_j[0])**2+(ang_j[1])**2+(ang_j[2])**2+(ang_j[3])**2
                
                if ang_i_err>threshold*ang_j_err:
                    flag[i] = False
                elif threshold*ang_i_err<ang_j_err:
                    flag[j] = False
                else:
                    if probs[i]<probs[j]:
                        flag[i] = False
                    else:
                        flag[j] = False
                '''
                if probs[i]<probs[j]:
                    flag[i] = False
                else:
                    flag[j] = False
            if flag[i] == False:
                break
    
    new_faces = faces[flag,:]
    new_probs = probs[flag]
    return new_faces,new_probs
    
# error shoud should be close to 0;# threshold shoud be one of 0\1\2\3\4
def filter_face_with_angle(points, faces, error=0.45,threshold=1):
    f_nums = faces.shape[0]
    flag = [True]*f_nums
    for i in range(f_nums):
        cnt=0
        for k in range(4):
            vec_A = points[faces[i][(k-1)%4],:] - points[faces[i][k],:]
            vec_B = points[faces[i][(k+1)%4],:] - points[faces[i][k],:]
            ang=angle(vec_A, vec_B)
            if abs(ang)>error:
                cnt+=1
            if cnt>threshold:
                break
        if cnt>threshold:
            flag[i] = False
    
    new_faces = faces[flag,:]
    return new_faces
    
# threshold shoud be a little larger than 1
def filter_face_with_diagonal_edge(points, faces,threshold=1.1):
    f_nums = faces.shape[0]
    flag = [True]*f_nums
    angle_sum = 2.0*math.pi
    faces_sort = np.sort(faces,axis = 1)
    for i in range(f_nums): 
        if flag[i] == False:
            continue
        for j in range(i+1,f_nums):
            if flag[j] == False:
                continue
            if sum( faces_sort[i,:]==  faces_sort[j,:])>=2:
                diagonal_flag = False
                for k in range(4):
                    if (faces[i,0] == faces[j,k] and faces[i,2] == faces[j,(k+1)%4])\
                        or (faces[i,2] == faces[j,k] and faces[i,0] == faces[j,(k+1)%4]):
                        diagonal_flag = True
                        break
                    if (faces[i,1] == faces[j,k] and faces[i,3] == faces[j,(k+1)%4])\
                        or (faces[i,3] == faces[j,k] and faces[i,1] == faces[j,(k+1)%4]):
                        diagonal_flag = True
                        break
                if diagonal_flag:
                    ang_i = [0]*4
                    ang_j = [0]*4
                    for k in range(4):
                        vec_A = points[faces[i][(k-1)%4],:] - points[faces[i][k],:]
                        vec_B = points[faces[i][(k+1)%4],:] - points[faces[i][k],:]
                        ang_i[k]=angle(vec_A, vec_B)
                        vec_A = points[faces[j][(k-1)%4],:] - points[faces[j][k],:]
                        vec_B = points[faces[j][(k+1)%4],:] - points[faces[j][k],:]
                        ang_j[k]=angle(vec_A, vec_B)
                    ang_i_sum = ang_i[0]+ang_i[1]+ang_i[2]+ang_i[3]
                    ang_i_err = (ang_i[0])**2+(ang_i[1])**2+(ang_i[2])**2+(ang_i[3])**2
                
                    ang_j_sum = ang_j[0]+ang_j[1]+ang_j[2]+ang_j[3]
                    ang_j_err = (ang_j[0])**2+(ang_j[1])**2+(ang_j[2])**2+(ang_j[3])**2
                
                    if ang_i_err>threshold*ang_j_err:
                        flag[i] = False
                    elif threshold*ang_i_err<ang_j_err:
                        flag[j] = False
                    else:
                        if abs(ang_i_sum-angle_sum)>abs(ang_j_sum-angle_sum):
                            flag[i] = False
                        else:
                            flag[j] = False
                        '''
                        if abs(ang_i_sum-angle_sum)>threshold*abs(ang_j_sum-angle_sum):
                            flag[i] = False
                        elif threshold*abs(ang_i_sum-angle_sum)<abs(ang_j_sum-angle_sum):
                            flag[j] = False
                        '''
            
            if flag[i] == False:
                    break
    
    new_faces = faces[flag,:]
    return new_faces
    

def pred2facesAndprobs(predicted, cand_faces, probability, duplicate=True):
    faces = []
    probs = []
    
    for i, data in enumerate(predicted):   # labels or predicted
        if data == 1:
            faces.append(cand_faces[i,:])
            probs.append(probability[i])
    faces = np.array(faces)
    probs = np.array(probs)
    
    if duplicate:
        faces_sort = np.sort(faces,axis = 1)
        f_nums = faces.shape[0]
        flag = [True]*f_nums
        face_vertex = {}
        for i in range(f_nums):
            key = str(faces_sort[i,0])+','+str(faces_sort[i,1])+','+str(faces_sort[i,2])+','+str(faces_sort[i,3])
            if key in face_vertex:
                flag[i] = False
                for j in range(i):
                    if sum( faces_sort[i,:]== faces_sort[j,:])==4:
                        probs[j]=(probs[i]+face_vertex[key]*probs[j])/(face_vertex[key]+1)
                        break
                face_vertex[key] += 1
            else:
                face_vertex.update({key: 1}) 
            if flag[i] == True:
                continue  
        new_faces = faces[flag,:]
        new_probs = probs[flag]
        return new_faces,new_probs
    else:
        return faces,probs

def pred2faces(predicted, cand_faces, duplicate=True):
    faces = []
    for i, data in enumerate(predicted):   # labels or predicted
        if data == 1:
            faces.append(cand_faces[i,:])
    faces = np.array(faces)
    if duplicate:
        faces_sort = np.sort(faces,axis = 1)
        f_nums = faces.shape[0]
        flag = [True]*f_nums
        face_vertex = {}
        for i in range(f_nums):
            key = str(faces_sort[i,0])+','+str(faces_sort[i,1])+','+str(faces_sort[i,2])+','+str(faces_sort[i,3])
            if key in face_vertex:
                face_vertex[key] += 1
                flag[i] = False
            else:
                face_vertex.update({key: 1}) 
            '''
            if flag[i] == False:
                continue
            for j in range(i+1,f_nums):
                if flag[j] == False:
                    continue
                if sum( faces_sort[i,:]== faces_sort[j,:])==4:
                    flag[j] = False
            '''
        new_faces = faces[flag,:]
        return new_faces
    else:
        return faces

