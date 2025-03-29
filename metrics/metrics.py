import numpy as np
import math
import torch
from pytorch3d.loss import chamfer_distance
from torch.distributions.categorical import Categorical


def watertightness_manifoldness(in_faces):
    n_f = in_faces.shape[0]
    edge_incident = {}
    for i in range(n_f):
        for j in range(3):
            first = min(in_faces[i][j],in_faces[i][(j+1)%3])
            secon = max(in_faces[i][j],in_faces[i][(j+1)%3])
            key = str(first)+','+str(secon)
            if edge_incident.has_key(key):
                edge_incident[key] += 1 
            else:
                edge_incident.update({key: 1}) 
    n_v = len(edge_incident)
    one_incident = 0
    two_incident = 0
    for key in edge_incident:
        if edge_incident[key] == 1:
            one_incident+=1
        if edge_incident[key] == 2:
            two_incident+=1
    wt_score = two_incident/n_v
    mf_score = (two_incident+one_incident)/n_v
    return wt_score,mf_score

def manifoldness(in_faces):
    pass

def EulerCharacteristic(in_faces):
    pass



def classified_metrics(predict, labels):
    pre_1 = predict.eq(1)
    pre_0 = predict.eq(0)
    lab_1 = labels.eq(1)
    lab_0 = labels.eq(0)
    TP = (lab_1 * pre_1).sum()
    FN = (lab_1 * pre_0).sum()
    FP = (lab_0 * pre_1).sum()
    TN = (lab_0 * pre_0).sum()
    # Display
    if ((TP + FN) == 0):
        recall = 0
    else:
        recall = TP/(TP+FN)
    if ((TP+FP) == 0):
        precision = 0
    else:
        precision = TP/(TP+FP)
    
    TP_plus_FP = TP + FP
    precision = TP / (TP_plus_FP + 1e-9)
    TP_plus_FN = TP + FN
    recall = TP / (TP_plus_FN + 1e-9)
    # Compute F1 score
    F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-9)
    
    # Compute Accuracy
    ACC = torch.sum(TP, dim=-1) / (TP_plus_FN + FP+ TN)
    return precision, recall, F1, ACC
   


def chamfer_distance(verts,pred_faces, gt_faces,n_sample_pts):
    pred_samples = sample_points_on_surface(verts, pred_faces, n_sample_pts)
    gt_samples = sample_points_on_surface(verts, gt_faces, n_sample_pts)
    cd_loss, _ = chamfer_distance(pred_samples, gt_samples)
    return cd_loss   
    

# Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
def norm(x, highdim=False):

    if(len(x.shape) == 1):
        raise ValueError("called norm() on single vector of dim " + str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called norm() with large last dimension " + str(x.shape) + " are you sure?")

    return torch.norm(x, dim=len(x.shape)-1)
    
def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)
    
def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)
    
def sample_points_on_surface(verts, faces, n_pts, return_inds_and_bary=False, face_probs=None):

    # Choose faces
    face_areas = face_area(verts, faces)
    if face_probs is None:
        # if no probs, just weight directly by areas to uniformly sample surface
        sample_probs = face_areas
        sample_probs = torch.clamp(sample_probs, 1e-30, float('inf')) # avoid -eps area
        face_distrib = Categorical(sample_probs)
    else:
        # if we have face probs, weight by those so we are more likely to sample more probable faces
        sample_probs = face_areas * face_probs
        sample_probs = torch.clamp(sample_probs, 1e-30, float('inf')) # avoid -eps area
        face_distrib = Categorical(sample_probs)

    face_inds = face_distrib.sample(sample_shape=(n_pts,))

    # Get barycoords for each sample
    r1_sqrt = torch.sqrt(torch.rand(n_pts, device=verts.device))
    r2 = torch.rand(n_pts, device=verts.device)
    bary_vals = torch.zeros((n_pts, 3), device=verts.device)
    bary_vals[:, 0] = 1. - r1_sqrt
    bary_vals[:, 1] = r1_sqrt * (1. - r2)
    bary_vals[:, 2] = r1_sqrt * r2

    # Get position in face for each sample
    coords = face_coords(verts, faces)
    sample_coords = coords[face_inds, :, :]
    sample_pos = torch.sum(bary_vals.unsqueeze(-1) * sample_coords, dim=1)

    if return_inds_and_bary:
        return sample_pos, face_inds, bary_vals
    else:
        return sample_pos