import os
import torch
import numpy as np

# === Argument management helpers

def set_args_defaults(args):

    # Manage cuda config
    if (not args.disable_cuda and not torch.cuda.is_available()):
        print("!!! WARNING: CUDA requested but not available!")

    if (not args.disable_cuda and torch.cuda.is_available()):
        args.device = torch.device('cuda:0')
        args.dtype = torch.float32
        torch.set_default_dtype(args.dtype)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("CUDA enabled :)")
    else:
        args.device = torch.device('cpu')
        args.dtype = torch.float32
        torch.set_default_dtype(args.dtype)
        torch.set_default_tensor_type(torch.FloatTensor)
        print("CUDA disabled :(")


# === Misc value conversion

# Really, definitely convert a torch tensor to a numpy array
def toNP(x):
    return x.detach().to(torch.device('cpu')).numpy()

class fake_context():
    def __enter__(self):
        return None
    def __exit__(self, _1, _2, _3):
        return False


# === File helpers
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


# === Geometric helpers in pytorch


# Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
def norm(x, highdim=False):

    if(len(x.shape) == 1):
        raise ValueError("called norm() on single vector of dim " + str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called norm() with large last dimension " + str(x.shape) + " are you sure?")

    return torch.norm(x, dim=len(x.shape)-1)


def norm2(x, highdim=False):

    if(len(x.shape) == 1):
        raise ValueError("called norm() on single vector of dim " + str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called norm() with large last dimension " + str(x.shape) + " are you sure?")

    return dot(x, x)

# Computes normalizes array of vectors along last dimension
def normalize(x, divide_eps=1e-6, highdim=False):
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " + str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " + str(x.shape) + " are you sure?")

    return x / (norm(x, highdim=highdim)+divide_eps).unsqueeze(-1)

def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def face_barycenters(verts, faces):
    coords = face_coords(verts, faces)
    bary = torch.mean(coords, dim=-2)
    return bary

def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)

def dot(vec_A, vec_B):
    return torch.sum(vec_A*vec_B, dim=-1)

# Given (..., 3) vectors and normals, projects out any components of vecs which lies in the direction of normals. Normals are assumed to be unit.
def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def normalize_np(x, divide_eps=1e-6):
    return x / (np.linalg.norm(x, axis=-1)+divide_eps)[:,np.newaxis]

def not_quadrilateral_coplane_np(verts, faces, eps=5e-4):
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]
    
    normal = normalize_np(np.cross(vec_1, vec_2, axis=-1))
    vec_3 = normalize_np(coords[:, 3, :] - coords[:, 0, :])
    
    out = np.sum(normal*vec_3, axis=-1)
    res = np.absolute(out)
    
    #print('res = ',res)
    
    return res>eps


def not_quadrilateral_coplane(verts, faces, eps=5e-1):
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]
    normal = normalize(cross(vec_1, vec_2))

    vec_3 = normalize(coords[:, 3, :] - coords[:, 0, :])

    res = torch.abs(dot(normal, vec_3))
        
    return res>eps


def face_normals(verts, faces):
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]

    normal1 = normalize(cross(vec_1, vec_2))

    vec_3 = coords[:, 3, :] - coords[:, 0, :]
    vec_4 = coords[:, 2, :] - coords[:, 0, :]

    normal2 = normalize(cross(vec_4, vec_3))
    
    ang = dot(normal1, normal2)
    if ang>0:
        return 0.5 * (normal1+normal2)
    else:
        return 0.5 * (normal1-normal2)

    

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_1, vec_2)

    vec_3 = coords[:, 3, :] - coords[:, 0, :]
    vec_4 = coords[:, 2, :] - coords[:, 0, :]

    raw_normal2 = cross(vec_4, vec_3)

    return 0.5 * norm(raw_normal)+ 0.5 * norm(raw_normal2)
