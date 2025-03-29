import torch
from utils import *


def fill_holes_greedy(in_faces):
    faces = toNP(in_faces).tolist()

    def edge_key(a,b):
        return (min(a,b), max(a,b))
    def face_key(f):
        return tuple(sorted(f))

    edge_count = {}
    neighbors = {}
    all_faces = set()
    def add_edge(a,b):
        if a not in neighbors:
            neighbors[a] = set()
        if b not in neighbors:
            neighbors[b] = set()
        key = edge_key(a,b)
        if key not in edge_count:
            edge_count[key] = 0

        neighbors[a].add(b)
        neighbors[b].add(a)
        edge_count[key] += 1

    def add_face(f):
        for i in range(3):
            a = f[i]
            b = f[(i+1)%3]
            add_edge(a,b)

        all_faces.add(face_key(f)) 

    def face_exists(f):
        return face_key(f) in all_faces

    for f in faces:
        add_face(f)

    # repeated passes (inefficient)
    any_changed = True
    while(any_changed):
        any_changed = False
        new_faces = []

        start_edges = [e for e in edge_count]

        for e in start_edges:
            if edge_count[e] == 1:
                a,b = e
                found = False
                
                # Look single triangle holes
                for s in [a,b]: # one of the verts in this edge
                    if found: break # quit once found
                    o = b if s == a else a # the other vert in this edge
                    for n in neighbors[s]: # a candidate third vertex
                        if found: break # quit once found
                        if n == o: continue # must not be same as edge
                        if face_exists([a,b,n]): continue # face must not exist
                        if (edge_count[edge_key(s,n)] == 1) and (edge_key(o,n) in edge_count) and (edge_count[edge_key(o,n)] == 1):  # must be single hole

                            # accept the new face
                            found = True
                            new_f = [a,b,n]
                            new_faces.append(new_f)
                            add_face(new_f)

        if any_changed:
            # if we found a single hole, look for more
            continue
        
        for e in start_edges:
            if edge_count[e] == 1:
                a,b = e
                found = False

                # Look for matching edge
                for s in [a,b]: # one of the verts in this edge
                    if found: break # quit once found
                    o = b if s == a else a # the other vert in this edge
                    for n in neighbors[s]: # a candidate third vertex
                        if found: break # quit once found
                        if n == o: continue # must not be same as edge
                        if face_exists([a,b,n]): continue # face must not exist
                        if edge_count[edge_key(s,n)] == 1:  # must be boundary edge

                            # accept the new face
                            found = True
                            new_f = [a,b,n]
                            new_faces.append(new_f)
                            add_face(new_f)



        faces.extend(new_faces)

    return torch.tensor(faces, dtype=in_faces.dtype, device=in_faces.device)


        
        
