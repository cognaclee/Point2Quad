### some basic info for quad mesh 
import openmesh as om
import numpy as np
from glob import glob
import os

class QuadMeshMetric:
    def __init__(self,mesh=None):
        if mesh is None:
            self.mesh = None
            self.points = None
            self.fv = None
        else:
            self.mesh = mesh
            self.points = mesh.points()
            self.fv = mesh.fv_indices()
            ## calculate basic infomations
            self.get_basic_info()
            self.get_principal_axes()
            self.get_normal()
            self.get_vertex_singal_area()
            self.calculate_face_area()
            self.calculate_Frobenius()
            self.calcualte_angle()
            #self.area_metric()
    
    def reset_mesh(self,mesh):
        self.mesh = mesh
        self.points = mesh.points()
        self.fv = mesh.fv_indices()
        if self.fv.shape[1]<4:
            return True
        ## reset basic infomations
        self.get_basic_info()
        self.get_principal_axes()
        self.get_normal()
        self.get_vertex_singal_area()
        self.calculate_face_area()
        self.calculate_Frobenius()
        self.calcualte_angle()
        #self.area_metric()
        return False
        
    def get_basic_info(self):
        f_ps = self.points[self.fv]
        # edge_vec
        self.v_L0 = f_ps[:,1] - f_ps[:,0]
        self.v_L1 = f_ps[:,2] - f_ps[:,1]
        self.v_L2 = f_ps[:,3] - f_ps[:,2]
        self.v_L3 = f_ps[:,0] - f_ps[:,3]
        self.v_D0 = f_ps[:,2] - f_ps[:,0]
        self.v_D1 = f_ps[:,3] - f_ps[:,1]
        self.L0 = np.linalg.norm(self.v_L0,axis=1)
        self.L1 = np.linalg.norm(self.v_L1,axis=1)
        self.L2 = np.linalg.norm(self.v_L2,axis=1)
        self.L3 = np.linalg.norm(self.v_L3,axis=1)
        F_l = np.stack((self.L0,self.L1,self.L2,self.L3),axis=1)
        self.L_min = np.min(F_l,axis=1)
        self.L_max = np.max(F_l,axis=1)
        self.F_l= F_l
        # diagonal
        self.D0 = np.linalg.norm(self.v_D0,axis=1)
        self.D1 = np.linalg.norm(self.v_D1,axis=1)
        F_d = np.stack((self.D0,self.D1),axis=1)
        self.D_max = np.max(F_d,axis=1)
        self.D_min = np.min(F_d,axis=1)
    
    def get_principal_axes(self):
        self.v_X1 = self.v_L0 - self.v_L2
        self.v_X2 = self.v_L1 - self.v_L3
    
    def get_normal(self):
        self.v_N0 = np.cross(self.v_L3,self.v_L0)
        self.v_N1 = np.cross(self.v_L0,self.v_L1)
        self.v_N2 = np.cross(self.v_L1,self.v_L2)
        self.v_N3 = np.cross(self.v_L2,self.v_L3)
        self.v_Nc = np.cross(self.v_X1,self.v_X2)
        self.v_n0 = self.v_N0/np.linalg.norm(self.v_N0,axis=1,keepdims=True)
        self.v_n1 = self.v_N1/np.linalg.norm(self.v_N1,axis=1,keepdims=True)
        self.v_n2 = self.v_N2/np.linalg.norm(self.v_N2,axis=1,keepdims=True)
        self.v_n3 = self.v_N3/np.linalg.norm(self.v_N3,axis=1,keepdims=True)
        self.v_nc = self.v_Nc/np.linalg.norm(self.v_Nc,axis=1,keepdims=True)

    def get_vertex_singal_area(self):
        self.area0 = np.sum(self.v_N0*self.v_nc,axis=1)
        self.area1 = np.sum(self.v_N1*self.v_nc,axis=1)
        self.area2 = np.sum(self.v_N2*self.v_nc,axis=1)
        self.area3 = np.sum(self.v_N3*self.v_nc,axis=1)
        self.area_all = np.stack((self.area0,self.area1,self.area2,self.area3),axis=1)    
        self.narea0 = np.sum(self.v_n0*self.v_nc,axis=1)
        self.narea1 = np.sum(self.v_n1*self.v_nc,axis=1)
        self.narea2 = np.sum(self.v_n2*self.v_nc,axis=1)
        self.narea3 = np.sum(self.v_n3*self.v_nc,axis=1)   
        self.narea_all = np.stack((self.narea0,self.narea1,self.narea2,self.narea3),axis=1)  

    def calculate_face_area(self):
        self.area_abs = (np.abs(self.area0) + np.abs(self.area1) + np.abs(self.area2) + np.abs(self.area3))/4.0 

    def calculate_Frobenius(self):
        self.F301 = (self.L3**2 + self.L0**2 + self.L1**2) / (2*np.linalg.norm(self.v_N0,axis=1))
        self.F012 = (self.L0**2 + self.L1**2 + self.L2**2) / (2*np.linalg.norm(self.v_N1,axis=1))
        self.F123 = (self.L1**2 + self.L2**2 + self.L3**2) / (2*np.linalg.norm(self.v_N2,axis=1))
        self.F230 = (self.L2**2 + self.L3**2 + self.L0**2) / (2*np.linalg.norm(self.v_N3,axis=1))
        self.F_all = np.stack((self.F301,self.F012,self.F123,self.F230),axis=1)
        
    def calcualte_angle(self):
        mark = (self.area_all<0).astype(int)
        cos0 = -np.sum(self.v_L0*self.v_L1,axis=1)/(self.L0*self.L1)
        cos1 = -np.sum(self.v_L1*self.v_L2,axis=1)/(self.L1*self.L2)
        cos2 = -np.sum(self.v_L2*self.v_L3,axis=1)/(self.L2*self.L3)
        cos3 = -np.sum(self.v_L3*self.v_L0,axis=1)/(self.L3*self.L0)
        cos = np.stack((cos0,cos1,cos2,cos3),axis=1)
        self.theta_all = np.power(-1,mark)*np.arccos(cos)*180/np.pi + 360*mark
        
    def area_metric(self):
        q_area = (self.area0 + self.area1 + self.area2 + self.area3) / 4.0
        #print('q_area.shape=',q_area.shape)
        self.area = q_area
        return q_area
    
    def aspect_ratio_metric(self,area_type = "area"):
        if area_type == "area":
            q_aspect_ratio = self.L_max * (self.L0 + self.L1 + self.L2 + self.L3 ) / self.area
        if area_type == "area_abs":
            q_aspect_ratio = self.L_max * (self.L0 + self.L1 + self.L2 + self.L3 ) / self.area_abs
        return np.mean(q_aspect_ratio)
    
    def condition_metric(self):
        cond0 = (self.L0*self.L0 + self.L3*self.L3)/self.area0
        cond1 = (self.L1*self.L1 + self.L0*self.L0)/self.area1
        cond2 = (self.L2*self.L2 + self.L1*self.L1)/self.area2
        cond3 = (self.L3*self.L3 + self.L2*self.L2)/self.area3
        cond = np.stack((cond0,cond1,cond2,cond3),axis=1)
        q_cond = np.max(cond,axis=1)/2.0
        return np.mean(q_cond)
    
    def edge_ratio_metric(self):
        q_edge_ratio = self.L_max/(self.L_min+1e-9)
        return np.mean(q_edge_ratio)
    
    ## The ratio of long axis to short axis
    def Max_edge_ratio(self):
        x1 = np.linalg.norm(self.v_X1,axis=1)
        x2 = np.linalg.norm(self.v_X2,axis=1)
        x = np.stack((x1/x2,x2/x1),axis=1)
        q = np.max(x,axis=1)
        return np.mean(q)
    
    def Jacobian_metric(self):
        q_jacobian = np.min(self.area_all,axis=1)
        # q_jacobian = q_jacobian/self.area
        return np.mean(q_jacobian)
        
    def Max_Aspect_Frobenius(self):
        Maf = np.max(self.F_all,axis=1)
        return np.mean(Maf)
    def Mean_Aspect_Frobenius(self):
        Meanaf = np.sum(self.F_all,axis=1)/4.0
        return np.mean(Meanaf)
    
    
    def Max_angle(self):
        Max_a = np.max(self.theta_all,axis=1)
        return np.mean(Max_a)
    
    def Angle_Distortion(self):
        martic = self.theta_all - 90
        ad = np.sqrt(np.sum(np.nan_to_num((self.theta_all - 90)* (self.theta_all - 90),nan=0))/(4*len(self.fv)))
        return ad

    ## maximum deviation at the corners of the quadrilatera, the less is better, range [0,+inf)
    def Min_angle(self):
        Min_a = np.min(self.theta_all,axis=1)
        return np.mean(Min_a)
    
    def Oddy(self):
        oddy0 = ((self.L0**2-self.L1**2)**2+4*np.sum(self.v_L0*self.v_L1,axis=1)**2)/(2*np.linalg.norm(self.v_N1)**2)
        oddy1 = ((self.L1**2-self.L2**2)**2+4*np.sum(self.v_L1*self.v_L2,axis=1)**2)/(2*np.linalg.norm(self.v_N2)**2)
        oddy2 = ((self.L2**2-self.L3**2)**2+4*np.sum(self.v_L2*self.v_L3,axis=1)**2)/(2*np.linalg.norm(self.v_N3)**2)
        oddy3 = ((self.L3**2-self.L0**2)**2+4*np.sum(self.v_L3*self.v_L0,axis=1)**2)/(2*np.linalg.norm(self.v_N0)**2)
        oddy = np.stack((oddy0,oddy1,oddy2,oddy3),axis=1)
        q = np.max(oddy,axis=1)
        return np.mean(q)
    
    def Radius_ratio(self):
        h = np.stack((self.L_max,self.D_max),axis=1)
        h_max = np.max(h,axis=1)
        L_sum = np.sum(self.F_l*self.F_l,axis=1)
        min_A = np.min(np.abs(self.area_all)/2,axis=1)
        q = L_sum*h_max/min_A
        return np.mean(q)
    
    ##  square of the minimum of the ratio of quad area to the average quad area and its inverse.
    def relative_size_squared(self):
        mean_a = np.mean(self.area)
        rs=np.stack((self.area/mean_a,mean_a/self.area),axis=1)
        q = np.min(rs,axis=1)**2
        self.R = q 
        return np.mean(q)
    
    ## near 1 is better
    def shape(self):
        shape0 = self.area0/(self.L0**2+self.L1**2)
        shape1 = self.area1/(self.L1**2+self.L2**2)
        shape2 = self.area2/(self.L2**2+self.L3**2)
        shape3 = self.area3/(self.L3**2+self.L0**2)
        shape = np.stack((shape0,shape1,shape2,shape3),axis=1)
        q = 2*np.min(shape,axis=1)
        self.S = q
        return np.mean(q)
    
    def shape_size(self):
        q = self.R*self.S
        return np.mean(q)
    
    ## similar with the scaled Jacobian, near 1 is better
    def shear(self):
        shear0 = self.area0/(self.L0*self.L3)
        shear1 = self.area1/(self.L1*self.L0)
        shear2 = self.area2/(self.L2*self.L1)
        shear3 = self.area3/(self.L3*self.L2)
        shear = np.stack((shear0,shear1,shear2,shear3),axis=1)
        q = 2*np.min(shear,axis=1)
        self.H = q
        return np.mean(q)
    
    def shear_size(self):
        return np.mean(self.H*self.R)
    
    ##  measures the angle between the principal axes 
    def skew(self):
        x1 = self.v_X1/np.linalg.norm(self.v_X1,axis=1,keepdims=True)
        x2 = self.v_X2/np.linalg.norm(self.v_X2,axis=1,keepdims=True)
        q = np.abs(np.sum(x1*x2,axis=1))
        return np.mean(q)

    ##  legth ratio between mim sqrt(2)*edge and max diagonal , near 1 is better
    def stretch(self):
        q = np.sqrt(2)*np.min(self.F_l,axis=1)/self.D_max
        return np.mean(q)
    
    ##  legth ratio between edge and mim diagonal, near 1 is better
    def taper(self):
        v_X12 =  -self.v_L0-self.v_L2
        q = np.linalg.norm(v_X12,axis=1)/self.D_min
        return np.mean(q)
    
    def warpage(self):
        f1 = np.sum(self.v_n0*self.v_n2,axis=1)**3
        f2 = np.sum(self.v_n1*self.v_n3,axis=1)**3
        fn = np.stack((f1,f2),axis=1)
        q = 1-np.min(fn,axis=1)
        return np.mean(q)

    def scaled_jacobian(self):
        scaled0 = self.area0/(self.L0*self.L3)
        scaled1 = self.area1/(self.L1*self.L0)
        scaled2 = self.area2/(self.L2*self.L1)
        scaled3 = self.area3/(self.L3*self.L2)
        scaled = np.stack((scaled0,scaled1,scaled2,scaled3),axis=1)
        q = np.min(scaled,axis=1)
        q_new = np.nan_to_num(q,nan=0)
        return np.mean(q_new)
        
 

def counts_edge_coincidence(faces):

    edges_cnts = {}
    f_n = faces.shape[0]
    for i in range(f_n):
        quad = faces[i,:]
        id_sort = quad.tolist()
        id_sort.sort()
        #print(id_sort)
        key1 = str(id_sort[0])+','+str(id_sort[1])
        key2 = str(id_sort[1])+','+str(id_sort[2])
        key3 = str(id_sort[2])+','+str(id_sort[3])
        key4 = str(id_sort[0])+','+str(id_sort[3])
        #print(key)
        keys = [key1,key2,key3,key4]
        for key in keys:
            if key in edges_cnts:
                edges_cnts[key] += 1
            else:
                edges_cnts.update({key: 1})

    counts1 = 0.0
    counts2 = 0.0
    counts3OrMore = 0.0
    all_counts = len(edges_cnts)
    for key, value in edges_cnts.items():
            if value == 1:
                counts1 +=1
            elif value == 2:
                counts2 +=1
            else:
                counts3OrMore +=1
            
            
    return counts1,counts2,counts3OrMore,all_counts


def read_m(filename): 
    points = []
    faces = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n') 
            oneline = line.split(' ')
            if oneline[0]=='Vertex':
                points.append([float(oneline[2]),float(oneline[3]),float(oneline[4])])
            elif oneline[0]=='Face':
                faces.append([int(oneline[2]),int(oneline[3]),int(oneline[4]),int(oneline[5])])
    return points, faces

def write_obj(vs, faces, filename):
    with open(filename, 'w+') as f:
        for vi, v in enumerate(vs):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for face in faces:
            #f.write("f %d %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1, face[3] + 1))
            f.write("f %d %d %d %d\n" % (face[0], face[1], face[2], face[3]))
       
        
if __name__ == '__main__':

    input_dir='./predicted_obj/'
    Metric = QuadMeshMetric()
    
    watertightness = 0.0
    manifoldness = 0.0 
    all_edges = 0.0
    

    q_edge_ratio = 0.0
    q_jacobian = 0.0
    q_ad = 0.0
    q_max_a = 0.0
    q_min_a = 0.0
    q_max_edge_ratio = 0.0
    q_oddy = 0.0
    qrs = 0.0
    shape = 0.0
    shape_size = 0.0
    shear = 0.0
    shear_size = 0.0
    skew = 0.0
    stretch = 0.0
    taper = 0.0
    q_s_jacobian = 0.0

    start = len(input_dir)
    mesh_names = sorted(glob(input_dir+'/*.'+'obj'))
    mesh_nums = len(mesh_names)
    for meshName in mesh_names:
    #for i in range(100):
        #meshName = mesh_names[i]
        print(meshName)
        mesh = om.PolyMesh()
        mesh = om.read_polymesh(meshName)
        invalide = Metric.reset_mesh(mesh)
        
        if invalide:
            mesh_nums -= 1
            continue
        
        '''
        bqes = 0.0
        for qe in mesh.edges():
            all_edges +=1.0
            if mesh.is_boundary(qe):
                bqes +=1.0
        watertightness +=bqes
        '''
        counts1,counts2,counts3OrMore,all_counts = counts_edge_coincidence(mesh.fv_indices())
        manifoldness +=(counts1+counts2)
        watertightness +=counts2
        all_edges +=all_counts
        
        
            
        
        
        Metric.area_metric()
        
        q_edge_ratio += Metric.edge_ratio_metric()
        
        q_jacobian += Metric.Jacobian_metric()
        q_ad += Metric.Angle_Distortion()
        q_max_a += Metric.Max_angle()
        q_min_a += Metric.Min_angle()
        q_max_edge_ratio += Metric.Max_edge_ratio()
        q_oddy += Metric.Oddy()
        qrs += Metric.relative_size_squared()
        shape += Metric.shape()
        shape_size += Metric.shape_size()
        shear += Metric.shear()
        shear_size += Metric.shear_size()
        skew += Metric.skew()
        stretch += Metric.stretch()
        taper += Metric.taper()

        q_s_jacobian += Metric.scaled_jacobian()


    print('1. watertightness=',watertightness/all_edges)
    print('2. manifoldness=',manifoldness/all_edges)
    print('3. scale_jacobian=',q_s_jacobian/mesh_nums)
    print('4. max_min_edge_ratio=',q_edge_ratio/mesh_nums)
    
    print('****'*20)
    
    print('5. jacobian=',q_jacobian/mesh_nums)
    print('6. angle_distortion=',q_ad/mesh_nums)
    print('7. max_angle=',q_max_a/mesh_nums)
    print('8. min_angle=',q_min_a/mesh_nums)
    print('9. principal_axes_ratio=',q_max_edge_ratio/mesh_nums)
    print('10. deviation_with_square=',q_oddy/mesh_nums)
    print('11. relative_area_ratio=',qrs/mesh_nums)
    print('12. shape=',shape/mesh_nums)
    print('13. shape_size=',shape_size/mesh_nums)
    print('14. shear=',shear/mesh_nums)
    print('15. shear_size=',shear_size/mesh_nums)
    print('16. skew=',skew/mesh_nums)
    print('17. stretch=',stretch/mesh_nums)
    print('18. taper=',taper/mesh_nums)
    
    
    
