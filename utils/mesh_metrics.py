### some basic info for quad mesh 
import openmesh as om
import numpy as np
import polyscope as ps 

mesh_path = r"../pred_results/test_face_infos_29_w3.6_obj/pred/wrench50K-quad.obj"
mesh = om.PolyMesh()
mesh = om.read_polymesh(mesh_path)




class QuadMeshMetric:
    def __init__(self,mesh):
        self.mesh = mesh
        self.points = mesh.points()
        self.fv = mesh.fv_indices()
        
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
        self.area = q_area
        return q_area
    
    def aspect_ratio_metric(self,area_type = "area"):
        if area_type == "area":
            q_aspect_ratio = self.L_max * (self.L0 + self.L1 + self.L2 + self.L3 ) / self.area
        if area_type == "area_abs":
            q_aspect_ratio = self.L_max * (self.L0 + self.L1 + self.L2 + self.L3 ) / self.area_abs
        return q_aspect_ratio
    
    def condition_metric(self):
        cond0 = (self.L0*self.L0 + self.L3*self.L3)/self.area0
        cond1 = (self.L1*self.L1 + self.L0*self.L0)/self.area1
        cond2 = (self.L2*self.L2 + self.L1*self.L1)/self.area2
        cond3 = (self.L3*self.L3 + self.L2*self.L2)/self.area3
        cond = np.stack((cond0,cond1,cond2,cond3),axis=1)
        q_cond = np.max(cond,axis=1)/2.0
        return q_cond
    
    def edge_ratio_metric(self):
        q_edge_ratio = self.L_max/self.L_min
        return q_edge_ratio
    
    def Max_edge_ratio(self):
        x1 = np.linalg.norm(self.v_X1,axis=1)
        x2 = np.linalg.norm(self.v_X2,axis=1)
        x = np.stack((x1/x2,x2/x1),axis=1)
        q = np.max(x,axis=1)
        return q
    
    def Jacobian_metric(self):
        q_jacobian = np.min(self.area_all,axis=1)
        # q_jacobian = q_jacobian/self.area
        return q_jacobian
        
    def Max_Aspect_Frobenius(self):
        Maf = np.max(self.F_all,axis=1)
        return Maf
    def Mean_Aspect_Frobenius(self):
        Meanaf = np.sum(self.F_all,axis=1)/4.0
        return Meanaf
    
    
    def Max_angle(self):
        Max_a = np.max(self.theta_all,axis=1)
        return Max_a
    
    def Angle_Distortion(self):
        martic = self.theta_all - 90
        ad = np.sqrt(np.sum(np.nan_to_num((self.theta_all - 90)* (self.theta_all - 90),nan=0))/(4*len(self.fv)))
        return ad

    def Min_angle(self):
        Min_a = np.min(self.theta_all,axis=1)
        return Min_a
    
    def Oddy(self):
        oddy0 = ((self.L0**2-self.L1**2)**2+4*np.sum(self.v_L0*self.v_L1,axis=1)**2)/(2*np.linalg.norm(self.v_N1)**2)
        oddy1 = ((self.L1**2-self.L2**2)**2+4*np.sum(self.v_L1*self.v_L2,axis=1)**2)/(2*np.linalg.norm(self.v_N2)**2)
        oddy2 = ((self.L2**2-self.L3**2)**2+4*np.sum(self.v_L2*self.v_L3,axis=1)**2)/(2*np.linalg.norm(self.v_N3)**2)
        oddy3 = ((self.L3**2-self.L0**2)**2+4*np.sum(self.v_L3*self.v_L0,axis=1)**2)/(2*np.linalg.norm(self.v_N0)**2)
        oddy = np.stack((oddy0,oddy1,oddy2,oddy3),axis=1)
        q = np.max(oddy,axis=1)
        return q
    
    def Radius_ratio(self):
        h = np.stack((self.L_max,self.D_max),axis=1)
        h_max = np.max(h,axis=1)
        L_sum = np.sum(self.F_l*self.F_l,axis=1)
        min_A = np.min(np.abs(self.area_all)/2,axis=1)
        q = L_sum*h_max/min_A
        return q
    
    def relative_size_squared(self):
        mean_a = np.mean(self.area)
        rs=np.stack((self.area/mean_a,mean_a/self.area),axis=1)
        q = np.min(rs,axis=1)**2
        self.R = q 
        return q
    
    def shape(self):
        shape0 = self.area0/(self.L0**2+self.L1**2)
        shape1 = self.area1/(self.L1**2+self.L2**2)
        shape2 = self.area2/(self.L2**2+self.L3**2)
        shape3 = self.area3/(self.L3**2+self.L0**2)
        shape = np.stack((shape0,shape1,shape2,shape3),axis=1)
        q = 2*np.min(shape,axis=1)
        self.S = q
        return q
    
    def shape_size(self):
        q = self.R*self.S
        return q
    
    def shear(self):
        shear0 = self.area0/(self.L0*self.L3)
        shear1 = self.area1/(self.L1*self.L0)
        shear2 = self.area2/(self.L2*self.L1)
        shear3 = self.area3/(self.L3*self.L2)
        shear = np.stack((shear0,shear1,shear2,shear3),axis=1)
        q = 2*np.min(shear,axis=1)
        self.H = q
        return q
    
    def shear_size(self):
        return self.H*self.R
    
    def skew(self):
        x1 = self.v_X1/np.linalg.norm(self.v_X1,axis=1,keepdims=True)
        x2 = self.v_X2/np.linalg.norm(self.v_X2,axis=1,keepdims=True)
        q = np.abs(np.sum(x1*x2,axis=1))
        return q

    def stretch(self):
        q = np.sqrt(2)*np.min(self.F_l,axis=1)/self.D_max
        return q
    
    def taper(self):
        v_X12 =  -self.v_L0-self.v_L2
        q = np.linalg.norm(v_X12,axis=1)/self.D_min
        return q
    
    def warpage(self):
        f1 = np.sum(self.v_n0*self.v_n2,axis=1)**3
        f2 = np.sum(self.v_n1*self.v_n3,axis=1)**3
        fn = np.stack((f1,f2),axis=1)
        q = 1-np.min(fn,axis=1)
        return q

    def scaled_jacobian(self):
        scaled0 = self.area0/(self.L0*self.L3)
        scaled1 = self.area1/(self.L1*self.L0)
        scaled2 = self.area2/(self.L2*self.L1)
        scaled3 = self.area3/(self.L3*self.L2)
        scaled = np.stack((scaled0,scaled1,scaled2,scaled3),axis=1)
        q = np.min(scaled,axis=1)
        q_new = np.nan_to_num(q,nan=0)
        
        print("avg is:",q_new.mean())
        return q
        
Metric = QuadMeshMetric(mesh)
Metric.get_basic_info()
Metric.get_principal_axes()
Metric.get_normal()
Metric.get_vertex_singal_area()
Metric.calculate_face_area()
Metric.calculate_Frobenius()
Metric.calcualte_angle()

q_area = Metric.area_metric()
q_aspect_ratio = Metric.aspect_ratio_metric()
q_cond = Metric.condition_metric()
q_edge_ratio = Metric.edge_ratio_metric()
q_jacobian = Metric.Jacobian_metric()
q_ad = Metric.Angle_Distortion()
q_maf = Metric.Max_Aspect_Frobenius()
q_meanaf =Metric.Mean_Aspect_Frobenius()
q_max_a = Metric.Max_angle()
q_min_a = Metric.Min_angle()
q_max_edge_ratio = Metric.Max_edge_ratio()
q_oddy = Metric.Oddy()
q_radius_ratio = Metric.Radius_ratio()
qrs = Metric.relative_size_squared()
shape = Metric.shape()
shape_size = Metric.shape_size()
shear = Metric.shear()
shear_size = Metric.shear_size()
skew = Metric.skew()
stretch = Metric.stretch()
taper = Metric.taper()
warpage = Metric.warpage()

q_s_jacobian = Metric.scaled_jacobian()




ps.init()
ps.register_surface_mesh("mesh",mesh.points(),mesh.fv_indices())
ps.get_surface_mesh("mesh").add_scalar_quantity("area",q_area,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("cond",q_cond,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("aspect_ratio",q_aspect_ratio,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("edge_ratio",q_edge_ratio,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("jacobian",q_jacobian,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("scaled_jacobian",q_s_jacobian,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Maf",q_maf,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Meanaf",q_meanaf,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Max_a",q_max_a,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Min_a",q_min_a,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Min_edge_ratio",q_max_edge_ratio,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Oddy",q_oddy,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("Rasius_ratio",q_radius_ratio,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("relative_size_squared",qrs,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("shape",shape,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("shape_size",shape_size,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("shear",shear,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("shear_size",shear_size,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("skew",skew,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("stretch",stretch,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("taper",taper,defined_on="faces")
ps.get_surface_mesh("mesh").add_scalar_quantity("warpage",warpage,defined_on="faces")
ps.show()
print(q_ad)