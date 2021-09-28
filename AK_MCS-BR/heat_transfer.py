# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:41:40 2018

@author: mmenz
"""
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spalinalg
from time import time
import pandas as pd

class Heat_transfer():
    def __init__(self,mesh_file,form):
        self.mesh_file = mesh_file
        self.format=form
        self.nodes = None
        self.gamma_1=None
        self.gamma_2=None
        self.gamma_3=None
        self.Surf1=None
        self.Surf2=None
        self.Q4_elem = None
#        self.cond_gamma_1 = cond_gamma_1  # vecteur [h, T_fl]
#        self.cond_gamma_2 = cond_gamma_2
#        self.cond_gamma_3 = cond_gamma_3
        self.A_1 = None
        self.A_2 = None
        self.A_3 = None
        self.A_4 = None
        self.A_5 = None
        self.b_1 = None
        self.b_2 = None    
        self.b_3 = None  
        self.data_5 = None
        self.row_5 = None
        self.col_5 = None
        self.data_4 = None
        self.row_4 = None
        self.col_4 = None
        self.data_2 = None
        self.row_2 = None
        self.col_2 = None
        self.data_1 = None
        self.row_1 = None
        self.col_1 = None   
        self.data_3 = None
        self.row_3= None
        self.col_3 = None
        
    def read_inp_mesh(self):
        
        msh_input_file = self.mesh_file
        f=open(msh_input_file,'r')
        ind_start_node = '*Node\n'
        ind_stop_node ='*Element'
        line=f.readline() 
        
        while (line == ind_start_node) == False : 
            line=f.readline() 
         
        line=f.readline()      
        line_split=line.split(',')
        nodes=np.array([float(line_split[1]),float(line_split[2])]).reshape(1,2)
        line=f.readline() 
        line_split=line.split(',')
        while (line_split[0] == ind_stop_node) == False : 
            nodes=np.append(nodes,np.array([float(line_split[1]),float(line_split[2])]).reshape(1,2),axis=0)
            line=f.readline() 
            line_split=line.split(',')
        
        #read elements Q4
        ind_stop_element = '*Nset'
        line=f.readline()      
        line_split=line.split(',')
        elem=np.array([float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]).reshape(1,4)
        line=f.readline() 
        line_split=line.split(',')
        while (line_split[0] == ind_stop_element) == False : 
            elem=np.append(elem,np.array([float(line_split[1]),float(line_split[2]),float(line_split[3]),float(line_split[4])]).reshape(1,4),axis=0)
            line=f.readline() 
            line_split=line.split(',')

        #Elsets surfaces
        Surf1=list()
        Surf2=list()
        ind_start_surf='*Elset'
        while (line_split[0] == ind_start_surf) == False : 
            line=f.readline()
            line_split=line.split(',')
        line=f.readline()
        line_split=line.split(',')
        Surf1=Surf1 + line_split
        while (line_split[0] == ind_start_surf) == False : 
            line=f.readline()
            line_split=line.split(',')
        line=f.readline()  
        line_split=line.split(',')
        Surf2=Surf2 + line_split
            
        #Nodesets pour convection gamma1=hot, gamma2=cool, gamma3=out
        ind_start_gamma1='*Nset, nset=gamma1\n'
        ind_start_gamma2='*Nset, nset=gamma2\n'
        ind_start_gamma3='*Nset, nset=gamma3\n'
        ind_stop_gamma='**'
        #pour abaqus_fine
        ind_start_gamma1='*Nset, nset=gamma1, instance=PART-1-1\n'
        ind_start_gamma2='*Nset, nset=gamma2, instance=PART-1-1\n'
        ind_start_gamma3='*Nset, nset=gamma3, instance=PART-1-1\n'
        ind_stop_gamma='*Elset'
        
        
        gamma_1=list()
        gamma_2=list()
        gamma_3=list()
        while (line == ind_start_gamma1) == False : 
            line=f.readline()
        line=f.readline()    
        while (line==ind_start_gamma2)==False:
            line_split=line.split(',')
            gamma_1=gamma_1 + line_split
            line=f.readline()
        line=f.readline()    
        while (line==ind_start_gamma3)==False:
            line_split=line.split(',')
            gamma_2=gamma_2 + line_split
            line=f.readline()
            
        line=f.readline()    
        line_split=line.split(',')
        while (line_split[0]==ind_stop_gamma)==False:
            gamma_3=gamma_3 + line_split
            line=f.readline()
            line_split=line.split(',')
            
                    

        self.nodes = nodes
        self.Q4_elem = elem
        self.gamma_1= [float(i) for i in gamma_1] 
        self.gamma_2=[float(i) for i in gamma_2] 
        self.gamma_3=[float(i) for i in gamma_3] 
        self.Surf1=[float(i) for i in Surf1] 
        self.Surf2=[float(i) for i in Surf2]
        print "Nombre de noeuds %d, Nombre d'elements Q4 %d"%(np.shape(nodes)[0],np.shape(elem)[0])
      #  self.define_cl()
        return nodes, elem, gamma_1,gamma_2,gamma_3, Surf1, Surf2
    
    
    def read_msh(self):
        
        msh_input_file = self.mesh_file
        f = pd.read_csv(msh_input_file)
        nb_nodes=int(f.loc[3,'$MeshFormat'].split(' ')[0])
        nodes=np.zeros((nb_nodes,2))
        
        nodesData = f.loc[ 4:3+nb_nodes,'$MeshFormat' ]
        i=0
        for row in nodesData: 
            line_split=row.split(' ') 
            nodes[i,:]=np.array([float(line_split[1]),float(line_split[2])])
            i=i+1
        
        
        nb_elem_tot=int(f.loc[nb_nodes+6,'$MeshFormat'].split(' ')[0])
        start_elem=nb_nodes+7
        #read elements Q4
        gamma_1=list()
        gamma_2=list()
        gamma_3=list()
        Surf1=list()
        Surf2=list()
        
        elemData = f.loc[start_elem:start_elem+nb_elem_tot-1,'$MeshFormat' ]
        
        ind_stop_1d='3'
        ind_stop_Surf1='5'

        i=0
        for row in elemData: 
            line_split=row.split(' ')  
            i=i+1
            if (line_split[1] == ind_stop_1d) == False : 
                if (line_split[3] == '1') :
                    gamma_1.append(line_split[5])
                    gamma_1.append(line_split[6])
                elif (line_split[3] == '2') :
                    gamma_2.append(line_split[5])
                    gamma_2.append(line_split[6])
                elif (line_split[3] == '3') :
                    gamma_3.append(line_split[5])
                    gamma_3.append(line_split[6])
            else : 
                break 
            
        elemData = f.loc[start_elem+i-1:start_elem+nb_elem_tot-1,'$MeshFormat' ] 
        nb_elem=nb_elem_tot-i+1
        elem=np.zeros((nb_elem,4)) 
        i=0
        nb_surf2=0
        ind_stop_Surf='4'
        for row in elemData: 
             line_split=row.split(' ')  
             elem[i,:]=np.array([float(line_split[5]),float(line_split[6]),float(line_split[7]),float(line_split[8])])
             i=i+1
             if line_split[3] == ind_stop_Surf:
                  nb_surf2=nb_surf2+1
                  
        Surf2=['1']+[nb_surf2]
        Surf1=[nb_surf2+1]+[nb_elem]
        gamma_1= [float(i) for i in gamma_1] 
        gamma_2=[float(i) for i in gamma_2] 
        gamma_3=[float(i) for i in gamma_3] 
        gamma_1=list(set(gamma_1))
        gamma_2=list(set(gamma_2))
        gamma_3=list(set(gamma_3))


        self.nodes = nodes
        self.Q4_elem = elem
  
        
    
        self.gamma_1= gamma_1
        self.gamma_2= gamma_2
        self.gamma_3=gamma_3
        
        
        self.Surf1=[float(i) for i in Surf1] 
        self.Surf2=[float(i) for i in Surf2]
        print "Nombre de noeuds %d, Nombre d'elements Q4 %d"%(np.shape(nodes)[0],np.shape(elem)[0])
      #  self.define_cl()
        return self.nodes, self.Q4_elem, self.gamma_1,self.gamma_2,self.gamma_3, self.Surf1, self.Surf2
    
    
    def read_msh_0(self):
        
        msh_input_file = self.mesh_file
        f=open(msh_input_file,'r')
        ind_start_node = '$Nodes\n'
        ind_stop_node ='$EndNodes\n'
        line=f.readline() 
    
        while (line == ind_start_node) == False : 
            line=f.readline() 
         
        line=f.readline() 
        line=f.readline()
        line_split=line.split(' ')
        nodes=np.array([float(line_split[1]),float(line_split[2])]).reshape(1,2)
        line=f.readline() 
        line_split=line.split(' ')
        while (line_split[0] == ind_stop_node) == False : 
            nodes=np.append(nodes,np.array([float(line_split[1]),float(line_split[2])]).reshape(1,2),axis=0)
            line=f.readline() 
            line_split=line.split(' ')
        
        line=f.readline() #$Elems
        line=f.readline() #nb elems 
        line_split=line.split(' ')
        nb_elem=line_split[0]
        #read elements Q4
        gamma_1=list()
        gamma_2=list()
        gamma_3=list()
        Surf1=list()
        Surf2=list()
        ind_stop_element = '$EndElements\n'
        ind_stop_1d='3'
        ind_stop_Surf1='5'
        line=f.readline()      
        line_split=line.split(' ')

        
        while (line_split[1] == ind_stop_1d) == False : 
            if (line_split[3] == '1') :
                gamma_1.append(line_split[5])
                gamma_1.append(line_split[6])
            elif (line_split[3] == '2') :
                gamma_2.append(line_split[5])
                gamma_2.append(line_split[6])
            elif (line_split[3] == '3') :
                gamma_3.append(line_split[5])
                gamma_3.append(line_split[6])
            line=f.readline() 
            line_split=line.split(' ')
        nb_1d=int(line_split[0])-1
        
        elem=np.array([float(line_split[5]),float(line_split[6]),float(line_split[7]),float(line_split[8])]).reshape(1,4)
        Surf2=['1']
        line=f.readline() 
        line_split=line.split(' ')
        
        while (line_split[3] == ind_stop_Surf1) == False : 
            elem=np.append(elem,np.array([float(line_split[5]),float(line_split[6]),float(line_split[7]),float(line_split[8])]).reshape(1,4),axis=0)
            line=f.readline() 
            line_split=line.split(' ')
        Surf2=Surf2+[float(line_split[0])-nb_1d-1]
        Surf1=[float(line_split[0])-nb_1d]+[int(nb_elem)-nb_1d]
        while (line_split[0] == ind_stop_element) == False : 
            elem=np.append(elem,np.array([float(line_split[5]),float(line_split[6]),float(line_split[7]),float(line_split[8])]).reshape(1,4),axis=0)
            line=f.readline() 
            line_split=line.split(' ')
        
       
        gamma_1= [float(i) for i in gamma_1] 
        gamma_2=[float(i) for i in gamma_2] 
        gamma_3=[float(i) for i in gamma_3] 
        gamma_1=list(set(gamma_1))
        gamma_2=list(set(gamma_2))
        gamma_3=list(set(gamma_3))


        self.nodes = nodes
        self.Q4_elem = elem
  
        
    
        self.gamma_1= gamma_1
        self.gamma_2= gamma_2
        self.gamma_3=gamma_3
        
        
        self.Surf1=[float(i) for i in Surf1] 
        self.Surf2=[float(i) for i in Surf2]
        print "Nombre de noeuds %d, Nombre d'elements Q4 %d"%(np.shape(nodes)[0],np.shape(elem)[0])
      #  self.define_cl()
        return self.nodes, self.Q4_elem, self.gamma_1,self.gamma_2,self.gamma_3, self.Surf1, self.Surf2
    
    
    def shape_function_Q4(self,xi,eta):
        #element Q4
        # fonctions de forme
        shape=np.array([(1-xi)*(1-eta)/4,(1+xi)*(1-eta)/4,(1+xi)*(1+eta)/4,(1-xi)*(1+eta)/4])
        naturalDerivatives= np.array([[-(1-eta)/4, -(1-xi)/4],[(1-eta)/4, -(1+xi)/4],[(1+eta)/4, (1+xi)/4],[-(1+eta)/4, (1-xi)/4]])
        return shape, naturalDerivatives
            
    def Jacobian(self,nodeCoordinates,naturalDerivatives):
#    
#        % JacobianMatrix    : Jacobian matrix
#        % invJacobian : inverse of Jacobian Matrix
#        % XYDerivatives  : derivatives w.r.t. x and y
#        % naturalDerivatives  : derivatives w.r.t. xi and eta
#        % nodeCoordinates  : nodal coordinates at element level
#       
        JacobianMatrix=np.dot(np.transpose(naturalDerivatives),nodeCoordinates)             
        invJacobian=np.linalg.inv(JacobianMatrix)
        XYDerivatives=np.dot(invJacobian,np.transpose(naturalDerivatives))
        return JacobianMatrix,invJacobian,XYDerivatives
    
            
    def compute_A_elem_1(self,ind_elem,nodeCoordinates):
        A_elem_1 = np.zeros((4, 4))
        locations=np.array([[ -0.577350269189626 ,-0.577350269189626],[0.577350269189626, -0.577350269189626],[ 0.577350269189626 , 0.577350269189626],[-0.577350269189626 , 0.577350269189626]])
        for q in np.arange(4): # pour chaque noeud de l'elem
            A_q = np.zeros((4, 4)) 
            xi, eta = locations[q,:] 
            shape, naturalDerivatives = self.shape_function_Q4(xi,eta)    
            JacobianMatrix,invJacobian,XYDerivatives=self.Jacobian(nodeCoordinates,naturalDerivatives)
            A_q=A_q+np.linalg.det(JacobianMatrix)*np.transpose(XYDerivatives).dot(XYDerivatives)
            A_elem_1 = A_elem_1 + A_q
        return A_elem_1


    def compute_A_b_elem_convection(self,nodeCoordinates,nodes_gamma):
        A_elem = np.zeros((4, 4))
        b_elem=np.zeros((4,1))
        det_J=0.5*np.linalg.norm(nodeCoordinates[nodes_gamma[0]]-nodeCoordinates[nodes_gamma[1]],2)
        for i in nodes_gamma:
           A_elem[i,i]=A_elem[i,i]+det_J*1/3
           b_elem[i,0]=det_J
           for j in nodes_gamma:
               A_elem[i,j]=A_elem[i,j]+det_J*1/3
        
        return A_elem, b_elem
            
    
    
    def assembly_A_1(self) :       
        nodes, elem = self.nodes, self.Q4_elem 
        n_nodes = nodes.shape[0]
        if self.format == 'full':
            A_1 = np.zeros((n_nodes,n_nodes))
            A_2 = np.zeros((n_nodes,n_nodes))
        if self.format == 'sparse':
            col = []
            row = []
            data = []
            col2 = []
            row2 = []
            data2 = []


        
        n_elem = np.shape(elem)[0]
        for ind_elem in range(n_elem):
            Id = map(int,elem[ind_elem])      
            S1 = nodes[Id[0]-1,:].reshape(1,2)
            S2 = nodes[Id[1]-1,:].reshape(1,2)
            S3 = nodes[Id[2]-1,:].reshape(1,2)
            S4 = nodes[Id[3]-1,:].reshape(1,2)
            nodeCoordinates=np.concatenate((S1,S2,S3,S4),axis=0)
            A_elem = self.compute_A_elem_1(ind_elem,nodeCoordinates)
            ind_min1, ind_max1=self.Surf1
            ind_min2, ind_max2=self.Surf2
            if ind_elem in np.arange(ind_min1-1, ind_max1) : 
                for i in range(4):
                    ind_i=Id[i]-1
                    for j in range(4):
                        ind_j=Id[j]-1
                        if self.format == 'full':
                            A_1[ind_i,ind_j] = A_1[ind_i,ind_j] + A_elem[i,j]
                        if self.format == 'sparse':
                            row.extend([ind_i])
                            col.extend([ind_j])
                            data.extend([A_elem[i,j]])
                        
            elif ind_elem in np.arange(ind_min2-1, ind_max2) :  
                for i in range(4):
                    ind_i=Id[i]-1
                    for j in range(4):
                        ind_j=Id[j]-1
                        if self.format == 'full':
                            A_2[ind_i,ind_j] = A_2[ind_i,ind_j] + A_elem[i,j]
                        if self.format == 'sparse':
                            row2.extend([ind_i])
                            col2.extend([ind_j])
                            data2.extend([A_elem[i,j]])
                                    
                            
                            
        if self.format == 'full':
            self.A_4=A_1
            self.A_5=A_2
        
        elif self.format == 'sparse':
            data = np.array(data)
            row = np.array(row)
            col = np.array(col)
            data2 = np.array(data2)
            row2= np.array(row2)
            col2 = np.array(col2)
            self.data_5 = data2
            self.row_5 = row2
            self.col_5 = col2
            self.data_4 = data
            self.row_4 = row
            self.col_4 = col
            A_4 = spa.coo_matrix((self.data_4, (self.row_4, self.col_4)), shape=(len(self.nodes), len(self.nodes)))
            A_5 = spa.coo_matrix((self.data_5, (self.row_5, self.col_5)), shape=(len(self.nodes), len(self.nodes)))
            self.A_4 = A_4.tocsc()
            self.A_5 = A_5.tocsc() 
    
    def assembly_convection(self) : 
        nodes, elem = self.nodes, self.Q4_elem 
        n_nodes = nodes.shape[0]
        print(n_nodes)
        gamma_1=self.gamma_1
        gamma_2=self.gamma_2
        gamma_3=self.gamma_3
        A_elem = np.zeros((4, 4))
        b_elem=np.zeros((4,1))
        if self.format == 'full':
            A_1 = np.zeros((n_nodes,n_nodes))
            A_2 = np.zeros((n_nodes,n_nodes))
            A_3= np.zeros((n_nodes,n_nodes))
        if self.format == 'sparse':
            col = []
            row = []
            data = []
            col2 = []
            row2 = []
            data2 = []
            col3= []
            row3 = []
            data3 = []
            
            

        b_1 = np.zeros((n_nodes,1))

        b_2 = np.zeros((n_nodes,1))
       
        b_3 = np.zeros((n_nodes,1))
        n_elem = np.shape(elem)[0]
        for ind_elem in range(n_elem):
            Id = map(int,elem[ind_elem])      
            S1 = nodes[Id[0]-1,:].reshape(1,2)
            S2 = nodes[Id[1]-1,:].reshape(1,2)
            S3 = nodes[Id[2]-1,:].reshape(1,2)
            S4 = nodes[Id[3]-1,:].reshape(1,2)
            nodeCoordinates=np.concatenate((S1,S2,S3,S4),axis=0)
            #verification si une arete appartient a un des bords 
            check_1=np.array([ node in gamma_1 for node in Id])
            compte_1={k: check_1.tolist().count(k) for k in set([True,False])}
            check_2=np.array([ node in gamma_2 for node in Id])
            compte_2={k: check_2.tolist().count(k) for k in set([True,False])}
            check_3=np.array([ node in gamma_3 for node in Id])
            compte_3={k: check_3.tolist().count(k) for k in set([True,False])}
            if (compte_1[1]==2) :     
                nodes_gamma=[]
                for i in np.arange(4) :
                      if check_1[i]== True: nodes_gamma.append(int(i))
                A_elem , b_elem = self.compute_A_b_elem_convection(nodeCoordinates,nodes_gamma)
            
                for i in range(4):
                    ind_i=Id[i]-1
                    b_1[ind_i,0] = b_1[ind_i,0] + b_elem[i,0]
                    for j in range(4):
                        ind_j=Id[j]-1
                        if self.format == 'full':
                            A_1[ind_i,ind_j] = A_1[ind_i,ind_j] + A_elem[i,j]
                        if self.format == 'sparse':
                            row.extend([ind_i])
                            col.extend([ind_j])
                            data.extend([A_elem[i,j]])
                        

                        
                        
            if (compte_2[1]==2) :     
                nodes_gamma=[]
                for i in np.arange(4) :
                      if check_2[i]== True: nodes_gamma.append(int(i))
                A_elem , b_elem = self.compute_A_b_elem_convection(nodeCoordinates,nodes_gamma)
                for i in range(4):
                    ind_i=Id[i]-1
                    b_2[ind_i,0] = b_2[ind_i,0] + b_elem[i,0]
                    for j in range(4):
                        ind_j=Id[j]-1
                        if self.format == 'full':
                            A_2[ind_i,ind_j] = A_2[ind_i,ind_j] + A_elem[i,j]
                        if self.format == 'sparse':
                            row2.extend([ind_i])
                            col2.extend([ind_j])
                            data2.extend([A_elem[i,j]])
                                  
                        
                        
            if (compte_3[1]==2) :     
                nodes_gamma=[]
                for i in np.arange(4) :
                      if check_3[i]== True: nodes_gamma.append(int(i))
                A_elem , b_elem = self.compute_A_b_elem_convection(nodeCoordinates,nodes_gamma)
            
                for i in range(4):
                    ind_i=Id[i]-1
                    b_3[ind_i,0] = b_3[ind_i,0] + b_elem[i,0]
                    for j in range(4):
                        ind_j=Id[j]-1
                        if self.format == 'full':
                            A_3[ind_i,ind_j] = A_3[ind_i,ind_j] + A_elem[i,j]
                        if self.format == 'sparse':
                            row3.extend([ind_i])
                            col3.extend([ind_j])
                            data3.extend([A_elem[i,j]])
                             
        if self.format == 'full':
            self.A_1=A_1
            self.A_2=A_2
            self.A_3=A_3
        
        elif self.format == 'sparse':
            data = np.array(data)
            row = np.array(row)
            col = np.array(col)
            data2 = np.array(data2)
            row2= np.array(row2)
            col2 = np.array(col2)
            data3 = np.array(data3)
            row3= np.array(row3)
            col3 = np.array(col3)
            self.data_2 = data2
            self.row_2 = row2
            self.col_2 = col2
            self.data_1 = data
            self.row_1 = row
            self.col_1 = col   
            self.data_3 = data3
            self.row_3= row3
            self.col_3 = col3
            A_1 = spa.coo_matrix((self.data_1, (self.row_1, self.col_1)), shape=(len(self.nodes), len(self.nodes)))
            A_2 = spa.coo_matrix((self.data_2, (self.row_2, self.col_2)), shape=(len(self.nodes), len(self.nodes)))
            self.A_1 = A_1.tocsc()
            self.A_2 = A_2.tocsc()
            A_3 = spa.coo_matrix((self.data_3, (self.row_3, self.col_3)), shape=(len(self.nodes), len(self.nodes)))
            self.A_3 = A_3.tocsc()


        self.b_1=b_1
        self.b_2=b_2
        self.b_3=b_3
                        
                        
                        
    def assembly_proj(self,phi):
        self.b_1r=np.transpose(phi).dot(self.b_1)
        self.b_2r=np.transpose(phi).dot(self.b_2)
        self.b_3r=np.transpose(phi).dot(self.b_3)
        if self.format == 'full':
            self.A_1r=np.transpose(phi).dot(self.A_1)
            self.A_1r=self.A_1r.dot(phi)
            self.A_2r=np.transpose(phi).dot(self.A_2)
            self.A_2r=self.A_2r.dot(phi)
            self.A_3r=np.transpose(phi).dot(self.A_3)
            self.A_3r=self.A_3r.dot(phi)
            self.A_4r=np.transpose(phi).dot(self.A_4)
            self.A_4r=self.A_4r.dot(phi)
            self.A_5r=np.transpose(phi).dot(self.A_5)
            self.A_5r=self.A_5r.dot(phi)
        if self.format == 'sparse':
            phi=spa.csc_matrix(phi)
            self.A_1r=np.transpose(phi).dot(self.A_1)
            self.A_1r=self.A_1r.dot(phi)
            self.A_2r=np.transpose(phi).dot(self.A_2)
            self.A_2r=self.A_2r.dot(phi)
            self.A_3r=np.transpose(phi).dot(self.A_3)
            self.A_3r=self.A_3r.dot(phi)
            self.A_4r=np.transpose(phi).dot(self.A_4)
            self.A_4r=self.A_4r.dot(phi)
            self.A_5r=np.transpose(phi).dot(self.A_5)
            self.A_5r=self.A_5r.dot(phi)
            
            

            

    def compute_A_b(self,x):  # [k_cu,k_ni,h_hot,T_hot,h_out,T_out,h_cool,T_cool]
        k=[x[0],x[1]]
        cond_gamma_1=[x[2]*1e3,x[3]] #hot 
        cond_gamma_2=[x[6]*1e3,x[7]]#cool
        cond_gamma_3=[x[4]*1e3,x[5]]  #out
        nodes, elem = self.nodes, self.Q4_elem 
        n_nodes = nodes.shape[0]
        b_1=self.b_1
        b_2=self.b_2
        b_3=self.b_3
        A_1=self.A_1
        A_2=self.A_2
        A_3=self.A_3
        A_4=self.A_4
        A_5=self.A_5
        if self.format == 'full':
            A = np.zeros((n_nodes,n_nodes))
            A = cond_gamma_1[0] *A_1 + cond_gamma_2[0]* A_2+ cond_gamma_3[0]*A_3 +k[0] *A_4 +k[1] *A_5

        if self.format == 'sparse':
            A= float(cond_gamma_1[0]) *A_1 + float(cond_gamma_2[0])* A_2+ float(cond_gamma_3[0])*A_3 +float(k[0]) *A_4 + float(k[1]) *A_5
            
        b = np.zeros((n_nodes,1))
        b= cond_gamma_1[0] *cond_gamma_1[1]*b_1 + cond_gamma_2[0]*cond_gamma_2[1]* b_2+ cond_gamma_3[0]*cond_gamma_3[1]*b_3
        return A, b
        
        
    def compute_A_br(self,x):  # [k_cu,k_ni,h_hot,T_hot,h_out,T_out,h_cool,T_cool]
        k=[x[0],x[1]]
        cond_gamma_1=[x[2]*1e3,x[3]] #hot 
        cond_gamma_2=[x[6]*1e3,x[7]]#cool
        cond_gamma_3=[x[4]*1e3,x[5]]  #out
        nodes, elem = self.nodes, self.Q4_elem 
        n_nodes = nodes.shape[0]
        A_1=self.A_1r
        A_2=self.A_2r
        A_3=self.A_3r
        A_4=self.A_4r
        A_5=self.A_5r
        b_1=self.b_1r
        b_2=self.b_2r
        b_3=self.b_3r
        if self.format == 'full':
            A = np.zeros((n_nodes,n_nodes))
            A = cond_gamma_1[0] *A_1 + cond_gamma_2[0]* A_2+ cond_gamma_3[0]*A_3 +k[0] *A_4 +k[1] *A_5

        if self.format == 'sparse':
            A= float(cond_gamma_1[0]) *A_1 + float(cond_gamma_2[0])* A_2+ float(cond_gamma_3[0])*A_3 +float(k[0]) *A_4 + float(k[1]) *A_5
            
        b = np.zeros((n_nodes,1))
        b= cond_gamma_1[0] *cond_gamma_1[1]*b_1 + cond_gamma_2[0]*cond_gamma_2[1]* b_2+ cond_gamma_3[0]*cond_gamma_3[1]*b_3
        return A, b
     
        
    
    
    
    
    
    
    def solve_heatproblem(self,x):
        A, b = self.compute_A_b(x)
        if self.format == 'full':
            sol = np.linalg.solve(A,b)
            return A,b,sol
        if self.format == 'sparse':
            sol = spalinalg.spsolve(A,b)
            return A,b ,sol
        
        
    def solve_heatproblem_near(self,x):
        A, b = self.compute_A_b(x)
        if self.format == 'full':
            sol = np.linalg.solve(A,b)
            return A,b,sol
        if self.format == 'sparse':
            lu_prec = spalinalg.splu(A)
            sol = lu_prec.solve(b)
            return lu_prec,b ,sol
        
        
    def solve_heatproblemr(self,x):
        A, b = self.compute_A_br(x)
        if self.format == 'full':
            sol = np.linalg.solve(A,b)
            return sol
        if self.format == 'sparse':
            sol = spalinalg.spsolve(A,b)
            return sol
