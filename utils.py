import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import copy

def member_rotation(member, nodes):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member

    xi = nodes[node_i][0]  # x-coord for node i
    yi = nodes[node_i][1]  # y-coord for node i
    xj = nodes[node_j][0]  # x-coord for node j
    yj = nodes[node_j][1]  # y-coord for node j

    # Angle of member with respect to horizontal axis

    dx = xj - xi  # x-component of vector along member
    dy = yj - yi  # y-component of vector along member
    L = math.sqrt(dx**2 + dy**2)  # Magnitude of vector (length of member)
    member_vector = np.array([dx, dy])  # Member represented as a vector

    # Need to capture quadrant first then appropriate reference axis and offset angle
    if dx > 0 and dy == 0:
        theta = 0
    elif dx == 0 and dy > 0:
        theta = math.pi / 2
    elif dx < 0 and dy == 0:
        theta = math.pi
    elif dx == 0 and dy < 0:
        theta = 3 * math.pi / 2
    elif dx > 0 and dy > 0:
        # 0<theta<90
        ref_vector = np.array([1, 0])  # Vector describing the positive x-axis
        theta = math.acos(ref_vector.dot(member_vector) / (L))  # Standard formula for the angle between two vectors
    elif dx < 0 and dy > 0:
        # 90<theta<180
        ref_vector = np.array([0, 1])  # Vector describing the positive y-axis
        theta = (math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors
    elif dx < 0 and dy < 0:
        # 180<theta<270
        ref_vector = np.array([-1, 0])  # Vector describing the negative x-axis
        theta = math.pi + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors
    else:
        # 270<theta<360
        ref_vector = np.array([0, -1])  # Vector describing the negative y-axis
        theta = (3 * math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors

    return [theta, L]

def calc_rotation_length(members, nodes):
    # Calculate rotation and length for each member and store
    rotations = np.array([])  # Initialise an array to hold rotations
    lengths = np.array([])  # Initialise an array to hold lengths
    for member in members:
        [angle, length] = member_rotation(member, nodes)
        rotations = np.append(rotations, angle)
        lengths = np.append(lengths, length)
    return rotations, lengths
    
def plot_deflection(members,nodes,mbrForces,members_area,UG,xFac,turn,output_dir='images'):

    members_area = members_area * 500
  
    
    #FIGURE TO PLOT DEFLECTED SHAPE
    fig = plt.figure(figsize=(3,3),dpi=80) 
    axes = fig.add_axes([0.1,0.1,2,2]) 
    fig.gca().set_aspect('equal', adjustable='box')

    #Plot members

    for index,member in enumerate(members):  
        
        node_s = nodes[member[0]]  # Node number for node i of this member
        node_e = nodes[member[1]]  # Node number for node j of this member

        #Index of DoF for this member
        i = 2*member[0] #horizontal DoF at node i of this member 
        j = 2*member[1] #horizontal DoF at node j of this member

        axes.plot([node_s[0],node_e[0]],[node_s[1],node_e[1]],'grey', lw=0.75) #Member
        if mbrForces[index]>0:
            color = 'r'
        else:
            color = 'b'

        if members_area[index] > 1e-1:
            axes.plot([node_s[0] + UG[i]*xFac, node_e[0] + UG[j]*xFac], [node_s[1] + UG[i+1]*xFac, node_e[1] + UG[j+1]*xFac],color,lw=members_area[index]) #Deformed member

    axes.set_xlabel('Distance (m)')
    axes.set_ylabel('Distance (m)')
    axes.set_title('Deflected shape')
    axes.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/structure_{turn:03d}.png', bbox_inches='tight')
    plt.close(fig)
    
def calc_member_forces_def(member,A,theta,L,U,E):
    
    node_i = member[0] #Node number for node i of this member
    node_j = member[1] #Node number for node j of this member    
    #Primary stiffness matrix indices associated with each node

    #Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)
    T = np.array([[c,s,0,0],[0,0,c,s]])

    disp = np.array([[U[node_i][0],U[node_i][1],U[node_j][0],U[node_j][1]]]).T #Glocal displacements
    disp_local = np.matmul(T,disp).ravel() #Local displacements    
    F_axial = (A*E/L)*(disp_local[1]-disp_local[0]) #Axial loads    
    mbrForce = F_axial #Store axial loads
    mbrDef = disp_local[1]-disp_local[0]
    return mbrForce,mbrDef

def build_matrix_reduced(matrix, restrained_DOF):

    # Reduce to structure stiffness matrix by deleting rows and columns for restrained DOF
    matrix = np.delete(matrix, restrained_DOF, 0)  # Delete rows
    if len(matrix.shape) > 1 and matrix.shape[1]>1:
        matrix = np.delete(matrix, restrained_DOF, 1)  # Delete columns 
    return matrix

def calc_displacement(K, force_vector, restrained_DOF):

    Kr = build_matrix_reduced(K, restrained_DOF)
    force_vector_restrained = copy.copy(
        force_vector
    )  # Make a copy of force_vector so the copy can be edited, leaving the original unchanged
    force_vector_restrained = np.delete(
        force_vector_restrained, restrained_DOF, 0
    )  # Delete rows corresponding to restrained DOF

    U = np.linalg.solve(Kr,force_vector_restrained)
    return U

def assemble_UG(U, n_DOF, restrained_DOF):
    # Construct the global displacement vector
    UG = np.zeros(n_DOF)  # Initialise an array to hold the global displacement vector
    c = 0  # Initialise a counter to track how many restraints have been imposed
    for i in np.arange(n_DOF):
        if i in restrained_DOF:
            # Impose zero displacement
            UG[i] = 0
        else:
            # Assign actual displacement
            UG[i] = U[c]
            c = c + 1

    return UG
    
def calculate_Kg(theta,mag,E,A):
    """
    Calculate the global stiffness matrix for an axially loaded bar
    """    
        
    c = math.cos(theta)
    s = math.sin(theta)
    
    K11 = (E*A/mag)*np.array([[c**2,c*s],[c*s,s**2]]) #Top left quadrant of global stiffness matrix
    K12 = (E*A/mag)*np.array([[-c**2,-c*s],[-c*s,-s**2]]) #Top right quadrant of global stiffness matrix   
    K21 = (E*A/mag)*np.array([[-c**2,-c*s],[-c*s,-s**2]]) #Bottom left quadrant of global stiffness matrix   
    K22 = (E*A/mag)*np.array([[c**2,c*s],[c*s,s**2]]) #Bottom right quadrant of global stiffness matrix          
    
    return [K11, K12, K21,K22]

def calc_DOF(members):
    nDoF = len(set(members.flatten())) * 2
    return nDoF

def build_K(members, members_area, orientations, lengths, E):
    n_DOF = calc_DOF(members)
    Kp = np.zeros([n_DOF, n_DOF])  # Initialise the primary stiffness matrix
    for n, mbr in enumerate(members):

        # Calculate the quadrants of the global stiffness matrix for the member
        theta = orientations[n]
        L = lengths[n]
        A = members_area[n]
        [K11, K12, K21, K22] = calculate_Kg(theta, L, E, A)

        node_i = mbr[0]  # Node number for node i of this member
        node_j = mbr[1]  # Node number for node j of this member

        # Primary stiffness matrix indices associated with each node

        i = 2* node_i
        j = 2 * node_j

        Kp[i : i + 2, i : i + 2] = Kp[i : i + 2, i : i + 2] + K11
        Kp[j : j + 2, j : j + 2] = Kp[j : j + 2, j : j + 2] + K22
        Kp[i : i + 2, j : j + 2] = Kp[i : i + 2, j : j + 2] + K12
        Kp[j : j + 2, i : i + 2] = Kp[j : j + 2, i : i + 2] + K21

    return Kp