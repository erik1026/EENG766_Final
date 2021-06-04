# This file will contain the main simulation class
from numpy.lib.shape_base import _put_along_axis_dispatcher
import imu as imu
import visual as vis
import numpy as np
import sys
import math as m
import copy 
import matplotlib.pyplot as plt
import rot as rot
from create_truth import R2v, v2R
import scipy.sparse.linalg as spla

class sim_world:
    '''
        Our simulated world environment. It is relatively simple, only contains the world points that represent 
        our object of interest, a single camera, and the imu.

        world_pts: a (nx3) array that contains the x,y,z coordinates of each world point
        cam: The camera that will be used to capture images
        imu: model of an imu that will be used for the optimization
        cam_pos: a (nx3) array that contains all the positions that the camera will move to throughout the flight
        cam_rot: a (nx3) array that contains all the euler angles for the cameras rotation about our world_pts
    '''
    
    def __init__(self,imu=None, scenerio=None, world_pts=None): 
        # If no world points are passed in then we'll create a rectangular prism to work with
        
        self.world_pts = np.array([[-.5, -.5, 0.5], [.5,-.5,0.5],[-.5,.5,0.5],
                                    [.5,.5,0.5], [-.5,-.5,1], [.5,-.5,1],
                                    [-.5,.5,1], [.5,.5,1]]) 
        
        # Create a camera for the system
        self.cam = vis.Camera()

        '''
        Create a series of positions for the camera that rotate the camera around the 
        object in a circle.Places the camera at a height of 5 meters and backs up until all the object points are 
        frame then finally angles the camera such that the center of mass of the object is in the center of
        the frame.
        '''
        len_of_flight = 36 #360deg/10 = 36 waypoints
        self.cam_pos = np.zeros((len_of_flight+1, 3))
        self.cam_rot = []#= np.zeros((len_of_flight+1, 3))
        world_offset = 3. 
        #self.cam.set_rot(rot.axis1(90, True)) # Rotate about the y-axis so that the camera is pointing along the world X
        self.cam.set_rot(rot.axis2(180, True).dot(rot.axis1(90, True)))
        self.radius = 8.
        self.z_offset = .5
        
        #self.cam.backup_to_view(self.world_pts, 250)
        tmp_offset = copy.deepcopy(self.cam.w_cam)
        for i in range(len_of_flight +1):
            angle = ( (2*m.pi) / 36 ) * i
            self.cam_pos[i] = np.array([m.cos(angle)*self.radius, m.sin(angle)*self.radius, self.z_offset])
            
            self.cam_rot.append(rot.axis2(angle).dot(self.cam.c_R_w) )

    def compute_h(self):
        # List of world to image point tuples
        for i in range(self.cam_pos.shape[0] + 1):
            self.cam.set_loc(self.cam_pos[i])
            self.cam.set_rot(self.cam_rot[i])#v2R(self.cam_rot[i]).dot( self.cam.c_R_w) )
            img_pts = self.cam.project_points(self.world_pts)
            PnP_tuples = list(zip(self.world_pts, img_pts))            
            #print(PnP_tuples)
            est_pos = self.cam.refine_PnP(PnP_tuples, self.cam.K, init_R=self.cam_rot[i], init_t=self.cam_pos[i], im_size=(self.cam.K[0,2]*2, self.cam.K[1,2]*2)).w_cam 
            print(f'Diff\n{self.cam.w_cam - est_pos}\n')

    def compute_H(self):
        pass
    
    def compute_A(self):
        pass

    def compute_Y(self):
        pass
    
    def add_delta(self):
        pass



if __name__ == "__main__":

    curr_state = sim_world()

    done = False
    num_iters = 0
    y = world.compute_Y()
    while not done:
        A = curr_state.compute_A()

        # Using a sparse matrix
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        delta_x = spla.spsolve(AtA, Aty)

        # Time for the fancy scaling stuff
        scale = 1
        scale_good = False
        while not scale_good:
            pred_y = y - A.dot(delta_x  * scale)
            next_state = copy.deepcopy(curr_state)
            next_state.add_delta(delta_x * scale)
            next_y = next_state.compute_Y()
            ratio = (y.dot(y) - next_y.dot(next_y)) / (y.dot(y) - pred_y.dot(pred_y))
            if ratio > 4 or ratio < 0.25:
                scale /= 2
                if scale < 0.01:
                    print(f'At scale: {scale}')
            else:
                scale_good = True
        curr_state = next_state
        y = next_y
        print(f'y is {la.norm(next_y)}, long, used scale, {scale}')
        num_iters += 1
        done = num_iters > 10 or la,.norm(delta_x) < (0.001 * sqrt(curr_state.nts))

        pass
