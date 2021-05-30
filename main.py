# This file will contain the main simulation class
from numpy.lib.shape_base import _put_along_axis_dispatcher
import imu as imu
import visual as vis
import numpy as np
import sys
import math as m
import copy 
import matplotlib.pyplot as plt

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
    
    def __init__(self,imu=None, scenerio=None, world_pts=None, camera=None, cam_pos=None, cam_rot=None, len_of_flight=None): 
        # If no world points are passed in then we'll create a rectangular prism to work with
        if world_pts is None:
            self.world_pts = np.array([[0, 0, 0.], [1,0,0],[0,1,0],
                                    [1,1,0], [0,0,1], [1,0,1],
                                    [0,1,1], [1,1,1]])
        
        if camera is None:
            self.cam = vis.Camera()
        else:
            self.cam = camera
        
        # if imu is None:
        #     self.imu = imu(scenerio)
        # else:
        #     self.imu = imu

        # If no positions are passed in then the camera is placed at a z offset of 5 meters
        # and rotates the camera in order to keep the world_pts in frame
        if cam_pos is None and cam_rot is None:
            len_of_flight = 36 #360deg/10 = 36 waypoints
            self.cam_pos = np.zeros((len_of_flight+1, 3))
            self.cam_rot = np.zeros((len_of_flight+1, 3))
            world_offset = 5. 
            
            self.cam.set_loc(np.array([world_offset,world_offset,world_offset]))
            
            self.cam.backup_to_view(self.world_pts, 5)
            tmp_loc = copy.deepcopy(self.cam.w_cam)
            for i in range(len_of_flight +1):
                angle = ( (2*m.pi) /36) * i
                self.cam_pos[i] = np.array([m.cos(angle), m.sin(angle), tmp_loc[2]])
                self.cam_rot[i] = np.array([])
                
            
            plt.plot(self.cam_pos[:, 0], self.cam_pos[:, 1])
            plt.show()
                
            
            
        
        elif(cam_pos is None and cam_rot is not None):
            print(f'Passed in Camera positions but no rotations were given, exiting...')
            sys.exit()
        elif (cam_pos is not None and cam_rot is None):
            print(f'Passed in Camera rotations but no positions were given, exiting...')
            sys.exit()
        else:
            assert(len(cam_pos) == len(cam_rot)), "Camera positions and rotations are not the same length!"
            self.cam_pos = cam_pos
            self.cam_rot = cam_rot
        

    def propogate_cam(self):
        '''
        This function will propogate the camera forward to it's next position in a circular fashion, nothing fancy
        '''



        pass


if __name__ == "__main__":
    
    
    world = sim_world()



    print("Inside main...") 
