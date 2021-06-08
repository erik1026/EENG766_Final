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
import scipy.linalg as la

def to_rot(R):
    u,s,vh = la.svd(R)
    return u.dot(vh)

def create_ij(offset,shape):
    '''
    This function is to create the "rows" and "cols" that will
    be used to index a matrix of size "shape" if the offset is
    passed in. To create a sparse matrix, you can flatten the array
    and use the output of this function to give you the rows and columns

    Args:
        offset:  a 2-element set of ints denoting what the offset will be for
            the matrix object.  Should be (row,col)
        shape:   how big is this thing that is being put in there.  Assumed 2d

    Returns:  a tuple of n-element, 1d numpy arrays, where n is shape[0]*shape[1]
    '''
    assert len(shape)==2, 'create_ij only works for 2d arrays'
    full_len = shape[0]*shape[1]
    rows = np.zeros(full_len)
    cols = np.zeros(full_len)
    for i in range(shape[0]):
        rows[i*shape[1]:(i+1)*shape[1]]=i
        cols[i*shape[1]:(i+1)*shape[1]]=np.arange(shape[1])
    return (rows+offset[0],cols+offset[1])

class FOGM:
    def __init__(self, rw, num_sensors=1, bias_rw=None, bias_tau=None, init_sd=None):
        '''
        Creates a class that implements (num_sensors independent) 
        First Order Gauss Markov Processes (FOGM).  The model
        assumes error = N(0,rw^2) + b
        where \dot{b} = -\frac{1}{bias_tau}b + nu
        where nu = N(0,bias_rw^2)
        init_sd is what standard deviation to use to get the initial bias.
        By default it should be equal to the "steady state" variance of the FOGM

        Args:
            rw:  standard deviation of white noise
            num_sensors:  How many of these things to track in this class
            bias_rw:  the standard deviation of the additive white noise for 
                the bias part (random walk)
            bias_tau: the time constant for the FOGM
            init_sd: the standard deviation of the initial bias

        '''
        self.rw = rw
        assert (bias_rw is None and bias_tau is None) or\
            (bias_rw is not None and bias_tau is not None), \
                "Need both bias_tau and bias_rw, or neither."
        self.biased = bias_rw is not None
        if self.biased and init_sd is None:
            init_sd = m.sqrt(bias_tau)*bias_rw
        
        self.bias_rw = bias_rw
        self.tau = bias_tau

        self.ns = num_sensors

        if self.biased:
            self.bias = np.random.randn(num_sensors)*init_sd
        else:
            self.bias = np.zeros(num_sensors)

    def sample(self,dt):
        '''
        Returns a sample from the FOGM random process.  Basically,
        it returns a random value and propagates the bias states forward
        dt time
        '''
        going_out = np.random.randn(self.ns)*self.rw + self.bias
        if self.biased:
            self.bias = self.bias*(1-dt/self.tau) + \
                np.random.randn(self.ns)*self.bias_rw*m.sqrt(dt)
        return going_out

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
        self.n_pos = 36 # 360/10 = 36 waypoints
        self.world_pts = np.array([[-.5, -.5, 0.5], [.5,-.5,0.5],[-.5,.5,0.5],
                                    [.5,.5,0.5], [-.5,-.5,1], [.5,-.5,1],
                                    [-.5,.5,1], [.5,.5,1]]) 
        
        # Create a camera for the system
        self.cam = vis.Camera()

        gyro_rw = .0001 # radians / s
        gyro_tau = 20 # s
        gyro_bias_rw = .00001
        
        accel_rw = .001 # m/s^2
        accel_tau = 100 # s
        accel_bias_rw = .0001
    
        R = 1.
        self.s_R_inv = 1/m.sqrt(R)

        self.gyro_noise = FOGM(gyro_rw, 3, gyro_bias_rw, gyro_tau)
        self.accel_noise = FOGM(accel_rw, 3, accel_bias_rw, accel_tau)
        self.dt = 1.
        self.n_grav = np.array([0,0.,9.8])

        ## gyro
        self.s_grw_inv = 1/self.gyro_noise.rw
        self.gyro_tau = self.gyro_noise.tau
        self.s_gbrw_inv = 1/(self.gyro_noise.bias_rw*m.sqrt(self.dt))
        ## accel
        self.s_arw_inv = 1/self.accel_noise.rw
        self.accel_tau = self.accel_noise.tau
        self.s_abrw_inv = 1/(self.accel_noise.bias_rw*m.sqrt(self.dt))

        self.g_biases = np.zeros((self.n_pos,3)) #gyro biases
        self.a_biases = np.zeros((self.n_pos,3)) #accel biases

        self.s_pos_rw_inv = 1000 # really small standard deviation, 
        # Now create the s_Q_inv matrix
        sQI = np.zeros((15,15))
        ## easy first .. biases
        sQI[-3:,-3:] = np.eye(3)*self.s_abrw_inv
        sQI[-6:-3, -6:-3] = np.eye(3)*self.s_gbrw_inv
        # velocity terms
        sQI[3:6,3:6] = np.eye(3) * self.s_arw_inv
        # attitude
        sQI[6:9,6:9] = np.eye(3)* self.s_grw_inv
        sQI[:3,:3] = np.eye(3) * self.s_pos_rw_inv
        self.s_Q_inv = sQI

        '''
        Create a series of positions for the camera that rotate the camera around the 
        object in a circle.Places the camera at a height of 5 meters and backs up until all the object points are 
        frame then finally angles the camera such that the center of mass of the object is in the center of
        the frame.
        '''
        
        self.cam_pos = np.zeros((self.n_pos, 3)) # w_Cam : Position of cam in world frame
        self.cam_rot = [] # c_R_w : Rotation from world frame to camera frame
        self.cam_vel = np.zeros((self.n_pos - 1, 3)) # Velocity in world frame
        self.gyros = np.zeros((self.n_pos -1, 3))
        self.accels = np.zeros((self.n_pos-1, 3))
        self.cam.set_rot(rot.axis0(-90, True).dot(rot.axis1(0, True).dot(rot.axis2(90, True))))
        self.radius = 8.
        self.z_offset = .5
        
        # Compute Camera positions and rotations
        tmp_offset = copy.deepcopy(self.cam.w_cam)
        for i in range(self.n_pos ):
            angle = ( (2*m.pi) / 36 ) * i
            self.cam_pos[i] = np.array([m.cos(angle)*self.radius, m.sin(angle)*self.radius, self.z_offset])
            
            self.cam_rot.append( ((self.cam.c_R_w).dot(rot.axis2(angle))) )

        # Compute Camera Velocities
        self.V = (2*m.pi*self.radius) / self.n_pos
        for i in range(self.n_pos - 1):
            self.cam_vel[i] = self.cam_rot[i].T.dot(np.array([self.V, 0., 0.])) # In the navigation frame
            print(f'{self.cam_vel[i]}')
        
        for i in range(self.n_pos-1):
            ## compute gyros
            dR = self.cam_rot[i+1].dot(self.cam_rot[i].T)
            self.gyros[i] = R2v(dR)/self.dt + self.gyro_noise.sample(self.dt)
            ## Really, the accels are two steps back as velocity is one
            ## step back and accel is one behind it.  However, I just
            ## leave 0's in there if data is not there yet.
            if i > 0:  #Don't go too far back
                ## compute accels
                n_dV = self.cam_vel[i]-self.cam_vel[i-1]
                accel_true = n_dV/self.dt - self.n_grav
                self.accels[i-1] = self.cam_rot[i-1].dot(accel_true) + self.accel_noise.sample(self.dt)
        
        
    def compute_h(self):
        '''
        # List of world to image point tuples
        for i in range(self.cam_pos.shape[0] + 1):
            self.cam.set_loc(self.cam_pos[i])
            self.cam.set_rot(self.cam_rot[i])#v2R(self.cam_rot[i]).dot( self.cam.c_R_w) )
            img_pts = self.cam.project_points(self.world_pts)
            PnP_tuples = list(zip(self.world_pts, img_pts))            
            #print(PnP_tuples)
            est_pos = self.cam.refine_PnP(PnP_tuples, self.cam.K, init_R=self.cam_rot[i], init_t=self.cam_pos[i], im_size=(self.cam.K[0,2]*2, self.cam.K[1,2]*2)).w_cam 
            print(f'Diff\n{self.cam.w_cam - est_pos}\n')
        '''
        num_world_points = self.world_pts.shape[0]
        meas = np.zeros((self.n_pos * num_world_points, 2)) # n measurements and m world points that produce an u and v
        for i in range(self.n_pos):
            self.cam.set_loc(self.cam_pos[i])
            self.cam.set_rot(self.cam_rot[i])
            individual_meas = self.cam.project_points(self.world_pts)
            meas[i*num_world_points:i*num_world_points +8, :] = individual_meas

        return meas
        

    def compute_H(self):

        # Derivatives of position is just an Identity matrix
        H = np.zeros((6,15))
        H[:,:3] = np.eye(3)

        # Need to figure out the derivatives for rotation...

        pass
    
    def compute_A(self):
        n_rows = self.n_pos * 3 + (self.n_pos-1)*15
        n_columns = self.n_pos * 15
        n_meas_entries = self.n_pos

        # Derivatives for measurments
        meas_entries = np.zeros(n_meas_entries)
        meas_rows = np.zeros(n_meas_entries)
        meas_cols = np.zeros(n_meas_entries)

        for i in range(self.n_pos):
            H = self.compute_H()
            meas_entries[(i*6) : (i*6)+6] = self.s_R_inv*H.flatten()
            offset = (i*6, i*15)
            rows, cols = create_ij(offset, H.shape)
            meas_rows[(i*6) : (i*6)+6] = rows
            meas_cols[(i*6) : (i*6)+6] = cols
        
        # Dynamics
        ds = 450 #dynamics size...
        n_dyn_entries = (self.n_ts-1)*ds
        dyn_entries = np.zeros(n_dyn_entries)
        dyn_rows = np.zeros(n_dyn_entries)
        dyn_cols = np.zeros(n_dyn_entries)
        
        for i in range(self.n_pos-1):
            F = self.s_Q_inv.dot(self.compute_F(i))
            offset = (self.n_pos+i*15,i*15)
            rows,cols = create_ij(offset,F.shape)
            dyn_entries[i*450:i*450+450] = F.flatten()
            dyn_rows[i*450:i*450+450] = rows
            dyn_cols[i*450:i*450+450] = cols
        
        entries = np.concatenate((meas_entries,dyn_entries))
        sp_rows= np.concatenate((meas_rows,dyn_rows))
        sp_cols= np.concatenate((meas_cols,dyn_cols))
            

    def f(self,idx):
        '''
        Propagate the "idx" timestep forward to predict its
        location at time idx+1

        Returns a tuple with (position, velocity, attitude, gyro bias, accel bias)
        '''
        g_bias = self.g_biases[idx] * (1-self.dt/self.gyro_tau)
        a_bias = self.a_biases[idx] * (1-self.dt/self.accel_tau)
        att= copy.copy(self.cam_rot[idx])
        vel = copy.copy(self.cam_vel[idx])
        pos = copy.copy(self.cam_pos[idx])
        divider = 1
        my_dt = self.dt/float(divider)
        
        for i in range(divider):
            next_vel = vel + my_dt*(self.n_grav + att.T.dot(self.accels[idx]-self.a_biases[idx]))
            pos += (next_vel+vel)/2 * my_dt
            att = v2R(my_dt*(self.gyros[idx]-self.g_biases[idx])).dot(att)
            vel = next_vel
        # Just for kicks, though probably not needed
        att = to_rot(att)
        return (pos,vel,att,g_bias,a_bias)

    def compute_Y(self):
        sens_meas = self.n_pos
        n_rows = sens_meas*6 +(self.n_pos-1)*15

        meas = self.compute_h()

        y = np.zeros(n_rows)
        # Start with sensor measurements
        for i in range(self.n_pos):
            y[i*6:i*6+6]= self.s_R_inv*(meas[i] - self.h(self.cam_pos[i]))
        
        # Now for dynamics measurements
        dyn_st = sens_meas * 6
        dyn_cost = np.zeros(15)
        for i in range(self.n_ts-1):
            pred_x = self.f(i)
            dyn_cost[:3] = pred_x[0]-self.cam_pos[i+1]
            dyn_cost[3:6] = pred_x[1] - self.cam_vel[i+1]
            dyn_cost[6:9] = R2v(pred_x[2].T.dot(self.attitudes[i+1]))
            dyn_cost[9:12] = pred_x[3] - self.g_biases[i+1]
            dyn_cost[12:15] = pred_x[4] - self.a_biases[i+1]
            y[dyn_st+i*15:dyn_st+i*15+15] = -self.s_Q_inv.dot(dyn_cost)

        return y
        
    
    def add_delta(self, delta):
        assert len(delta) == 15*self.n_pos
        for i in range(self.n_pos):
            si = i*15 # starting index
            self.positions[i] += delta[si:si+3]
            self.velocities[i] += delta[si+3:si+6]
            self.attitudes[i] = self.attitudes[i].dot(v2R(delta[si+6:si+9]))
            self.g_biases[i] += delta[si+9:si+12]
            self.a_biases[i] += delta[si+12:si+15]
        



if __name__ == "__main__":

    curr_state = sim_world()
    meas = curr_state.compute_h()
    #print(meas)
    '''
    done = False
    num_iters = 0
    y = curr_state.compute_Y()
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
        done = num_iters > 10 or la.norm(delta_x) < (0.001 * m.sqrt(curr_state.nts))

        pass
    '''
