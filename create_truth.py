import numpy as np
import scipy.linalg as la
import pickle as pkl
from math import sqrt, radians
import matplotlib.pyplot as plt
import copy

def norm_R(R):
    u,s,vh = la.svd(R)
    return u.dot(vh)

def vec2ss(vec):
    '''
    Turn a 3 vec into a 3x3 skew symmetric matrix
    (Assuming "coordinate frame" rotations)
    '''
    going_out = np.zeros((3,3))
    going_out[0,1] = vec[2]
    going_out[0,2] = -vec[1]
    going_out[1,2] = vec[0]
    going_out -= going_out.T
    return going_out

def v2R(vec):
    '''
    Turn a 3-vec into a 3x3 DCM
    '''
    return la.expm(vec2ss(vec))

def R2v(C):
    '''
    Turn a 3x3 DCM into a 3-vec
    '''
    ss = la.logm(C)
    going_out = np.zeros(3)
    going_out[0] = (ss[1,2]-ss[2,1])/2.
    going_out[1] = (ss[2,0]-ss[0,2])/2.
    going_out[2] = (ss[0,1]-ss[1,0])/2.
    # # Warning... When the rotation about an axis approaches pi, you can get some
    # # bad results.  Use this function with care if you are living in that area
    # # alot.  While the warnings print out in this code, it is mainly used to
    # # control roll, not to determine values given to the data consumer, so I am 
    # # ignoring the problems for now (i.e. I don't know how to fix it! :)
    # if isinstance(ss[2,1], complex) or isinstance(ss[2,0], complex):
    #     print("bad C",C,'ss is',ss,'\ngoing_out is',going_out, )
    # if isinstance(ss[1,0], complex) or isinstance(ss[1,2], complex):
    #     print("bad C2",C,'ss is',ss,'\ngoing_out is',going_out)
    # if isinstance(ss[0,1], complex) or isinstance(ss[0,2], complex):
    #     print("bad C3",C,'ss is\n',ss,'\ngoing_out is',going_out)
    return going_out


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
            init_sd = sqrt(bias_tau)*bias_rw
        
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
                np.random.randn(self.ns)*self.bias_rw*sqrt(dt)
        return going_out
    
        
        

class CreateSim:
    '''
    We will assume an object that always moves in the
    direction it is pointing at stays generally flat (roll=0).

    At each time step, it will move forward in the direction it is 
    pointing by V.  Roll will also go back to 0 with a tau of 2s.
    Pitch and Yaw will change at a rate specified by the inputs 
    
    The accelerometer and gyro information will be corrupted
    off the true path thus described.

    '''


    def __init__(self, params, init_b_C_n=None, init_pos=None ):
        if 'dt' not in params:
            params['dt'] = .1
        self.dt = params['dt']

        self.gyro_noise = params['gyroFOGM']
        self.accel_noise = params['accelFOGM']

        if init_b_C_n is None:
            init_b_C_n = np.eye(3)
        
        if 'R' not in params:
            params['R'] = 1
        self.R_sd = sqrt(params['R'])

        
        if init_pos is None:
            init_pos= np.zeros(3)


        self.n_grav = np.array([0,0.,9.8])
        #Setup the simulation timestep.
        if self.dt <1: #Biggest timestep should be .1 second
            self.dt_divider=10
        else:
            self.dt_divider = np.ceil(10.*self.dt)
        
        self.n_pos = 1
        # Positions and attitudes will have n_pos entries.  velocities, gyros, and accels will have n_pos-1
        self.positions = np.zeros((self.n_pos,3))
        self.positions[0] = init_pos
        self.velocities = None
        self.b_C_ns = np.zeros((self.n_pos,3,3))
        self.b_C_ns[0] = init_b_C_n
        self.gyros = None
        self.accels = None

        if 'landmarkLocs' not in params:
            params['landmarkLocs'] = np.array([-10000,0,0., -2000, 8000, 1000,  -2000, -8000, -1500]).reshape((3,3))
        self.landmark_locs = params['landmarkLocs']
        self.meas = self.h(init_pos) + np.random.randn(len(self.landmark_locs))*self.R_sd
        self.meas.reshape((1,len(self.landmark_locs)))

        self.params=params
     
    def h(self, location):
        '''
        Take in the current location and generate what the measurements
        should be (without noise). Requires the landmark locations (from
        self) and the location being measured (a parameter)

        Args:
            location:  A (3,) numpy array with object location in it

        Returns:  A (len(landmark_locs), ) numpy array
        '''
        vec_diffs = np.zeros(len(self.landmark_locs))
        for i,landmark in enumerate(self.landmark_locs):
            vec_diffs[i] = la.norm(location-landmark)
        return vec_diffs

    def H(self,location):
        '''
        Take in the current location and generate what the derivative of
        the measurements (rows) should be w.r.t. the location (columns).
        
        Requires the landmark locations (from self) and the object location being 
        measured (a parameter)

        Args:
            location:  A (3,) numpy array with location in it

        Returns:  A (len(landmark_locs),3) numpy array
        '''
        H = np.zeros((len(self.landmark_locs),3))
        for i,landmark in enumerate(self.landmark_locs):
            H_row = np.zeros(3)
            dist = la.norm(location-landmark)
            for j in range(3):
                H_row[j] = location[j]-landmark[j]
            H_row /= dist
            H[i] = H_row
        return H

    def propagate(self, V, w_yaw, w_pitch, num_dts_forward=1):
        '''
        This takes the current robot location and propagates it forward.
        For each dt, it stores it in store_state

        Args:
            V: The velocity of the unicycle.  Can be either a scalar or an
                np.array that is num_dts_forward long
            w_yaw: How fast we should be yawing the "aircraft"
            w_pitch: How fast we should be "pitching" the "aircraft"
            num_dts_forward: How many timesteps to take (and record)

        Returns:
            The indicies to the created robot locs.  
            i.e. UnicycleSim.store_state[UnicycleSim.propagate(V,w)] will
            give you a new state value
        '''
        max_i = self.n_pos + num_dts_forward
        if hasattr(V, "__len__"): #Make sure it is the right length
            assert len(V)==num_dts_forward, "V must be scalar or length of number of updates"
        if hasattr(w_yaw, "__len__"): #Make sure it is the right length
            assert len(w_yaw)==num_dts_forward, "w_yaw must be scalar or length of number of updates"
        if hasattr(w_pitch, "__len__"): #Make sure it is the right length
            assert len(w_pitch)==num_dts_forward, "w_pitch must be scalar or length of number of updates"

        # Unfortunately, resize is pretty tempermental.  It doesn't work in debug mode, for example :(
        self.positions.resize((max_i,3))
        if self.velocities is not None:
            self.velocities.resize((max_i-1,3))
            self.gyros.resize((max_i-1,3))
            self.accels.resize((max_i-1,3))
        else:
            self.velocities = np.zeros((max_i-1,3))
            self.gyros = np.zeros((max_i-1,3))
            self.accels = np.zeros((max_i-1,3))
        self.b_C_ns.resize((max_i,3,3))
        self.meas.resize((max_i,len(self.landmark_locs)))

        #Run the actual propagation
        my_dt = self.dt/self.dt_divider
        idxs = []
        curr_pos = copy.copy(self.positions[self.n_pos-1])
        curr_b_C_n = copy.copy(self.b_C_ns[self.n_pos-1])
        for i in range(num_dts_forward):
            idx =self.n_pos + i
            # print('idx is',idx)
            if not hasattr(V, "__len__"): #assume this makes it a scalar
                curr_V=V
            else:
                curr_V = V[i]

            n_V = curr_b_C_n.T.dot(np.array([curr_V,0,0.]))
            self.velocities[idx-1]=n_V

            if not hasattr(w_yaw,"__len__"):
                curr_w_yaw=w_yaw
            else:
                curr_w_yaw = w_yaw[i]

            if not hasattr(w_pitch,"__len__"):
                curr_w_pitch=w_pitch
            else:
                curr_w_pitch = w_pitch[i]

            curr_rot_v = R2v(curr_b_C_n)
            curr_w_roll = -curr_rot_v[0]/2.
            # print('curr_w_roll is ',curr_w_roll, 'pitch',curr_w_pitch,'yaw',curr_w_yaw)
            
            for _ in range(self.dt_divider):
                dPos = curr_b_C_n.T.dot(np.array([curr_V,0,0.]))

                curr_pos += dPos * my_dt
                dR = v2R(np.array([curr_w_roll, curr_w_pitch, curr_w_yaw])*my_dt)
                curr_b_C_n = dR.dot(curr_b_C_n)

            self.positions[idx] = curr_pos
            
            self.b_C_ns[idx] = norm_R(curr_b_C_n)
            idxs.append(idx)
            
            #While I'm at it, create all the measurements
            self.meas[idx] = self.h(self.positions[idx]) + np.random.randn(3)*self.R_sd
        print('End of loop, pos idx',idx,'position',curr_pos)
        # Now that I have the true state stored, create gyro and accel 
        # measurements for these states...
        for i in range(self.n_pos-1,max_i-1):
            ## compute gyros
            dR = self.b_C_ns[i+1].dot(self.b_C_ns[i].T)
            self.gyros[i] = R2v(dR)/self.dt + self.gyro_noise.sample(self.dt)
            ## Really, the accels are two steps back as velocity is one
            ## step back and accel is one behind it.  However, I just
            ## leave 0's in there if data is not there yet.
            if i > 0:  #Don't go too far back
                ## compute accels
                n_dV = self.velocities[i]-self.velocities[i-1]
                accel_true = n_dV/self.dt - self.n_grav
                self.accels[i-1] = self.b_C_ns[i-1].dot(accel_true) + self.accel_noise.sample(self.dt)
        self.n_pos = max_i
        return idxs


if __name__ == "__main__":
    # Let's put parameters for the simulation here
    gyro_rw = .0001 # radians / s
    gyro_tau = 20 # s
    gyro_bias_rw = .00001
    gyro_FOGM = FOGM(gyro_rw, 3, gyro_bias_rw, gyro_tau)
    
    accel_rw = .001 # m/s^2
    accel_tau = 100 # s
    accel_bias_rw = .0001
    accel_FOGM = FOGM(accel_rw, 3, accel_bias_rw, accel_tau)

    params={}
    params['dt']=.1
    params['gyroFOGM']=gyro_FOGM
    params['accelFOGM']=accel_FOGM
    params['R'] = 1 #1m standard deviation
    params['landmarkLocs'] = np.array([-10000,0,0., -2000, 8500, 1000,  -2000, -8000, -1500]).reshape((3,3))

    sim = CreateSim(params)

    # I'm going to fly the following pattern:
    # Repeat 4 times (to make a box)
    ## Left turn for 3s at 30 deg/sec
    ## Forward for 5s
    ## On second leg, go up at 10 degrees pitch
    ## On 4th leg, go down at 10 degrees pitch
    # Repeat 4 times
    ## Right turn for 6s at 15 deg/sec
    ## Forward for 6s
    ## On second leg, go down at 8 degrees pitch
    ## On 4th leg, go up at 8 degrees pitch
    print('Beginning .. positions[0]',sim.positions[0])
    V = 5
    # lambda function to convert seconds to number of timesteps
    num_steps = lambda x : int(round(x/params['dt']))
    for i in range(4):
        #Left turn
        sim.propagate(V,radians(-30.),0, num_steps(3))
        # Forward for 5 seconds
        if i==0:
            sim.propagate(V, 0, radians(10), num_steps(1))
            sim.propagate(V,0,0,num_steps(3))
            sim.propagate(V, 0, radians(-10), num_steps(1))
        elif i==2:
            sim.propagate(V,0,radians(-15), num_steps(1))
            sim.propagate(V,0,0,num_steps(3))
            sim.propagate(V,0,radians(15), num_steps(1))
        else:
            sim.propagate(V,0,0,num_steps(5))
    print('sim.positions[0] is ',sim.positions[0])
    print('positions[1]',sim.positions[1])
    plt.plot(sim.positions[:,0])
    plt.plot(sim.positions[:,1])
    plt.plot(sim.positions[:,2])
    plt.title('positions vs. time')
    
    plt.figure()
    plt.plot(sim.velocities[:,0])
    plt.plot(sim.velocities[:,1])
    plt.plot(sim.velocities[:,2])
    plt.title('velocities')

    plt.figure()
    R_vecs = np.zeros((sim.n_pos,3))
    for i in range(sim.n_pos):
        R_vecs[i] = R2v(sim.b_C_ns[i])
    plt.plot(R_vecs)

    plt.figure()
    plt.plot(sim.gyros)
    plt.title('gyros')

    plt.figure()
    plt.plot(sim.accels)
    plt.title('accels')
    plt.show()
    # for i in range(4):
    #     #Right turn
    #     sim.propagate(V,radians(15.),0,num_steps(6))
    #     if i==1:
    #         sim.propagate(V, 0, radians(-7), num_steps(1))
    #         sim.propagate(V, 0, 0, num_steps(4))
    #         sim.propagate(V, 0, radians(7), num_steps(1))
    #     elif i==3:
    #         sim.propagate(V,0,radians(12), num_steps(1))
    #         sim.propagate(V,0,0,num_steps(4))
    #         sim.propagate(V,0,radians(-12), num_steps(1))
    #     else:
    #         sim.propagate(V,0,0,num_steps(6))
    file = open('sim_data.pkl','wb')
    pkl.dump(sim,file)

