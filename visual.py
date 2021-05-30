'''This class maintains all the (geometric) information to define a camera and how
objects in the world will be project into the camera
'''

import numpy as np
import scipy.linalg as la
import math as m
from utils import assert_np_matrix, assert_np_3vec, assert_np_2vec
from utils import assert_3D_dcm, create_skew_symm_mat, assert_cam
import rot as rot
import copy

class Camera:
    c_R_w = np.eye(3)
    w_cam = np.zeros( (3,) )
    K = np.array([[1000., 0., 500.],[0,1000.,500.],[0.,0., 1.]])
    im_size = (1000,1000)
    _P = np.hstack( (K, np.zeros((3,1))) )

    def __init__( self, K=None, c_R_w = None, w_cam = None, im_size = None ):
        '''If im_size is not defined, it will be assumed as 2* the last
        column of K

        Args:
            K:  3x3 intrinsic calibration matrix (default = [1000,0,500; 0,1000,500; 0,0,1])
            c_R_w: 3x3 DCM matrix going from world to camera frame (default = I)
            w_cam: location of the camera in the world frame (default = 0)
            im_size: tuple with size of image (width, height).  (default = 2 times center point from K)

        Raises:
            ValueError:  Wrong type on inputs
        '''
        if not K is None:
            assert_np_matrix(K,(3,3))
            self.K = K
            #default behavior for im_size
            self.im_size=(int(self.K[0,2]*2),int(self.K[1,2]*2))
        if not c_R_w is None:
            assert_3D_dcm(c_R_w)
            self.c_R_w = c_R_w
        if not w_cam is None:
            assert_np_3vec(w_cam)
            self.w_cam = w_cam.reshape((3,))
        if not im_size is None:
            self.im_size= (int(im_size[0]), int(im_size[1]))
        self.update_P()
    
    def update_P(self):
        '''If c_R_w, w_cam, or K are modified directly, this should be called afterwards
        If instead, the user calls set..., it is called automatically

        Returns: the new P matrix (in addition to setting it inside the class)
        '''

        #HINT!  if the K matrix is the identity matrix, then self._P.dot(x) should do the exact
        #same thing as self.transform_points(x)
        
        t_tmp = -self.c_R_w.dot(self.w_cam.reshape( (3,1) ) )
        self._P = self.K.dot(np.hstack( (self.c_R_w, t_tmp) ) )
        return self._P
    
    def get_P(self):
        return self._P

    def set_rot(self, c_R_w):
        assert_3D_dcm(c_R_w)
        self.c_R_w = c_R_w
        self.update_P()

    def set_K(self, K):
        assert_np_matrix(K,(3,3))
        self.K = K
        self.update_P()
    
    def set_loc(self, w_cam):
        '''sets the location of the camera in world coordinates'''
        assert_np_3vec(w_cam)
        self.w_cam = w_cam.reshape((3,))
        self.update_P()

    def in_image(self, pt, buffer=None, my_eps = 1E-10 ):
        ''' A quick binary check to decide if pt (a pixel location) is within the image.
        This uses the "im_size" stored within the camera class.  If buffer is non-zero, 
        then adds a buffer to the image which it cannot be within.
        
        Args:
            pt is any data structure that can reference [0] and [1]
            buffer is the size (in pixels) that an object cannot be in close to the edge of the image
            my_eps is for numerical comparisons (i.e. anything > -my_eps will be in the left of the image)

        Returns:  a boolean

        '''
    
        if buffer is None:
            buffer=-my_eps
        else:
            buffer -= my_eps
        
        return pt[0] >=buffer and (pt[0] <= (self.im_size[0] - buffer)) \
            and pt[1] >= buffer and pt[1] <= (self.im_size[1] - buffer)
    
    def transform_points(self, w_X):
        '''
        Takes either a list of points or single point and transforms them into 
        _camera_ space.  Note that no projection takes place. Just a rigid body
        transform to say where a point is w.r.t. the camera's reference frame

        Args: w_X -- a single point (3vec) or list (or np.array) of world (3-vec) points

        Returns: c_X -- a list or single points of camera (3-vec) points
        '''
        tf_pt = lambda R,t,X : R.dot(X.reshape((3,)) - t)

        if type(w_X) is np.ndarray and len(w_X.shape)==1:
            assert_np_matrix(w_X, (3,) )
            return tf_pt( self.c_R_w, self.w_cam, w_X )

        for w_x in w_X:
                assert_np_3vec(w_x)

        N = len(w_X)
        c_X = np.zeros((N,3))
        for ii,w_x in enumerate(w_X):
            c_X[ii]= tf_pt( self.c_R_w, self.w_cam, w_x )

        for c_x in c_X:
            assert_np_3vec(c_x)
        return c_X

    def project_points(self, w_X):
        '''
        Takes in a point or a list (array, etc) of points in world coordinates and
        returns their location in the camera.  Note that there is _no_ checking if 
        the pixels would actually be in the image, in front of the camera, etc.)
        
        Args:
            w_X -- list of world locations of points to be projected.  Each point must 
            be a numpy ((3,)) vector
        
        Returns:
            z a Nx2 numpy array, where N is the number of points passed in.
        '''

        #The single point function
        if type(w_X) is np.ndarray and len(w_X.shape)==1:
            assert_np_matrix(w_X, (3,) )
            Rx = self._P[:,:3].dot(w_X)
            Px = Rx + self._P[:,3]
            return Px[:2]/Px[2]

        #Iteration

        for w_x in w_X:
            assert_np_matrix(w_x, (3,) )

        N = len(w_X)
        going_out = np.zeros((N,2))
        P = self._P
        bigger_x = np.ones((4,))
        for ii in range(N):
            bigger_x[:3]=w_X[ii]
            tmp = P.dot(bigger_x)
            going_out[ii] = tmp[0:2] / tmp[2]

        N = len(w_X)
        assert_np_matrix(going_out,(N,2))
        return going_out

    def backproject_points(self, p_x_and_dist_tuples):
        '''  
        This function takes in a list of tuples containing a point in the pixel plane
        and a "z" distance.  It returns where that point would be in the real world 
        given the current camera parameters:

        Args:
            p_x_and_dist_tuples -- a list of 2-vector x's giving location in the pixel
                plane and a "z" distance associated with each one

        Returns:  An Nx3 np array with a world points in each row
        '''
        for tup in p_x_and_dist_tuples:
            assert_np_2vec(tup[0])

        N = len(p_x_and_dist_tuples)
        going_out = np.zeros((N,3))
        K_inv = la.inv(self.K)
        for ii,tup in enumerate(p_x_and_dist_tuples):
            x = tup[0]
            tmp = K_inv.dot(np.hstack( (x.reshape((2,)), 1.) ) )
            going_out[ii] = self.c_R_w.T.dot(tmp) * float(tup[1]) + self.w_cam

        assert_np_matrix(going_out,(N,3))
        return going_out

    def rotate_at_point(self, w_point, random_roll=False, roll=None):
        '''Takes a point in the world and rotates the camera such that the 
        point is in the center of the image.  Can then rotate about the imaging 
        axis or set that "roll" to a specific value.

        Args:
            w_point -- position of the point to be looked at in world coordinates
            random_roll -- Create a random roll or leave it at 0 (default = false)
            roll -- Specific roll, in radians, to use (assumes random_roll = false)
        
        Returns:
            Rotation matrix that c_R_w is set to internally as well
        '''
        assert_np_3vec(w_point)
        if len(w_point.shape)==2:
            w_point =w_point.reshape((3,))
        
        #Find the vector between the two:
        pointing_vec = w_point - self.w_cam
        pointing_vec /= la.norm(pointing_vec) #pointing_vec is now unit length
        pitch = -m.asin(pointing_vec[1])
        yaw = m.atan2( pointing_vec[0], pointing_vec[2] )
        point_at_R_w = rot.axis0(pitch).dot(rot.axis1(yaw))
        if roll is None:
            roll = 0
        if random_roll:
            roll = ((np.random.rand(1))*2*m.pi - m.pi).item(0)
        
        self.set_rot(rot.axis2(roll).dot(point_at_R_w) )
        return self.c_R_w

    def points_from_line_normal(self, n):
        '''This returns points on the edge of the image that intersect with
        the line normal that is passed in

        Args:  n -- the normal to the line in image (calibrated) coordinates. 
            Should be a 3-element numpy vector

        Returns:  a list with either 0 or 2 pixel locations.  Each pixel location
        should be a 2-element numpy vector
        '''
        assert_np_3vec(n)
        n = n.reshape((3,))

        ns=[]
        ns.append( np.array([1., 0., 0.]) ) #x=0
        ns.append( np.array([0., 1., 0.]) )#y=0
        ns.append( np.array([1/self.im_size[0], 0., -1.]) ) #x=width
        ns.append( np.array([0., 1/self.im_size[1], -1.]) ) #y=width

        intersect_in_image = []
        intersects = []
        n_intersects = 0
        for n_e in ns:
            intersect = np.cross( n_e, n )
            if abs(intersect[2]) < 1E-11:
                intersect_in_image.append(False)
                intersects.append(None)
            else:
                tmp_intersect = self.in_image(intersect / intersect[2])
                if tmp_intersect: 
                    n_intersects += 1
                    norm_intersect = intersect / intersect[2]
                    intersects.append( norm_intersect[:2] )
                else:
                    intersects.append( None ) 
                intersect_in_image.append( tmp_intersect )
        going_out=[]
        if n_intersects <= 2: #Should be either 2 or 0
            for i,tst in enumerate(intersect_in_image):
                if tst: going_out.append(intersects[i])
        elif n_intersects == 3: #It passes through one corner
            #Need to find (and return) the two edges that are opposite
            if intersect_in_image[0] and intersect_in_image[2]:
                going_out = [intersects[0], intersects[2]]
            else:
                going_out= [intersects[1], intersects[3]]
        else:  #has 4 intersects / passes through 2 corners
            going_out= [intersects[0], intersects[2]]

        going_out = np.array(going_out)
        assert_np_matrix(going_out, (0,2))
        return going_out

    def relative_move(self, c_move):
        '''
        Take a vector in the camera coordinate system (c_move) and 
        move the camera that way.  This allows for relative movement from
        the current camera location

        Args:
            c_move:  the "movement" vector in the camera coordinate system

        Returns:  the new w_cam value (plus sets it in the class)
        '''
        assert_np_3vec(c_move)

        w_move = self.c_R_w.T.dot( c_move.reshape((3,)) )
        self.set_loc (self.w_cam + w_move)
        return self.w_cam

    def backup_to_view(self, w_points, pixel_buffer=0):
        '''Takes in a list of points and "backs up" the camera until all points 
        are in the field of view of the camera.  Note that this modifies the 
        w_cam internal state of the function, but nothing else.

        Args: 
        w_points -- a list of 3-element numpy arrays that represent points
           in the world coordinate frame
        pixel_buffer -- If you want points not only "in" the image, but in at least
            a number of pixels, you can set this value.  Default=0

        Returns:  the new w_cam.  Plus the object itself is modified
        '''
        for w_pt in w_points:
            assert_np_3vec(w_pt)

        buffer = pixel_buffer

            #define the target locations for the pixels if outside the FoV
        t_top = buffer
        t_left = buffer
        t_right = self.im_size[0] - buffer
        t_bottom = self.im_size[1] - buffer

        # The key to this is taking a single point and knowing how far back I
        # need to move the camera to get it in the field of view.  If I can only move
        # backwards, then cycling through all the points and moving backwards if needed
        # will guarantee I have moved back far enough

        #first, make all points be in front of the camera
        cam_pts = self.transform_points(w_points)
        min_z = np.min(cam_pts[:,2] )
        if (min_z < 0.01):
            #move the camera back by this amount
            self.relative_move(np.array([0., 0., min_z-.02]) )

        for pt in w_points:
            im_pt = np.dot(self._P, np.hstack( (pt,1.) ) )
            im_loc = self.project_points([pt])[0]
            if not self.in_image( im_loc, buffer=pixel_buffer ):
                delta_z = 0
                #Test if outside in all four directions and move if needed
                if im_loc[0] < t_left:
                    comp_delta = (im_pt[0] - t_left * im_pt[2])/(t_left - self.K[0,2])
                    delta_z = comp_delta
                if im_loc[0] > t_right:
                    comp_delta = (im_pt[0] - t_right * im_pt[2])/ (t_right - self.K[0,2]) 
                    delta_z = max(delta_z,comp_delta)
                if im_loc[1] < t_top:
                    comp_delta = (im_pt[1] - t_top * im_pt[2])/ (t_top - self.K[1,2]) 
                    delta_z = max(delta_z,comp_delta)
                if im_loc[1] > t_bottom:
                    comp_delta = (im_pt[1] - t_bottom * im_pt[2])/ (t_bottom - self.K[1,2]) 
                    delta_z = max(delta_z,comp_delta)
                #I add just a small fudge factor to account for numerical stuff...
                delta_z +=1E-3
                #Make the move
                self.relative_move( np.array([0., 0., -delta_z]) )

        return self.w_cam

    def get_viz_info(self):
        '''Returns information so the vizualizer can render an image using
        the camera information

        Returns:
            A tuple with the required information, being:
            Camera location, forward vector (imaging axis), up vector,
            smaller field of view (in radians), and the image size tuple.
        '''
        for_vec = self.c_R_w[2]
        up_vec = -self.c_R_w[1]
        fov_width = m.atan2(self.im_size[0]/2.,self.K[0,0])*2.
        fov_height = m.atan2(self.im_size[1]/2.,self.K[1,1])*2.
        #Error check assumptions for vizualizations. Centered & symmetric K
        if self.K[0,1]!= 0 \
            or (self.K[0,2]*2.0 != self.im_size[0] and self.K[0,2]*2.0 != (self.im_size[1]-1)) \
            or (self.K[1,2]*2.0 != self.im_size[1] and self.K[1,2]*2.0 != (self.im_size[1]-1)) \
            or self.K[0,0] != self.K[1,1]:
            print('WARNING:  internal K information does not really lend itself to viz, but get_viz_info was called')
            print('self.K is',self.K)
            print('im_size is',self.im_size)
        min_fov = min(fov_width,fov_height)
        return(self.w_cam, for_vec, up_vec, min_fov, self.im_size)
        
    def proj_deriv_X(self, w_X):
        '''
        This takes the derivative of the project_points function w.r.t. X.

        Args: 
            w_X: a 3-element vector representing the location of a point in the world
                coordinate frame
        
        Returns:  a 2x3 numpy array containing the derivatives of u,v w.r.t. X
        '''

        assert_np_3vec(w_X)
        w_X = w_X.reshape((3,))

        tmp_X = np.ones((4,))
        tmp_X[:3]=w_X 

        cam_X = self._P.dot(tmp_X)
        third_val = cam_X[2]
        uv_unscaled = cam_X[:2]
        front_half = self._P[:,:3]
        going_out = front_half[:2]/third_val - \
            np.outer(uv_unscaled/(third_val*third_val), front_half[2])

        assert_np_matrix(going_out, (2,3))
        return going_out

    dskew = np.array([create_skew_symm_mat(np.array([1., 0., 0])),
                      create_skew_symm_mat(np.array([0., 1., 0])),
                      create_skew_symm_mat(np.array([0., 0., 1.]))])

    def proj_deriv_Rt(self, w_X):
        '''
        This takes the derivative of the project_points function w.r.t. R and t.
        dR is defined as a Skew Symmetric matrix as in utils.create_skew_symm_mat that
        is multiplied _to the right_ of c_R_w in the projection function.  In other words
        c_R_w(k+1) = c_R_w(k).dot(scipy.linalg.expm(utils.create_skew_symm_mat(dR)))
        and t is w_cam of the rotation

        Args:
            w_X: a 3-element vector representing the location of a point in the world
                coordinate frame
        
        Returns: a 2x6-element numpy array containing the derivatives of u,v w.r.t. dR,dt
        '''

        assert_np_3vec(w_X)
        w_X = w_X.reshape((3,))
        
        #doing projection as KR[I|-t]w_X
        dPX_dRt=np.zeros((3,6))
        
        #This code has been optimized for speed, not readability.... :(
        tmp_X = np.ones((4,))
        tmp_X[:3] = w_X

        KR = self.K.dot(self.c_R_w)
        w_diff = w_X - self.w_cam
        PX = self._P.dot(tmp_X)

        for ii in range(3):
            #It = np.hstack((np.eye(3),-self.w_cam.reshape((3,1))))
            dPX_dRt[:,ii] = KR.dot(self.dskew[ii].dot(w_diff))
        dPX_dRt[:,3:]=-KR
        #Now to handle the division to get to u,v
        going_out = np.zeros((2,6))
        going_out[0] = (dPX_dRt[0] - dPX_dRt[2]*PX[0]/PX[2])/PX[2]
        going_out[1] = (dPX_dRt[1] - dPX_dRt[2]*PX[1]/PX[2])/PX[2]

        assert_np_matrix(going_out,(2,6))
        return going_out

    def apply_dRt(self, dRt):
        '''  Apply a delta_Rt value to the camera location and matrices

        Args:  dRt: a 6 element vector with delta R followed by delta t

        Returns: Nothing. Modifies the internal state of this Camera object
        '''
        assert_np_matrix(dRt,(6,))
        dR = la.expm(create_skew_symm_mat(dRt[0:3]))
        self.set_rot(self.c_R_w.dot(dR))
        self.set_loc(self.w_cam + dRt[3:])

    def compute_dRt(self, other_cam):
        '''
        This function should return the dRt that would be used
        to move the Camera class being used to get other_cam.  Note
        that the R part may not be symmetric 
        (cam1.compute_dRt(cam2) != -cam2.compute_dRt(cam1))

        Args: other_cam is van.Camera object

        Returns: (6,) object such that self.apply_dRt should yield the R 
            and t of other_cam
        '''

        assert_cam(other_cam)
        
        going_out = np.zeros((6,))
        #Get the skew symmetric matrix that could have been used
        SS = la.logm(self.c_R_w.T.dot(other_cam.c_R_w))
        going_out[0] = SS[1,2]
        going_out[1] = SS[2,0]
        going_out[2] = SS[0,1]
        going_out[3:] = other_cam.w_cam - self.w_cam

        assert_np_matrix(going_out,(6,))
        return going_out


    @staticmethod
    def refine_PnP(w_X_and_p_X, K, init_R=None, init_t=None, im_size=None):
        '''
        This function returns a Camera class whose position best fits (in a
        least-squared sense) the projection of world points into pixel points.
        Note that this function should implement both optimization on the 
        manifold of rotations _and_ a "trust region" method to handle large
        non-linearities within Gauss-Newton optimization.
        
        Args:
            w_X_and_p_X: A list of tuples.  Each tuple has a point in world 
                coordinates and a corresponding pixel location of that point
            K: The calibration matrix used for projection
            init_R: A 3x3 numpy array that can be the initial rotation for the
                optimization procedure
            init_t: A (3,) numpy array that is the initial location for the 
                optimization procedure
        
        Returns: A van.Camera object at the refined location.
        '''
        #Check the inputs and format them for the GN optimization
        if init_R is None:
            curr_R = np.eye(3)
        else:
            assert_3D_dcm(init_R)
            curr_R = init_R

        if init_t is None:
            curr_t = np.zeros((3,))
        else:
            assert_np_3vec(init_t)
            curr_t = init_t.reshape((3,))

        num_pts = len(w_X_and_p_X)
        world_pts = []
        pixel_pts = []
        for w_X,p_X in w_X_and_p_X:
            assert_np_3vec(w_X)
            assert_np_2vec(p_X)
            world_pts.append(w_X)
            pixel_pts.append(p_X)
        world_pts = np.array(world_pts)
        pixel_pts = np.array(pixel_pts)

        #When to stop iterating
        min_dx = 1E-5
        max_iters = 100

        #initialize the GN procedure
        curr_cam = Camera(K,curr_R,curr_t,im_size)
        est_pts = curr_cam.project_points(world_pts)
        new_diff = pixel_pts - est_pts
        num_iters = 0
        dx = min_dx * 2.0

        #Run the GN
        while num_iters < max_iters and la.norm(dx) > min_dx:
            y = new_diff.flatten()
            J = np.zeros((num_pts*2,6))
            for ii,w_X in enumerate(world_pts):
                J[ii*2:(ii+1)*2]= curr_cam.proj_deriv_Rt(w_X)
            dx = la.lstsq(J,y)[0]
            scalar = 1
            test_ratio = -1
            y_ssd = np.sum(np.square(y))
            #Do the test for if the dx is valid, use a smaller dx if needed
            while (scalar > .0001 and (test_ratio <.25 or test_ratio > 4)):
                est_diff = y - np.dot(J,scalar*dx)
                tmp_cam =copy.copy(curr_cam)
                tmp_cam.apply_dRt(dx)
                new_pts = tmp_cam.project_points(world_pts)
                new_diff = pixel_pts - new_pts
                new_y = new_diff.flatten()
                test_ratio = (y_ssd - np.sum( np.square( new_y ) ) ) / (y_ssd - np.sum( np.square( est_diff ) ) )
                scalar /= 2.0
            if scalar < .0001:
                print('WARNING:  refine_PnP is taking an overly small step size! test_ratio is',test_ratio)
            curr_cam = tmp_cam
            num_iters +=1    

        assert_cam(curr_cam)
        return curr_cam