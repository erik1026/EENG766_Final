import numpy as np
import cv2
import math as m
import random as r
import scipy.linalg as la
import scipy.sparse as sp
import visual as vis

#Thanks to Stephen Dunlap for the following 2 assert functions!!!
def assert_np_matrix(m, shape):
    '''Verify that the variable is an np.ndarray, has the correct dimensionality, 
    and is of the correct shape. Raise an exception if it fails any test.

    Args:
        m: Matrix to test
        shape (tuple(int,int)) or (tuple(int,)): Required shape (rows, cols). 
            If one element is 0, ignores that dimension (allows for a 3XN array, where
            you don't know what N is at compile-time)

    Raises:
        ValueError: Wrong type, dimensionality or size
    '''

    if not type(m) is np.ndarray:
        raise ValueError('Wrong type', 'matrix must be np.ndarray')

    if not len(shape) == len(m.shape):
        raise ValueError('Wrong dimensions', 
                         f'array must be {len(shape)}-dimensional, got {len(m.shape)}-d array')

    if len(shape)==1:
        if shape[0]!=0 and m.shape[0] != shape[0]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
    else:
        if not shape[0] and not m.shape[1] == shape[1]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
        
        if not shape[1] and not m.shape[0] == shape[0]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
        
        if shape[0] and shape[1] and not m.shape == shape:
            #print('ERROR: set_translation(): w_T_c must be a 1x3 matrix')
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')

def assert_np_matrix_choice(m, shapes):
    '''
    Take in m and see if it is one of a list of shapes

    Args: 
        m:  (numpy) matrix to test
        shapes: a list of tuples describing shapes.  Each
            must be either 1-D or 2-D (rows, cols)

    Raises:
        ValueError: Wrong type, dimensionality or size
    '''
    if not type(m) is np.ndarray:
        raise ValueError('Wrong type', 'matrix must be np.ndarray')

    passed = False
    for shape in shapes:
        try:
            assert_np_matrix(m, shape)
        except ValueError:
            pass
        else:
            passed = True
            break
    if not passed:
        raise ValueError('Wrong shape', f'matrix is {m.shape}, but needs to be one of {shapes}')

def assert_np_3vec(m):
    '''Alias for assert_np_matrix_choice(m, ((3,1),(1,3),(3,)) )'''
    assert_np_matrix_choice(m, ((3,),(3,1),(1,3)) )

def assert_np_2vec(m):
    '''Alias for assert_np_matrix_choice(m, ((2,1),(1,2),(2,)) )'''
    assert_np_matrix_choice(m, ((2,),(2,1),(1,2)) )

def assert_cam(cam):
    '''
    Assert that the input 'cam' is of type vis.Camera
    
    Raises:  ValueError
    '''
    if not type(cam) is vis.Camera:
        raise ValueError('Wrong type', 'input must be of type vis.Camera')

#Not really sure if this function should be here or in vis.rot
#Could raise InvalidDCM in vis.rot instead...?
def assert_3D_dcm(M):
    '''
    Takes in M and makes sure it is a 3D direction cosine matrix.

    Raises: ValueError
    '''
    assert_np_matrix(M,(3,3))
    if not m.isclose(np.linalg.det(M), 1.0):
        raise ValueError('Not DCM', 'input must have determinant = 1.0')
    for i in range(3):
        if not m.isclose(np.linalg.norm(M[i]), 1.0):
            raise ValueError('Not DCM', f'row {i} of M is not unit length')
        for j in range(i+1,3):
            if not m.isclose(M[i].dot(M[j]), 0.0, abs_tol=1E-9):
                raise ValueError('Not DCM',f'rows {i} and {j} are not orthogonal')


def create_skew_symm_mat ( axis_rots ):
    '''
    Creates a 3x3 skew symmetric matrix from a 3-vector. Note that this is
    the skew-symmetric for differential rotations about the axis, _not_ the
    matrix that allows you to do a vector cross-product. (Though it can be 
    used to create an F or E matrix because the sign doesn't matter...)

    Args:  
    axis_rots can be a list, tuple, or ndarray of one dimension.
        Anything that allows it to reference its elements as axis_rots[i], 
        where i is 0, 1, or 2

    Returns:  3x3 numpy array
    '''
    assert(len(axis_rots)==3 and axis_rots.size == 3)
    
    return np.array([[0., axis_rots[2], -axis_rots[1]],
                     [-axis_rots[2], 0., axis_rots[0]], 
                     [axis_rots[1], -axis_rots[0], 0.]])


def imshow_reduced(window_name, img, max_width=800, max_height=None):
    '''This function is an extension to cv2.imshow. If the passed in image
    is too big, it will be downsized (by factors of 2) until its width is less
    than "max_width".  If max_height is passed in, will also check the displayed
    image is smaller than max_height.  By default, no check of the height occurs.
    
    Need to use cv2.waitKey after calling this function (just like cv2.imshow)

    Args:
        First two args same as cv2.imshow
        window_name:  string that will be on the top of the window
        img: cv2 image structure (really a numpy array)
        max_width:  maximum width of image before it shows
        max_height:  (if used), maximum height of image before it shows
    '''

    im_shape = img.shape
    tmp_im = img
    while im_shape[1] > max_width:
        tmp_im = cv2.pyrDown(tmp_im)
        im_shape = tmp_im.shape 
    if max_height is not None:
        while im_shape[0] > max_height:
            tmp_im = cv2.pyrDown(tmp_im)
            im_shape = tmp_im.shape 

    cv2.imshow(window_name, tmp_im)

def create_random_world_points(cam, num_pts, dist_range = None):
    '''Create num_pts random world locations within the view of the camera
    at location w_t_cam and with orientation c_R_w.  dist_range defines how
    far away from the camera the points can be.  Default is between 10 and 60 meters.

    Args:
        cam:  A vis.Camera object
        num_pts: How many (random) points to create
        dist_range:  A tuple specifying the range the points should be in away from the camera (default (10,60))

    This function returns a tuple of the world points and their pixel locations
    '''
    if dist_range is None:
        dist_range = (10.,60.)
    diff_dist = dist_range[1] - dist_range[0]

    d_rand = np.random.rand(num_pts)*diff_dist + dist_range[0]

    #Create some random image locations
    x_rand = np.random.rand(num_pts,2)
    x_rand[:,0] *= cam.im_size[0]
    x_rand[:,1] *= cam.im_size[1]
    x_d_tuples = list((x_rand[ii],d_rand[ii].item()) for ii in range(num_pts))
    #Took random image points and projected them into space
    return (cam.backproject_points(x_d_tuples), x_rand)

def create_random_camera(K=None, loc_range = (-200,200), angle_range=(-m.pi,m.pi)):
    '''
    This function creates a vis.Camera object at a random location and with a random
    orientation.  Both are uniformly distributed in the ranges passed in.

    Args:
    K: A calibration matrix.
    loc_range:  x,y, and z limits on where the camera can be.  Default is (-200,200)
    angle_range:  limits on rotation angles.  Default is (-pi,pi)

    Returns:  A vis.Camera object
    '''
    loc_dist = loc_range[1] - loc_range[0]
    angle_dist = angle_range[1] - angle_range[0]
    w_cam = np.random.rand(3)*loc_dist + loc_range[0]
    angles = np.random.rand(3)*angle_dist + angle_range[0]
    my_R = vis.rot.euler_angles(angles,[2,1,0])
    if K is not None:
        assert_np_matrix(K, (3,3))
        return vis.Camera(K,my_R,w_cam)
    else:
        return vis.Camera(c_R_w=my_R, w_cam=w_cam )

def create_circular_scenario(radius=100, n_cams = 72, n_points = 400, K=None):
    '''
    This function spaces cameras out around a circle, all looking in at
    the center of the circle.  The circle will be in the "x-z" plane.

    Args:
        radius:  The radius of the circle on which the cameras will lie (default=100)
        n_cams:  number of cameras on the circle.  Must be >= 8 (default=72)
        n_points: Number of points to create in scenario.  Will create the number
            passed in rounded up to the closest value dividable by 8. (default=400)
        K: Calibration matrix for the cameras to be created.  If None, will use the 
            default K in the Camera class. (default = None)
        
    Returns: A tuple with a list of Camera objects and an Nx3 array of world points.
        Note that camera 0 will be at the origin and rotation will be identity.
    '''

    #First, set up the cameras.
    #Must be >= 8 for later code to work...
    assert n_cams >= 8

    angle_cams = 2*m.pi/n_cams

    #Now to generate the cameras
    cams = []
    for ii in range(n_cams):
        cam_loc = radius*np.array([m.cos(angle_cams*ii-m.pi/2), 0., m.sin(angle_cams*ii-m.pi/2)]) \
            + np.array([0.,0., radius])
        cam_rot = vis.rot.axis1(-angle_cams*ii)
        cams.append(vis.Camera(K, cam_rot, cam_loc))

    #%% Create the world points.  This will find world points from
    #8 different cameras set up (roughly) evenly around the circle
    # If n_cams is divisible by 8, it will be at (0, 45, 90, etc. degrees)

    #Used for generating points
    min_dist = .15*radius
    max_dist = 2*radius-min_dist
    n_pts_per_cam = m.ceil(n_points/8) #For 400 total points
    cam_step = m.floor(n_cams/8)

    #Actually generate the points
    w_pts = np.zeros((n_pts_per_cam*8,3))
    for ii in range(8):
        curr_cam = cams[ii*cam_step]
        #Note the [0] at the end to get world points (I am ignoring pixel points for now)
        w_pts[ii*n_pts_per_cam:(ii+1)*n_pts_per_cam] = \
            vis.utils.create_random_world_points(curr_cam, n_pts_per_cam, (min_dist,max_dist))[0]
        
    return (cams,w_pts)

def create_feats(cam, w_pts, found_percent=.8, rand_add_percent=.2, bad_desc_percent = .1,  sd_noise=1):
    '''
    This takes in the true world points and creates cv2.KeyPoint
    values for all that project to within the image as specified by
    the camera passed in.  Noise and problems are all specified by the
    parameters passed in.  It adds spurious points (not corresponding 
    to any world point) as determined by rand_add_percent 
    perc_outliers . It also adds noise to the projected points.  The 
    "descriptors" currently passed out are the world point it corresponds
    to, or a negative number if it was an outlier.

    Args:
        cam: A vis.Camera object used to project the world points
        w_pts: The true world points
        percent_found:  How many of the features that project in the image to "find"
            and make actual features (default = .8)
        rand_add_percent: How many completely spurious points (not corresponding 
            with any world points) to add.  1= the same number as there are inliers.  
            .5 is half as many as inliers.  Actual number added is a random value.
            (default = .2)
        bad_desc_percent: How many features (probabilisticly) to assign the wrong 
            descriptor to. This only permutes within the features found in the image,
            so it won't pull in some other random "descriptor"  (default = .05)
        sd_noise: Standard deviation of the (Gaussian isotropic) noise to add to 
            pixel locations of the inliers (default = 1)
    
    Returns: Two lists.  One of cv2.KeyPoint values, the second of "descriptors"
    '''

    #This projects all points
    p_Xs = cam.project_points(w_pts)
    #only keep the ones that are in the image
    in_feats=[]
    in_desc=[]
    for i,p_x in enumerate(p_Xs):
        if cam.in_image(p_x) and r.random() < found_percent:
            in_feats.append(p_x)
            in_desc.append(i)
    in_feats = np.array(in_feats)
    in_feats = in_feats + np.random.randn(*in_feats.shape) * sd_noise
    in_kps = [cv2.KeyPoint(p_x[0],p_x[1],1.) for p_x in in_feats]

    if rand_add_percent > 0.:
        size_outliers = m.floor(rand_add_percent)*len(in_feats)
        my_percent = rand_add_percent - m.floor(rand_add_percent)
        #Have some randomness in the number of outliers...
        size_outliers += \
            np.count_nonzero(np.random.choice([False,True], 
                         size=(len(in_feats),), 
                         p=[1-my_percent, my_percent]))
        out_feats = np.random.rand(size_outliers,2)*cam.im_size
        out_kps = [cv2.KeyPoint(p_x[0],p_x[1],1.) for p_x in out_feats]
        in_kps.extend(out_kps)
        in_desc.extend( -np.arange(1,size_outliers+1) )

    in_desc=np.array(in_desc)
    if bad_desc_percent >= 0.:
        to_shuffle_idx = np.random.rand(len(in_desc)) <= bad_desc_percent
        in_desc[to_shuffle_idx] = np.random.permutation(in_desc[to_shuffle_idx])


    combined = list(zip(in_kps, in_desc))
    r.shuffle(combined)
    return tuple(zip(*combined))

def inv_sparse_block_diagonal(A, blk_size):
    '''
    Takes in a matrix that is block diagonal, with each 
    block being of size blk_size X blk_size.  This can be a sparse matrix as
    long as you can index into it (i.e., A[0:blk_size,0:blk_size] is valid).
    Outputs the inverse as a scipy.sparse.bsr_matrix.

    Warning:  This does nothing to check that A is actually block diagonal,

    Args:
        A: The block diagonal matrix to invert
        blk_size:  The size of each block diagonal matrix

    Returns a scipy.sparse.bsr_matrix of the same size as A
    '''
    assert A.shape[0]==A.shape[1]
    assert (A.shape[0] % blk_size) == 0
    assert (A.shape[1] % blk_size) == 0
    assert sp.issparse(A)
    
    num_blks = int(round(A.shape[1]/blk_size))

    data = np.zeros((num_blks,blk_size,blk_size))
    idx = np.arange(num_blks)
    idx_ptr = np.arange(num_blks+1)

    for ii in range(num_blks):
        tmp = A[ii*blk_size:(ii+1)*blk_size,ii*blk_size:(ii+1)*blk_size]
        data[ii] = la.inv(tmp.toarray()) #No sparse-ness attempted here...

    return sp.bsr_matrix((data,idx,idx_ptr))

def align_pts(true_pts, test_pts):
    '''
    This function takes in two sets of points and scales, rotates, and translates
    the "test_pts" to best align with the true_pts.  Note that these are locations
    in the world, not full state vectors

    This is algorithm 1 in "A tutorial on quantitative trajectory evaluation for 
    visual(-inertial) odometry" by Zichao Zhang and David Scaramuzza

    Args:
        true_pts -- set of points to align to
        test_pts -- set of points to align with true_pts
    
    Returns:  A scaled, rotated, and translated version of test_pts
    '''
    assert type(true_pts) is np.ndarray
    assert type(test_pts) is np.ndarray
    assert true_pts.shape==test_pts.shape

    mu_true = np.mean(true_pts,0)
    mu_test = np.mean(test_pts,0)
    # sig_true = np.var(true_pts)  #Not used later on
    sig_test = la.norm(test_pts-mu_test)**2.
    Sigma = np.zeros((true_pts.shape[1],true_pts.shape[1]))
    for true_v,test_v in zip(true_pts,test_pts):
        Sigma += np.outer(true_v-mu_true,test_v-mu_test)
    U,D,Vh = la.svd(Sigma)
    W = np.eye(true_pts.shape[1])
    if la.det(U)*la.det(Vh) < 0:
        W[-1,-1] = -1
    R = U.dot(W.dot(Vh))
    s = np.sum(D.dot(W))/sig_test #D is a vector, not a matrix from SVD
    # s=1
    t = mu_true - s * R.dot(mu_test)
    #R, s, and t found.  Now going_out = s*R*test + t
    return s*test_pts.dot(R.T) + t #Flipped the transpose rather than transposing test_pts then transposing again


## Some robust cost functions
class GemanMcClure:
    def __init__(self, s_sq=9):
        self.s_sq = s_sq
        self.count = 0

    def __call__(self, res):
        dist_sq = la.norm(res)**2
        if dist_sq > self.s_sq:
            self.count += 1
        return 1/(dist_sq/self.s_sq + 1)
    
    def resetCount(self):
        self.count = 0
    
    def getCount(self):
        return self.count

class Huber:
    def __init__(self, s=1):
        self.s = s
        self.s_sq = s*s

    def __call__(self, res):
        dist_sq = la.norm(res)**2
        if dist_sq <= self.s_sq:
            return 1.
        else:
            return m.sqrt(self.s/m.sqrt(dist_sq))
#### End robust cost functions
