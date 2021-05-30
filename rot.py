'''This module has a bunch of different functions that deal with
rotations in 3-D space.  It has no internal storage, so no class.

Some notes:
* Just for consistency, axes are labeled 0, 1, and 2.
* All angles are in radians unless otherwise specified

This module has it own custom exception, `InvalidDCM` that is thrown
if the user passes in stuff that is invalid for a DCM (e.g., an invalid 
axis combination for euler_angles)
'''
import numpy as np
import math as m

class InvalidDCM(Exception):
    pass

def axis0(angle, degrees=False):
    '''Creates a DCM to rotate about axis 0 by angle

    Args:
        angle -- how far to rotate about the axis
        degrees -- whether angle should be in degrees (true) or radians (false, the default)
    
    Returns:
        A 3x3 numpy matrix that performs the rotation
    '''
    if degrees:
        #i_ = internal
        i_angle = m.radians(angle)
    else:
        i_angle = angle

    return np.array([[1., 0., 0.],[0., m.cos(i_angle), m.sin(i_angle)],[0., -m.sin(i_angle), m.cos(i_angle)]])

def axis1(angle, degrees=False):
    '''Creates a DCM to rotate about axis 1 by angle

    Args:
        angle -- how far to rotate about the axis
        degrees -- whether angle should be in degrees (true) or radians (false, the default)
    
    Returns:
        A 3x3 numpy matrix that performs the rotation
    '''
    if degrees:
        #i_ = internal
        i_ang = m.radians(angle)
    else:
        i_ang = angle

    return np.array([[m.cos(i_ang),0.,-m.sin(i_ang)],[0.,1.,0.],[m.sin(i_ang),0.,m.cos(i_ang)]])

def axis2(angle, degrees=False):
    '''Creates a DCM to rotate about axis 2 by angle

    Args:
        angle -- how far to rotate about the axis
        degrees -- whether angle should be in degrees (true) or radians (false, the default)
    
    Returns:
        A 3x3 numpy matrix that performs the rotation
    '''
    if degrees:
        #i_ = internal
        i_ang = m.radians(angle)
    else:
        i_ang = angle

    return np.array([[m.cos(i_ang),m.sin(i_ang),0.],[-m.sin(i_ang),m.cos(i_ang),0.,],[0.,0.,1.]])

def euler_angles(angles, axes, degrees=False):
    '''This creates a DCM that combines three angles, in the order specified by axes.
    The angle in location and axis [0] is applied first.

    Args:
        angles -- How far each rotation should happen
        axes -- indicies for which axis (0,1,2) should be rotated around

    Returns:
        A 3X3 numpy matrix that represents the rotation

    Raises:
        InvalidDCM
    '''
    if axes[0]==axes[1] or axes[1]==axes[2]:
        raise InvalidDCM('Two adjacent axes cannot be the same when using euler_angles')

    #fc = function_chooser
    fc = {0: axis0, 1: axis1, 2: axis2}
    
    going_out = np.identity(3)
    for ii in range(3):
        going_out = fc[axes[ii]](angles[ii],degrees).dot(going_out)

    return going_out

