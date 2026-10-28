"""Retiming of a fixed joint path using explicit engineering review limits."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d

def retime_path(q, dt, speed_limit, acceleration_limit, time_scale=1.0):
    """Preserve path waypoints; stretch local time until sampled derivatives pass.

    Returns resampled q and its positions on the ORIGINAL time axis. This is
    numerical trajectory preparation, not a guarantee of physical feasibility.
    """
    if not np.isfinite(time_scale) or time_scale<1:raise ValueError('time_scale must be finite and >= 1')
    original=np.arange(len(q))*dt
    intervals=np.maximum(dt,1.15*np.max(np.abs(np.diff(q,axis=0)),axis=1)/speed_limit)
    intervals=maximum_filter1d(intervals,size=7,mode='nearest')
    for _ in range(30):
        nodes=np.r_[0,np.cumsum(intervals)]
        count=int(np.ceil(nodes[-1]/dt))
        # Scale nodes to an exact multiple of control period, then sample at 50 Hz.
        nodes*=count*dt/nodes[-1]
        times=np.arange(count+1)*dt
        result=PchipInterpolator(nodes,q)(times)
        vel=np.gradient(result,dt,axis=0,edge_order=2)
        acc=np.gradient(vel,dt,axis=0,edge_order=2)
        factor=np.maximum(np.max(np.abs(vel),axis=1)/speed_limit,np.sqrt(np.max(np.abs(acc),axis=1)/acceleration_limit))
        if factor.max() <= 1.001:
            if time_scale>1:
                # Resample the ORIGINAL path, retaining transition knots. A
                # second interpolation of the already sampled path leaks motion
                # across boundaries where a joint must become exactly fixed.
                count=int(np.ceil(nodes[-1]*time_scale/dt))
                nodes*=count*dt/nodes[-1]
                times=np.arange(count+1)*dt
                result=PchipInterpolator(nodes,q)(times)
            return result,PchipInterpolator(nodes,original)(times)
        factor=maximum_filter1d(np.maximum(factor,1),size=9,mode='nearest')
        midpoint=(nodes[:-1]+nodes[1:])/2
        intervals=np.diff(nodes)*np.maximum(1,np.interp(midpoint,times,factor)*1.03)
    raise RuntimeError('Retiming did not meet sampled velocity/acceleration limits')
