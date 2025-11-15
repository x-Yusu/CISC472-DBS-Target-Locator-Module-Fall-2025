"""
masks.py
This file contains code for extracting appropriate masks and providing iterators to serve them

Author:
Leopold Ehrlich
"""

def get_masks():
    """
    Provides generator of masks, broadcastable over time dimension
    """
    for i in range(1,4):
        mask = np.zeros((100,100,100))

        start, stop = i*20-5, i*20+5
        mask[start:stop,start:stop,start:stop] = 1

        yield mask[...,None]