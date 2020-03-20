#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:33:28 2019

@author: lingsr
"""

        
class TrackableObject:
    def __init__(self , objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False
        self.flag = None