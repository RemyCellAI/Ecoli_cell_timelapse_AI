# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:25:38 2022

@author: http://juddzone.com/ALGORITHMS/least_squares_ellipse.html
"""

import numpy as np

def ls_ellipse(xx,yy):

   eansa = 0
   # change xx from vector of length N to Nx1 matrix so we can use hstack
   x = xx[:,np.newaxis]
   y = yy[:,np.newaxis]

   J = np.hstack((x*x, x*y, y*y, x, y))
   K = np.ones_like(x) #column of ones

   JT=J.transpose()
   JTJ = np.dot(JT,J)
   mdet = np.linalg.det(JTJ)
   if mdet > 0.1: # Check if the determinant is non-zero
       InvJTJ=np.linalg.inv(JTJ);
       ABC= np.dot(InvJTJ, np.dot(JT,K))
    
       # ABC has polynomial coefficients A..E
       # Move the 1 to the other side and return A..F
       # A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
       eansa=np.append(ABC,-1)
       flag = 1
   else:
       flag = 0
   return eansa, flag




from numpy.linalg import eig, inv
def polyToParams(v,printMe = False):

   # convert the polynomial form of the ellipse to parameters
   # center, axes, and tilt
   # v is the vector whose elements are the polynomial
   # coefficients A..F
   # returns (center, axes, tilt degrees, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   Amat = np.array(
   [
   [v[0],     v[1]/2.0, v[3]/2.0],
   [v[1]/2.0, v[2],     v[4]/2.0],
   [v[3]/2.0, v[4]/2.0, v[5]    ]
   ])

   # if printMe: print( '\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A2=Amat[0:2,0:2]
   A2Inv=inv(A2)
   ofs=v[3:5]/2.0
   cc = -np.dot(A2Inv,ofs)
   # if printMe: print '\nCenter at:',cc

   # Center the ellipse at the origin
   Tofs=np.eye(3)
   Tofs[2,0:2]=cc
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   # if printMe: print( '\nAlgebraic form translated to center\n',R,'\n')

   R2=R[0:2,0:2]
   s1=-R[2, 2]
   RS=R2/s1
   (el,ec)=eig(RS)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   # if printMe: print('\nAxes are\n',axes  ,'\n')

   rads=np.arctan2(ec[1,0],ec[0,0])
   deg=np.degrees(rads) #convert radians to degrees (r2d=180.0/np.pi)
   # if printMe: print( 'Rotation is ',deg,'\n')

   inve=inv(ec) #inverse is actually the transpose here
   # if printMe: print( '\nR1otation matrix\n',inve)
   return (cc[0],cc[1],axes[0],axes[1],deg,inve)


   
