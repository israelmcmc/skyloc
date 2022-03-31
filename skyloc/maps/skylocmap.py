from mhealpy import HealpixMap, HealpixBase
import mhealpy as hp

import numpy as np

import matplotlib.pyplot as plt

from .skylocbase import SkyLocBase
import skyloc.plot.axes

from astropy.coordinates import UnitSphericalRepresentation, SkyCoord
import astropy.units as u
from astropy.units import cds

class SkyLocMap(SkyLocBase, HealpixMap):
    """
    Initialize HealpixMap with an attached coordinate frame.

    Args:
        frame (BaseCoordinateFrame): Astropy's coordinate frame. Default: 'ICRS'
        args, kwargs: Passed to HealpixMap
    """

    def pix2skycoord(self, pix):
        """
        Return the sky coordinate for the center of a given pixel
        
        Args:
            pix (int or array): Pixel number
        
        Return:
            SkyCoord
        """
        
        lon,lat = self.pix2ang(pix, lonlat = True)

        skycoord = SkyCoord(lon*u.deg, lat*u.deg, frame = self._frame)

        return skycoord

    def skycoord2pix(self, skycoord):
        """
        Return the pixel containing a given sky coordinate

        Args:
            skycoord (SkyCoord): Sky coordinate

        Return
            int
        """
        
        coord = skycoord.represent_as(UnitSphericalRepresentation)

        pix = self.ang2pix(coord.lon.deg, coord.lat.deg, lonlat = True)

        return pix

    @classmethod
    def error_radius_map(cls,
                         skycoord,
                         sigma = None,
                         nside = None,
                         roi_radius = 5
    ):
        """
        Create a probability map given a source location and error.

        A 2D Gaussian probability distribution is assumed:

        .. math::

            P(r) = \frac{2}{\sigma^2} \exp \left( -r^2 / \sigma^2 \right)

        Where :math:`r` is the angular distance to the specified source location.

        Args:
            skycoord (SkyCoord): Best estimate for the source location
            sigma (Angle or Quantity): Error radius.
            nside (int): Override the nside estimated based on the error radius.
            roi_radius (float): Size of the region of interest --i.e. high-resolution-- 
                as multiple of sigma.
        """

        # Compute appropiate nside        
        sigma_rad = sigma.to_value(u.rad)
            
        if nside is None:
            # Pixel size is 10 times smaller than sigma
            approx_nside = 10*np.sqrt(4*np.pi/12)/sigma_rad
            order = int(np.ceil(np.log2(approx_nside)))
            nside = hp.order2nside(order)

        roi_radius *= sigma

        m = cls.circular_roi(center = skycoord,
                             radius = roi_radius,
                             nside = nside,
                             density = True)
        
        # Fill
        source_ang = skycoord.represent_as(UnitSphericalRepresentation)

        source_vec = hp.ang2vec(source_ang.lon.deg,
                                source_ang.lat.deg,
                                lonlat = True)
        
        pixels_vec = np.array(m.pix2vec(range(m.npix)))
        
        pixels_dist = np.arccos(np.matmul(source_vec, pixels_vec))

        m[:] = np.exp(-pixels_dist*pixels_dist / sigma_rad/sigma_rad)

        # Normalize
        m.density(False)

        m /= np.sum(m)

        return m
        
    @classmethod
    def annulus_map(cls,
                    center,
                    radius,
                    sigma = None,
                    nside = None,
                    roi_width = 5
                    
    ):
        """
        Create a probability for an annulus error.

        The triangulation of a signal by two detector typically results 
        in this pattern.
        
        The followin distribution is distribution is assumed.

        Args:
            skycoord (SkyCoord): Best estimate for the source location
            radius (Angle or Quantity): Angular distance from the center to the middle 
                of the ring.
            sigma (Angle or Quantity): Error on the radius
            cont (float): Alternatively, specify the error as width and containment
                fraction. cont = 0.632121 is equivalent to cont_radius = sigma
            cont_width (Angle or Quantity): Error radius for a given containment
            nside (int): Override the nside estimated based on the error radius.
            roi_width (float): Size of width of the annular region of interest 
                --i.e. high-resolution-- as multiple of sigma.
        """

        # First, get an equivalent single-resolution order, such that the pixel
        # size is smaller than the annulus width
        sigma_rad = sigma.to_value(u.rad)
        
        if nside is None:

            # Pixel size is 10 times smaller than sigma
            approx_nside = 10*np.sqrt(4*np.pi/12)/sigma_rad
            order = int(np.ceil(np.log2(approx_nside)))
            nside = hp.order2nside(order)

        # Region of interest mesh
        
        roi_width *= sigma

        m = cls.annular_roi(center = center,
                            radius = radius,
                            width = roi_width,
                            nside = nside,
                            density = True)

        # Fill
        center_ang = center.represent_as(UnitSphericalRepresentation)

        center_vec = hp.ang2vec(center_ang.lon.deg,
                                center_ang.lat.deg,
                                lonlat = True)
        
        pixels_vec = np.array(m.pix2vec(range(m.npix)))
        
        pixels_dist = np.arccos(np.matmul(center_vec, pixels_vec))

        radius_rad = radius.to_value(u.rad)

        m[:] = np.exp(-(pixels_dist - radius_rad)**2 / 2/sigma_rad/sigma_rad)
        
        # Normalize
        m.density(False)

        m /= np.sum(m)

        return m
    
        
    def plot(self, *args, **kwargs):

        return super().plot(*args, **kwargs, coord = self.frame)
        
    def plot_zoom(self,
                  center,
                  radius,
                  rot = 0*u.deg,
                  frame = 'icrs',
                  *args, **kwargs):

        fig = plt.figure(figsize = [4,4], dpi = 150)        

        ax = fig.add_axes([0,0,1,1],
                          projection = 'zoom_sin',
                          center = center,
                          radius = radius,
                          rot = rot,
                          frame = frame)
        
        return self.plot(ax = ax, *args, **kwargs)

    def plot_globe(self,
                   center,
                   rot = 0*u.deg,
                   frame = 'icrs',
                   *args, **kwargs):

        fig = plt.figure(figsize = [4,4], dpi = 150)        

        if frame == 'galactic':
            center = center.transform_to('galactic')
            frame = 'G'
        elif frame == 'icrs':
            center = center.transform_to('icrs')
            frame = 'C'
        else:
            raise ValueError("Only 'icrs' and 'galactic are currently supported'")

        center = center.represent_as(UnitSphericalRepresentation)
        
        ax = fig.add_axes([0,0,1,1],
                          projection = 'orthview',
                          rot = [center.lon.deg, center.lat.deg, rot.to_value(u.deg)],
                          coord = frame)
        
        ax.coords[0].set_ticks_visible(True)
        ax.coords[1].set_ticks_visible(True)
        ax.coords[0].set_ticklabel_visible(True)
        ax.coords[1].set_ticklabel_visible(True)

        return self.plot(ax = ax, *args, **kwargs)

    
