from mhealpy import HealpixMap, HealpixBase
import mhealpy as hp

import numpy as np

from astropy.coordinates import UnitSphericalRepresentation, SkyCoord
import astropy.units as u
from astropy.units import cds

class SkyLocMap(HealpixMap):
    """
    Initialize HealpixMap with an attached coordinate frame.

    Args:
        frame (BaseCoordinateFrame): Astropy's coordinate frame. Default: 'ICRS'
        args, kwargs: Passed to HealpixMap
    """

    def __init__(self, *args, **kwargs):
        
        self._frame = kwargs.pop('frame', 'icrs')

        super().__init__(*args, **kwargs)

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
    def from_coord_and_error(cls, skycoord, radius,
                             cont = 0.632121,
                             nside = None,
    ):
        """
        Create a probability map given a source location and  error radius. 
        A 2D Gaussian probability distribution is assumed.

        Args:
            skycoord (SkyCoord): Best estimate for the source location
            radius (Angle or Quantity): Error radius 
            cont (float): Fraction containment that the error radius 
                corresponds to. Default is equivalent to radius = sigma
            nside (int): Override the nside estimated based on the error radius.
        """

        if not (cont > 0 and cont < 1):
            raise ValueError("Containment must be in the range (0,1)")
        
        sigma = radius.to_value(u.rad) / np.sqrt(np.log(1/(1-cont)))

        if nside is None:
            # Pixel size is 5 times smaller than sigma
            approx_nside = 10*np.sqrt(4*np.pi/12)/sigma
            order = int(np.ceil(np.log2(approx_nside)))
        else:
            order = hp.nside2order(nside)

        source_ang = skycoord.represent_as(UnitSphericalRepresentation)
        
        # Empty map
        mEq = HealpixBase(order = order, scheme = 'ring')
        
        source_vec = hp.ang2vec(source_ang.lon.deg,
                                source_ang.lat.deg,
                                lonlat = True)
        
        disc_pix = mEq.query_disc(source_vec, 5*sigma)

        # MOC map
        m = HealpixMap.moc_from_pixels(mEq.nside, disc_pix, density=True)

        # Fill
        pixels_vec = np.array(m.pix2vec(range(m.npix)))
        
        pixels_dist = np.arccos(np.matmul(source_vec, pixels_vec))

        m[:] = np.exp(-pixels_dist*pixels_dist / sigma/sigma)

        # Normalize
        m.density(False)

        m /= np.sum(m)

        return cls(base = m,
                   data = m.data,
                   density = m.density,
                   frame = skycoord.frame)

        
