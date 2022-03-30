from mhealpy import HealpixMap, HealpixBase
import mhealpy as hp

import numpy as np

from astropy.coordinates import UnitSphericalRepresentation, SkyCoord
import astropy.units as u
from astropy.units import cds

class SkyLocBase(HealpixBase):
    """
    Initialize HealpixMap with an attached coordinate frame.

    Args:
        frame (BaseCoordinateFrame): Astropy's coordinate frame.
    """

    def __init__(self, frame = 'ICRS', *args, **kwargs):
        
        self._frame = kwargs.pop('frame', 'icrs')

        super().__init__(*args, **kwargs)

    @property
    def frame(self):
        return self._frame
        
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
    def circular_roi(cls,
                     center,
                     radius,
                     nside):
        """
        Prepare a mesh with high resolution inside a circular disc.

        Args:
            center (SkyCoord): Disc center
            radius (Angle or Quantity): Disc radius.
            nside (int): NSIDE inside the disc.
        """

        source_ang = center.represent_as(UnitSphericalRepresentation)
        
        # Empty map
        mEq = HealpixBase(nside = nside, scheme = 'ring')
        
        source_vec = hp.ang2vec(source_ang.lon.deg,
                                source_ang.lat.deg,
                                lonlat = True)
        
        disc_pix = mEq.query_disc(source_vec, radius.to_value(u.rad))

        # MOC map
        m = HealpixBase.moc_from_pixels(mEq.nside, disc_pix)

        return cls(base = m,
                   frame = center.frame)

        
    @classmethod
    def annular_roi(cls,
                    center,
                    radius,
                    width,
                    nside):
        
        """
        Create a probability for an annulus error.

        The triangulation of a signal by two detector typically results 
        in this pattern.
        
        The followin distribution is distribution is assumed.

        Args:
            skycoord (SkyCoord): Best estimate for the source location
            radius (Angle or Quantity): Error radius 
            cont (float): Fraction containment that the error radius 
                corresponds to. Default is equivalent to radius = sigma
            nside (int): Override the nside estimated based on the error radius.
        """

