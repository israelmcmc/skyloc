from mhealpy import HealpixMap, HealpixBase
import mhealpy as hp

import numpy as np

from astropy.coordinates import UnitSphericalRepresentation, SkyCoord
import astropy.units as u
from astropy.units import cds

from mocpy import MOC

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
                     nside,
                     *arg, **kwargs):
        """
        Prepare a mesh with high resolution inside a circular disc.

        Args:
            center (SkyCoord): Disc center
            radius (Angle or Quantity): Disc radius.
            nside (int): NSIDE inside the disc.
        """

        center_ang = center.represent_as(UnitSphericalRepresentation)
        
        # Empty map
        mEq = HealpixBase(nside = nside, scheme = 'ring')
        
        center_vec = hp.ang2vec(center_ang.lon.deg,
                                center_ang.lat.deg,
                                lonlat = True)
        
        disc_pix = mEq.query_disc(center_vec, radius.to_value(u.rad))

        # MOC map
        m = HealpixBase.moc_from_pixels(mEq.nside, disc_pix)

        return cls(base = m,
                   frame = center.frame,
                   *args, **kwargs)

        
    @classmethod
    def annular_roi(cls,
                    center,
                    radius,
                    width,
                    nside,
                    *args,
                    **kwargs):        
        """
        Prepare a mesh with high resolution inside an annulus.

        Args:
            center (SkyCoord): Center of the annulus
            radius (Angle or Quantity): Angular distance from the center to the middle 
                of the ring.
            width (Angle or Quantity): Width of the ring --i.e. along the 
                radial direction.
            nside (int): NSIDE inside the annulus
        """

        # Using mocpy due to this issue with healpy
        # https://github.com/healpy/healpy/issues/641

        center = center.represent_as(UnitSphericalRepresentation)

        order = hp.nside2order(nside)
        
        # Outer disc
        moc_o = MOC.from_cone(lon = center.lon,
                              lat = center.lat,
                              radius = radius + width/2,
                              max_depth = order)

        # Inner disc
        moc_i = MOC.from_cone(lon = center.lon,
                              lat = center.lat,
                              radius = radius - width/2,
                              max_depth = order)

        # Annulus
        moc = moc_o.difference(moc_i)

        # Convert the pixel ranges into a pixel list
        # There doesn't seem to be an access method for "._interval_set". Weird.
        # Not ideal, I hope this doesn't break in future versions
        # See https://github.com/cds-astro/mocpy/issues/72
        
        hires_pix = []

        for i,f in np.transpose(hp.uniq2range(nside, moc._uniq_format())):
            hires_pix += list(np.arange(i,f, dtype = int))

        # Add the complement pixels in order to have a well defined mesh
        uniq = np.append(hp.nest2uniq(nside, hires_pix), 
                         moc.complement()._uniq_format().astype(int))

        # Create the empty map
        return cls(uniq = uniq, *args, **kwargs)


