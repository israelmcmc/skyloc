from mhealpy import HealpixMap

from astropy.coordinates import UnitSphericalRepresentation, SkyCoord
import astropy.units as u
from astropy.units import cds

class SkyLocMap(HealpixMap):
    """
    Initialize HealpixMap with an attached coordinate frame.

    Args:
        frame (BaseCoordinateFrame): Astropy's coordinate frame
        args, kwargs: Passed to HealpixMap
    """

    def __init__(self, frame = 'icrs', *args, **kwargs):
        
        self._frame = frame

        super().__init__(*args, **kwargs)

    def pix2skycoord(self, pix):

        lon,lat = self.pix2ang(pix, lonlat = True)

        skycoord = SkyCoord(lon*u.deg, lat*u.deg, frame = self._frame)

        return skycoord

    def skycoord2pix(self, skycoord):

        coord = skycoord.represent_as(UnitSphericalRepresentation)

        pix = self.ang2pix(coord.lon.deg, coord.lat.deg, lonlat = True)

        return pix

    
