from matplotlib.projections import register_projection

from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation, SkyCoord

from matplotlib.transforms import Bbox

import numpy as np

class ZoomSinAxes(WCSAxes):

    name = "zoom_sin"

    def __init__(self,
                 fig,
                 rect,
                 center,
                 radius,
                 rot = 0*u.deg,
                 frame = 'icrs',
                 **kwargs):

        # Get equivalent WCS FITS header
        if not isinstance(rect, Bbox):
            rect = Bbox.from_bounds(*rect)

        naxis1 = int(fig.get_figwidth() * fig.dpi * rect.width)
        naxis2 = int(fig.get_figheight() * fig.dpi * rect.height)
        
        cdelt = 2*radius.to_value(u.deg)/naxis1

        if frame == 'galactic':
            center = center.transform_to('galactic')
            ctype1 = "GLON-SIN"
            ctype2 = "GLAT-SIN"

        elif frame == 'icrs':
            center = center.transform_to('icrs')
            ctype1 = "RA---SIN"
            ctype2 = "DEC--SIN"

        else:
            raise ValueError("Only 'icrs' and 'galactic are currently supported'")
            
        center = center.represent_as(UnitSphericalRepresentation)
   
        crpix1 = (naxis1+1)/2
        crpix2 = (naxis2+1)/2

        header = { "NAXIS": 2, 
                   "CTYPE1": ctype1,
                   "NAXIS1": naxis1,
                   "CRPIX1": crpix1,
                   "CRVAL1": center.lon.deg, 
                   "CDELT1": -cdelt,
                   "CUNIT1": 'deg     ',
                   "CTYPE2": ctype2,
                   "NAXIS2": naxis2,
                   "CRPIX2": crpix2,
                   "CRVAL2": center.lat.deg,
                   "CDELT2": cdelt,
                   "CUNIT2": 'deg     ',
                   "PV1_1": 0,
                   "PV1_2": 90,
                   "LONPOLE": -rot.to_value(u.deg)}
        
        super().__init__(fig, rect, WCS(header),
                         xlim = (0, naxis1),
                         ylim = (0, naxis2),
                         aspect = 1,
                         **kwargs)

register_projection(ZoomSinAxes)

