#!/usr/bin/env python
"""Construct an RGB raster image from three-band CCD images

RGBImage class is used to generate an RGB raster image from a set of
three images corresponding to red, green, and blue bands.  The custom
mapping function can be used to bring out details you need to see in
the final RGB image.

USAGE

Here is an example usage.  Say you want to create a JPEG image from a
set of three HST ACS images stored in FITS files.  The following set
of instructions will load 2D images into (NumPy) array objects,
corresponding to red, green, and blue bands:

  import pyfits as P
  hdus1 = P.open('f814.fits')
  hdus2 = P.open('f606.fits')
  hdus3 = P.open('f450.fits')
  img1,img2,img3 = hdus1[0].data,hdus2[0].data,hdus3[0].data

Now feed these images into RGBImage object:

  from RGBImage import *
  im = RGBImage(img1,img2,img3,
                scales=[7000,4000,15000],
                mapping=map_Lupton04,
                beta=2.)

In the above, the counts in each band image are scaled by given
factors.  The mapping is given by map_Lupton04 function (which
implements Lupton et al. (2004) algorithm), with a customized
nonlinearity factor (beta=2.).

To previw the resulting RGB image, do:

  im.show()

To save the resulting image as a JPEG file with the highest quality,
do:

  im.save_as('test.jpg',quality=100)

Note that a custom mapping function can be supplied by users, as long
as the following prototype is follows:

  def map_NAME(r,g,b,args={}):
    (... do mapping ...)
    return r,g,b

The argument args is a dictionary which should contain the variable
parameters used in the mapping, if necessary.  (Detail: RGBImage class
sends all the optinal arguments supplied at the construction to the
mapping function as a dictionary.)

REQUIREMENTS

  The following Python modules:
    -- Image (Python Imaging Library; tested with v1.1.4)
    -- pyfits (PyFITS; tested with v1.1b2)
    -- numpy (NumPy; tested with v0.9.9.2614)

REFERENCE

Lupton et al. (2004), PASP, 116, 133

The implementation closely follows the IDL procedure DJS_RGB_MAKE by
David Schlegel, which is build upon NW_RGB_MAKE by Nick Wherry.

TODO

More mapping functions need to be implemented (currently only Lupton
et al. (2004) and square root mappings have been coded up).

Handling of saturated pixels has not been implemented for Lupton04
mapping.

Work on better documentation.

HISTORY

  June 22, 2006 (ver 0.1) -- Implements the most essential part of the
  class.
"""
__version__ = '0.1 (June 22, 2006)'
__credits__ = '''The code is written by Taro Sato (nomo17k@gmail.com)'''

from PIL import Image as I
import astropy.io.fits as P
import numpy as N


def map_sqrt(r,g,b,lo=0.,hi=1.,args={}):
    """Square root mapping

DESCRIPTION

  The mapping is given by:

    R = f(r), G = f(g), B = f(b) ,

  where

           / 0,                     x < lo ,
    f(x) = | F(x - lo)/F(hi - lo),  lo <= x <= hi ,
           \ 1,                     hi < x ,

  where

    F(x) = sqrt(x) .

  Hence the mapped values are normalized to [0,1].
    """
    # prepare various parameters
    if args.has_key('lo'): lo = args['lo']
    if args.has_key('hi'): hi = args['hi']
    r,g,b = N.asarray(r).copy(),N.asarray(g).copy(),N.asarray(b).copy()
    # make sure all the entries are >= 0
    r = N.where(r<lo, 0., r)
    g = N.where(g<lo, 0., g)
    b = N.where(b<lo, 0., b)
    mask = (lo<=r) * (r<=hi)
    r = N.where(mask, N.sqrt(r-lo)/N.sqrt(hi-lo), r)
    mask = (lo<=g) * (g<=hi)
    g = N.where(mask, N.sqrt(g-lo)/N.sqrt(hi-lo), g)
    mask = (lo<=b) * (b<=hi)
    b = N.where(mask, N.sqrt(b-lo)/N.sqrt(hi-lo), b)
    r = N.where(r>hi, 1., r)
    g = N.where(g>hi, 1., g)
    b = N.where(b>hi, 1., b)
    return r,g,b


def map_Lupton04(r,g,b,beta=3.,args={}):
    """Lupton et al. (2004) mapping

DESCRIPTION

  First define radius: rad = beta * (r + g + b).  Then the mapping is
  given by

    R = r*f(rad), G = g*f(rad), B = b*f(rad) ,

  where

           / 0,                     x <= 0 ,
    f(x) = |
           \ arcsinh(x) / x,        x > 0 .

  Hence the mapped values are NOT normalized to [0,1].  Saturation and
  max(R,G,B) > 1 cases have not been taken care of.
    """
    # prepare various parameters
    if args.has_key('beta'): beta = args['beta']
    r,g,b = N.asarray(r).copy(),N.asarray(g).copy(),N.asarray(b).copy()
    # make sure all the entries are >= 0
    r = N.where(r>0., r, 0.)
    g = N.where(g>0., g, 0.)
    b = N.where(b>0., b, 0.)
    # compute nonlinear mapping
    radius = beta * (r + g + b)
    radius_ok = radius > 0.0
    nlfac = radius * 0.0
    nlfac[radius_ok] = N.arcsinh(radius[radius_ok])/radius[radius_ok]
    r = r*nlfac
    g = g*nlfac
    b = b*nlfac
    if args.get('desaturate'):
        # optionally desaturate pixels that are dominated by a single
        # colour to avoid colourful speckled sky
        a = (r+g+b)/3.0
        N.putmask(a, a == 0.0, 1.0)
        rab = r / a / beta
        gab = g / a / beta
        bab = b / a / beta
        mask = N.array((rab, gab, bab))
        w = N.max(mask, 0)
        N.putmask(w, w > 1.0, 1.0)
        w = 1 - w
        w = N.sin(w*N.pi/2.0)
        r = r*w + a*(1-w)
        g = g*w + a*(1-w)
        b = b*w + a*(1-w)
    # optionally add a grey pedestal
    if args.has_key('pedestal'):
        pedestal = args['pedestal']
        r += pedestal
        g += pedestal
        b += pedestal
    return r, g, b


class RGBImage(object):

    def __init__(self,r,g,b,scales=[1.,1.,1.],
                 mapping=map_Lupton04,**args):
        """Container for RGB image

  r,g,b -- 2D array of images corresponding to red, green, and blue
  scales -- Scale factors for each band; defaults to [1,1,1]
  mapping -- Mapping function to use; defaults to map_Lupton04

  In addition, optional keyword arguments can be passed to change the
  default mapping parameters (if any).
        """
        self.__dict__.update(args)
        self.scales = scales
        self.mapping = mapping
        # input data sanity check
        self.r = N.asarray(r).copy()*self.scales[0]
        self.g = N.asarray(g).copy()*self.scales[1]
        self.b = N.asarray(b).copy()*self.scales[2]
        if not (self.r.shape==self.g.shape==self.b.shape):
            raise ValueError('input data shape not consistent')
        # self.shape needs to be (horizontal,vertical) in size
        self.shape = (self.r.shape[1],self.r.shape[0])
        # set normalized RGB pixel values
        self.make_image()

    def make_image(self):
        """Generate RGB image with the current mapping function"""
        # call the specified mapping function
        try: R,G,B = self.mapping(self.r,self.g,self.b,args=self.__dict__)
        except NameError:
            raise ValueError('mapping method "%s" is not defined' %
                              self.mapping.__name__)
        # reorganize for putdata(data)
        data = N.array([R.ravel(),G.ravel(),B.ravel()])
        data = N.transpose(data)
        data = (255.*data).astype(int)
        data = N.where(data>255,255,data)
        data = N.where(data<0,0,data)
        pdata = []
        # putdata(x) does not work unless the (R,G,B) is given as tuple!!
        for each in data: pdata.append(tuple(each))
        self.pdata = pdata
        # make the image
        self.img = I.new('RGB',size=self.shape)
        self.img.putdata(self.pdata)

    def show(self):
        """Show the RGB image"""
        self.img.show()

    def save_as(self,fname,**args):
        """Save the RGB image to a file

  DESCRIPTION

    The file format will be deduced from the name of the extension.
    Keyword arguments can be used to pass options specific to the file
    format; see the documentation of Python Imaging Library for more
    info.
        """
        s = 'self.img.save(fname'
        # this way the given options are sent to the save function
        for key,val in args.iteritems(): s += (',%s=%s' % (key,val))
        s += ')'
        exec(s)


if __name__ == '__main__':
    """test code"""
    hdus1 = P.open('f814_mosaic_wherry.fits')
    hdus2 = P.open('f606_mosaic_wherry.fits')
    hdus3 = P.open('f450_mosaic_wherry.fits')

    img1,img2,img3 = hdus1[0].data,hdus2[0].data,hdus3[0].data

    lup = RGBImage(img1,img2,img3,
                   scales=[7000,4000,15000],beta=3)
    lup.show()
    lup.save_as('test.jpg',quality=100)
