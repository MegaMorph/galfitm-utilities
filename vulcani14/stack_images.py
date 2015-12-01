import pyfits
import numpy
from numpy import cos, sin
import math
from scipy.ndimage.interpolation import *
from meanclipstack import *

count = 1

def mapping(p, xcentre, ycentre, xscale, yscale, angle, win, hin, wout, hout):
    # x and y are coords in the output image
    x, y = p
    # offset to centre of output image
    dx = x - wout/2.0 - 0.5
    dy = y - hout/2.0 - 0.5
    # scale
    dx /= xscale
    dy /= yscale
    # rotate
    theta = (angle+90) * math.pi / 180.0
    x = dx * cos(theta) + dy * sin(theta)
    y = -dx * sin(theta) + dy * cos(theta)
    # offset to centre of input image
    x += xcentre - 1
    y += ycentre - 1
    #if p[0]%10 == 0 and p[1]%10 == 0:
    #    #print p, (x, y)
    return (x, y)

def stack_images(filenames, outsize, extension=0, valueoffsets=0.0, valuescales=1.0,
                 xcentres=0.0, ycentres=0.0, scales=1.0, angles=0.0, axisratios=1.0, clip=3.0):
    global count
    w = tryget(outsize, 0)
    h = tryget(outsize, 1)
    stack = []
    for j, fn in enumerate(filenames):
        #print fn
        p = pyfits.open(fn)
        im = p[extension].data
        wim, him = im.shape
        # rotate the object
        angle = tryget(angles, j)
        #print 'angle', angle
        # scale and circularise the object
        scale = tryget(scales, j)
        axisratio = tryget(axisratios, j)
        xscale = scale / axisratio
        yscale = scale
        #print 'axisratio', axisratio
        #print 'xscale', xscale
        #print 'yscale', yscale
        # centre the object
        xcentre = tryget(xcentres, j)
        ycentre = tryget(ycentres, j)
        #print 'xcentre', xcentre, wim/2.0
        #print 'ycentre', ycentre, him/2.0
        im = im + tryget(valueoffsets, j)
        im = im * tryget(valuescales, j) / (xscale*yscale)
        # apply transform
        out = geometric_transform(im, mapping, output_shape=(w, h),
              extra_arguments=(xcentre, ycentre, xscale, yscale, angle, wim, him, w, h))
        stack.append(out)
        #pyfits.writeto('stack_%i.fits'%count, out, clobber=True)
        count += 1
        p.close()
    mean = meanclipstack(stack, 0, clip, verbose=False)[0]
    return mean, stack


def tryget(x, i):
    try:
        return x[i]
    except TypeError:
        return x
    except IndexError:
        return x
