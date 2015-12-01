import os.path
import random
import numpy
import pyfits
from match import matchids
import cosmology
import matplotlib
from matplotlib import pyplot
from RGBImage import *
from scipy.interpolate import RectBivariateSpline
from stack_images import stack_images
import Image
from radialprofile import azimuthalAverage

matched_cats = None

bands = ('H', 'z', 'g')

n_cut1 = 1.4
n_cut2 = 1.9
re_cut1 = 0.7
re_cut2 = 0.85

sel_grad_tex = {'low_re_grad': r'$\mathcal{R}^H_g\! <\ %.2f$'%re_cut1,
                'mid_re_grad': r'$%.2f <\ \mathcal{R}^H_g\! <\ %.2f$'%(re_cut1, re_cut2),
                'high_re_grad': r'$\mathcal{R}^H_g\! >\ %.2f$'%re_cut2,
                'low_n_grad': r'$\mathcal{N}^H_g\! <\ %.2f$'%n_cut1,
                'mid_n_grad': r'$%.2f <\ \mathcal{N}^H_g\! <\ %.2f$'%(n_cut1, n_cut2),
                'high_n_grad': r'$\mathcal{N}^H_g\! >\ %.2f$'%n_cut2}

size = 75
scales = 2.5
maxstack = 1000

def stack(par='n'):
    global matched_cats
    if matched_cats is None:
        d = get_galfit_cat()
        #m = get_masses_cat()
        b = get_benedettas_cat()
        print len(b)
        d, b = match_cats(d, b)
        print len(b)
        matched_cats = select(d, b)
    d, b = matched_cats
    print len(d), len(b)
    if par == 'n':
        grad = b.field('N_GALFIT_RFBAND_8') / b.field('N_GALFIT_RFBAND_3')
        low_grad = grad < n_cut1
        high_grad = grad > n_cut2
    elif par == 're':
        grad = b.field('RE_GALFIT_RFBAND_8_corr') / b.field('RE_GALFIT_RFBAND_3_corr')
        low_grad = grad < re_cut1
        high_grad = grad > re_cut2
    mid_grad = numpy.logical_not(low_grad | high_grad)
    n_r, lown, highn = get_n(b)
    MH, ur, red, green, blue = get_phot(b)
    plot_cmr(MH, ur, red, green, blue, 'cmr_H')
    plot_cmr(b.r_abs, ur, red, green, blue, 'cmr_r')
    plot_cmr(grad, ur, red, green, blue, 'c_%s_grad'%par)
    #plot_cmr(b.RE_GALFIT_RFBAND_1_corr, ur, red, green, blue, 're_ur')
    plot_cmr(d.RE_GALFIT_BAND[:,0], ur, red, green, blue, 're_ur')
    print 'N(red, green, blue)', len(red.nonzero()[0]), len(green.nonzero()[0]), len(blue.nonzero()[0])
    print 'N(lown, highn)', len(lown.nonzero()[0]), len(highn.nonzero()[0])
    print 'N(low_grad, mid_grad, high_grad)', len(low_grad.nonzero()[0]), len(mid_grad.nonzero()[0]), len(high_grad.nonzero()[0])
    for sel_colour in ('red', 'green', 'blue'):
        for sel_n in ('highn', 'lown'):
            sel = eval('%s & %s'%(sel_colour, sel_n)).nonzero()[0]
            label_col_n = '%s_%s'%(sel_colour, sel_n)
            coadd_col_n = stack_galfit_images(d.FILE_GALFIT[sel], label_col_n)
            for sel_grad in ('low_grad', 'mid_grad', 'high_grad'):
                sel = eval('%s & %s & %s'%(sel_colour, sel_n, sel_grad)).nonzero()[0]
                label = '%s_%s_%s'%(sel_colour, sel_n, sel_grad.replace('_grad', '_%s_grad'%par))
                coadd = stack_galfit_images(d.FILE_GALFIT[sel], label)
                if coadd is not None:
                    residual_image(coadd, coadd_col_n, label)
    make_figures(par)

def ticksoff(ax):
    #pyplot.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def forceAspect(ax,aspect=1):
    x1, x2 = ax.get_xbound()
    dx = x2-x1
    y1, y2 = ax.get_ybound()
    dy = y2-y1
    aspect = dx/dy * aspect
    ax.set_aspect(aspect)

def make_figures(par='n'):
    matplotlib.rcParams.update({'font.size': 12})
    if par == 'n':
        vmin, vmax = -0.2, 0.2
        yticks = (-0.1, 0.0, 0.1)
    elif par == 're':
        vmin, vmax = -0.3, 0.3
        yticks = (-0.2, 0.0, 0.2)
    # Currently getting the sizes (number of images in) each stack is a bit of a hack.
    # Need to manually save the output lines contining this info to a file 'stack_sizes',
    # which is read in here:
    stack_sizes = {}
    for l in file('stack_sizes'):
        ls = l.split()
        stack_sizes[ls[0]] = int(ls[1])
    for sel_colour in ('red', 'green', 'blue'):
        dd = 1 # int(size/8.0)
        dd1 = dd
        dd2 = size - dd1 + 1
        for sel_n in ('highn', 'lown'):
            i = 1
            pyplot.figure(figsize=(7, 7))
            label_col_n = '%s_%s'%(sel_colour, sel_n)
            f = 'stack_%s.png'%label_col_n
            if os.path.exists(f):
                ax = pyplot.subplot(4, 4, i, title=r'$\rm %s\ \ %s$'%(sel_colour, sel_n.replace('n', '$-$n')))
                pyplot.imshow(Image.open(f).crop((dd1, dd1, dd2, dd2)), interpolation='nearest')
                ticksoff(ax)
                count = stack_sizes.get('%s_%s'%(sel_colour, sel_n))
                pyplot.xlabel((r'${\rm all}\ \mathcal{%s}^H_g$'+'\n$(%i)$')%(par[0].upper(), count), linespacing=1.5)
            for col, sel_grad in enumerate(('low_%s_grad', 'mid_%s_grad', 'high_%s_grad')):
                i += 1
                count = stack_sizes.get('%s_%s_%s'%(sel_colour, sel_n, sel_grad%par))
                if count is None:
                    count = 0
                label = '%s_%s_%s'%(sel_colour, sel_n, sel_grad%par)
                f = 'stack_%s.png'%label
                if os.path.exists(f):
                    ax = pyplot.subplot(4, 4, i)
                    pyplot.imshow(Image.open(f).crop((dd1, dd1, dd2, dd2)), interpolation='nearest')
                    ticksoff(ax)
                    if col == 0:
                        pyplot.ylabel(r'$Hzg\ \ {\rm stack}$', size='large')
                    pyplot.xlabel('%s\n$(%i)$'%(sel_grad_tex[sel_grad%par], count), linespacing=1.5)
                    ax.xaxis.set_label_position('top')
                f = 'stackres_%s.png'%label
                if os.path.exists(f):
                    ax = pyplot.subplot(4, 4, i+4)
                    pyplot.imshow(Image.open(f).crop((dd1, dd1, dd2, dd2)), interpolation='nearest')
                    ticksoff(ax)
                    if col == 0:
                        pyplot.ylabel(r'$Hzg\ \ {\rm residuals}$', size='large')
                delta = residual_delta(label)
                if delta is not None:
                    ax = pyplot.subplot(4, 4, i+8)
                    pyplot.imshow(delta[-dd:dd:-1, dd:-dd], interpolation='nearest', vmin=vmin, vmax=vmax)
                    #pyplot.colorbar()
                    ticksoff(ax)
                    if col == 0:
                        pyplot.ylabel(r'$H/g\ \ {\rm residuals}$', size='large')
                ax = pyplot.subplot(4, 4, i+12)
                residual_prof(label)
                pyplot.axis((0, size*0.5/scales, 1.0*vmin, 1.0*vmax))
                ax.xaxis.set_ticks((0, 5, 10, 15))
                ax.yaxis.set_ticks(yticks)
                forceAspect(ax)
                if col == 0:
                    pyplot.ylabel(r'${\rm residual\ profile}$', size='large')
                else:
                    ax.set_yticklabels([])
                pyplot.xlabel(r'$\rm radius\ [pix]$', size='large')
            pyplot.subplots_adjust(wspace=0.0, hspace=0.15, bottom=0.075, top=0.92, left=0.02, right=0.98)
            pyplot.savefig('stackfig_%s_%s_%s_grad.ps'%(sel_colour, sel_n, par))
            pyplot.savefig('stackfig_%s_%s_%s_grad.pdf'%(sel_colour, sel_n, par))
            pyplot.close('all')
    os.system('psmerge stackfig_*_%s_grad.ps > stackfig_%s_grad.ps'%(par, par))


def residual_image(coadd, coadd_col_n, label='all'):
    res = (coadd - coadd_col_n)/(coadd_col_n + 1)  # the constant should fade noise
    for i, band in enumerate(bands):
        pyfits.writeto('stackres_%s_%s.fits'%(label, band), res[i], clobber=True)
    res *= 3.0
    res += 0.5
    colimg = RGBImage(res[0], res[1], res[2], beta=0.0001)
    colimg.save_as('stackres_%s.png'%label)
    colimg = numpy.asarray(colimg.img)
    for i, band in enumerate(bands):
        pyfits.writeto('stackrescol_%s_%s.fits'%(label, band), colimg[:,:,i], clobber=True)
    return res


def residual_prof(label):
    colours = ['r', 'g', 'b']
    for i, band in enumerate(bands):
        f = 'stackres_%s_%s.fits'%(label, band)
        if os.path.exists(f):
            res = pyfits.getdata(f)
            rad, prof = azimuthalAverage(res, returnradii=True, binsize=1.0)
            middle = rad < res.shape[0]/2.0
            rad, prof = [x[middle] for x in (rad, prof)]
            rad /= scales
            pyplot.plot(rad, prof, '-', color=colours[i])
            pyplot.axis('on')
        else:
            pyplot.axis('off')


def residual_delta(label):
    H = 'stackres_%s_H.fits'%label
    g = 'stackres_%s_g.fits'%label
    if not(os.path.exists(H) &  os.path.exists(g)):
        return None
    H = pyfits.getdata(H)
    g = pyfits.getdata(g)
    delta = H-g
    return delta


def centroid(im, box=(15,15)):
    dx, dy = [int((im.shape[i]-box[i])/2.0) for i in range(len(im.shape))]
    im = im[dx:-dx, dy:-dy]
    x, y = [numpy.arange(n) for n in im.shape]
    imx = im.sum(1)
    imy = im.sum(0)
    cx = (x*imx).sum()/float(imx.sum())
    cy = (y*imy).sum()/float(imy.sum())
    return cx+dx, cy+dy


def centroid_delta(im, box=(15,15)):
    cx, cy = centroid(im, box)
    sx, sy = [(n - 1)/2.0 for n in im.shape]
    return cx-sx, cy-sy
    

def stack_galfit_images(infns, label='all', par='n', debug=False):
    #fns = random.sample(infns, min(len(infns),maxstack))
    fns = infns[:min(len(infns),maxstack)]
    #for j in range(len(fns)):
    #    fns[j] = fns[j].replace('/home/boris/gama/galapagos/galapagos_2.0.3_galfit_0.1.2.1_GAMA_9', '/Users/spb/gala9')
    bandscales = numpy.array([0.06, 0.1, 0.4])/10.0
    beta=2.0
    outsize = size
    n = len(fns)
    if n == 0:
        return None
    if maxstack > 5 and n < 5:
        return None
    print label, len(infns), n
    if n < 5:
        print fns
    nbands = len(bands)
    coadd = []
    modelcoadd = []
    for i, band in enumerate(bands):
        valueoffsets = []
        valuescales = []
        xcentres = []
        ycentres = []
        radii = []
        angles = []
        axisratios = []
        for j, fn in enumerate(fns):
            p = pyfits.open(fn)
            results = p['final_band'].data
            # the following scales to the largest image
            # not suitable when building residual images (need same shape for all stacked images)
            #outsize = max(outsize, max(p[0].shape))
            # index to scale according to the parameters in this band
            bandidx = (results.field('BAND')==band).nonzero()[0][0]
            # index to scale according to r-band parameters
            rbandidx = (results.field('BAND')=='r').nonzero()[0][0]
            valueoffsets.append(-results.field('COMP1_SKY')[bandidx])
            valuescales.append(10**(0.4*(results.field('COMP2_MAG')[rbandidx]-18)))
            xcentres.append(results.field('COMP2_YC')[rbandidx])
            ycentres.append(results.field('COMP2_XC')[rbandidx])
            if par == 're':
                radii.append(results.field('COMP2_RE')[rbandidx])
            elif par == 'n':
                radii.append(results.field('COMP2_RE')[bandidx])
            angles.append(results.field('COMP2_PA')[rbandidx])
            axisratios.append(results.field('COMP2_AR')[rbandidx])
        axisratios = 1.0
        coaddi, stacki = stack_images(fns, outsize, 'input_%s'%band, valueoffsets, valuescales,
                                      xcentres, ycentres, scales, angles, axisratios, clip=3.0)
        coadd.append(coaddi)
        pyfits.writeto('stack_%s_%s.fits'%(label, band), coaddi, clobber=True)
        for si in stacki:
            pyfits.append('stackall_%s_%s.fits'%(label, band), si)
        # stack the models to verify
        coaddi, stacki = stack_images(fns, outsize, 'model_%s'%band, valueoffsets, valuescales,
                                      xcentres, ycentres, scales, angles, axisratios, clip=3.0)
        modelcoadd.append(coaddi)
        pyfits.writeto('stackmodel_%s_%s.fits'%(label, band), coaddi, clobber=True)
        for si in stacki:
            pyfits.append('stackmodelall_%s_%s.fits'%(label, band), si)
    coadd = numpy.array(coadd)
    for i, b in enumerate(bandscales):
        coadd[i] *= b
    colimg = RGBImage(coadd[0], coadd[1], coadd[2], beta=beta)
    colimg.save_as('stack_%s.png'%label)
    coadd = numpy.array(modelcoadd)
    for i, b in enumerate(bandscales):
        coadd[i] *= b
    colimg = RGBImage(coadd[0], coadd[1], coadd[2], beta=beta)
    colimg.save_as('stackmodel_%s.png'%label)
    return coadd
    

def select(d, b):
    MR = b.r_abs
    #ReR = b.RE_GALFIT_RFBAND_1_corr
    ur = b.u_abs - b.r_abs
    ReR = d.RE_GALFIT_BAND[:,0]
    #MR = mR - cosmology.dmod_flat(b.Z, 70.0, 0.3)
    mR = d.MAG_GALFIT_BAND[:,0]
    s = b.Z > 0.05
    s &= b.Z < 0.2
    s &= MR < -21.2
    s &= MR > -22.5
    s &= ReR > 6.0
    s &= ReR < 8.0
    s &= ur > 1.0
    s &= ur < 3.0
    s &= mR < 19.8
    # sort by r-band apparent magnitude, brightest first
    idx = mR[s].argsort()[::-1]
    return d[s][idx], b[s][idx]


def plot_cmr(x, ur, red, green, blue, fn):
    pyplot.plot(x[red], ur[red], 'r.')
    pyplot.plot(x[green], ur[green], 'g.')
    pyplot.plot(x[blue], ur[blue], 'b.')
    #pyplot.axis(ymin=0.5, ymax=3.5)
    pyplot.savefig('%s.pdf'%fn)
    pyplot.close('all')

def get_n(b):
    n_r = b.N_GALFIT_RFBAND_1
    lown = n_r <= 2.5
    highn = n_r > 2.5
    return n_r, lown, highn

def get_phot(b):
    MH = b.H_abs
    ur = b.u_abs - b.r_abs
    blue = ur < 1.6
    red = ur > 2.1
    green = numpy.logical_not(red | blue)
    return MH, ur, red, green, blue

def match_cats(d, m):
    i = matchids(d.CATAID,m.CATAID)
    matched = i >= 0
    mm = m[matched]
    dm = d[i[matched]]
    return dm, mm

def get_galfit_cat():
    path = '/home/boris/gama/galapagos/galapagos_2.0.3_galfit_0.1.2.1_GAMA_9'
    #path = './galapagos_2.0.3_galfit_0.1.2.1_GAMA_9'
    cat = os.path.join(path, 'GAMA_9_ffvqqff_gama_only.fits')
    d = pyfits.getdata(cat)
    return d

def get_masses_cat():
    path = '/home/ppzsb1/gama_cats'
    cat = os.path.join(path, 'StellarMassesv06.fits')
    d = pyfits.getdata(cat, 1)
    return d

def get_benedettas_cat():
    cat = '/home/ppzsb1/MegaMorph/benedetta/gama_verylast_cataid.fits'
    d = pyfits.getdata(cat, 1)
    return d

if __name__ == "__main__":
    stack('n')
    stack('re')
