#!/usr/bin/env python

from glob import glob
import os
import numpy as np
import astropy.io.fits as pyfits
import matplotlib
from matplotlib import pyplot
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from numpy import log, log10, exp, power, pi
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import scoreatpercentile
from scipy.special import gamma, gammaincinv
from scipy.optimize import fmin
from scipy.integrate import quad
from scipy.interpolate import interp1d
from RGBImage import *

matplotlib.rcParams.update({'font.size': 16,
                            'font.family': 'serif',
                            'font.serif': 'times',
                            'text.usetex': True})
labelsize = 12

rescolmap = LinearSegmentedColormap.from_list('rescol',
                ('blue', 'black', 'white', 'red'), N=256, gamma=1.0)
labels = {'MAG': '$m$', 'Re': '$R_e$', 'n': '$n$',
          'AR': '$b/a$', 'PA': '$\\theta$'}
marker = ['o', '^', 's', 'D', 'x', '+', '*']
linestyle = [':', '-', '.-', '.-.']

bands = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K']
w = np.array([3543, 4770, 6231, 7625, 9134, 10305, 12483, 16313, 22010],
             np.float)
xlim = (2000, 23000)
# varlist_std = ('MAG', 'Re', 'n', 'AR', 'PA')
varlist_std = ('MAG', 'Re', 'n')


def n_or_none(x, n):
    if x is not None:
        x = np.atleast_1d(x)
        if len(x) == 1:
            x = np.ones(n) * x
    return x


def plotimg(fns, bands=None, ids=None, name=None, label=False,
            lupton=True, uservmin=None, uservmax=None, usernorm=None,
            uservrange=None, zoom=1, nonparam=0, cmap_img=pyplot.cm.gray,
            cmap_res=pyplot.cm.gray, norm_res=None):
    """ Create a plot of data, model, residual from supplied filename

    If fns is a sequence of filenames, plot model and residual from each,
    for comparison.
    """

    if bands is None:
        nbands = 1
    else:
        nbands = len(bands)
    uservmin, uservmax, usernorm, uservrange = [n_or_none(x, nbands) for x in (
        uservmin, uservmax, usernorm, uservrange)]
    fns = np.atleast_1d(fns)
    nid = len(fns)
    rows = 1 + 2 * nid + 3 * nonparam
    fig = pyplot.figure(figsize=(2.5 * rows, 2.5 * nbands))
    for i, fn in enumerate(fns):
        extensions = ['input', 'model', 'residual']
        if nonparam > 0:
            extensions.append('nonparam')
        img = fit_images(fn, bands, extensions=extensions, zoom=zoom)
        if bands is None:
            bandlist = [' ']
        else:
            bandlist = bands
        if ids is not None:
            iid = ids[i]
        else:
            iid = os.path.splitext(os.path.basename(fn))[0]
        iid = iid.replace('_', '\_')
        if i == 0:
            vmin = []
            vmax = []
            norm = []
            vrange = []
            for ib, b in enumerate(bandlist):
                ax = fig.add_subplot(nbands, rows,
                                     1 + ib * rows + i * 2)
                if label and ib == nbands - 1:
                    ax.set_xlabel('image', fontsize=labelsize)
                ticksoff(ax)
                if uservmin is None:
                    vmin.append(scoreatpercentile(img[0][ib].ravel(), 0.1))
                else:
                    vmin.append(uservmin[ib])
                if uservmax is None:
                    vmax.append(scoreatpercentile(img[0][ib].ravel(), 99.9))
                else:
                    vmax.append(uservmax[ib])
                if usernorm is None:
                    norm.append(vmax[ib] / 100.0)
                else:
                    norm.append(usernorm[ib])
                if uservrange is None:
                    vrange.append(scoreatpercentile(img[2][ib].ravel(), 99.9) -
                                  scoreatpercentile(img[2][ib].ravel(), 0.1))
                else:
                    vrange.append(uservrange[ib])
                print('{}: vmin={:.2f}, vmax={:.2f}, '
                      'norm={:.2f}, vrange={:.2f}'.format(
                          b, vmin[ib], vmax[ib], norm[ib], vrange[ib]))
                im = img[0][ib][::-1]
                if lupton:
                    im = np.arcsinh(im / norm[ib])
                    vmin[ib] = np.arcsinh(vmin[ib] / norm[ib])
                    vmax[ib] = np.arcsinh(10 * vmax[ib] / norm[ib])
                    vrange[ib] = np.arcsinh(vrange[ib] / norm[ib])
                ax.imshow(im, cmap=cmap_img, vmin=vmin[ib],
                          vmax=vmax[ib], interpolation='none')
                ax.set_ylabel('${}$'.format(b), fontsize=labelsize)
        for ib, b in enumerate(bandlist):
            ax = fig.add_subplot(nbands, rows,
                                 2 + ib * rows + i * 2)
            if label and ib == nbands - 1:
                ax.set_xlabel('model {}'.format(iid), fontsize=labelsize)
            ticksoff(ax)
            im = img[1][ib][::-1]
            if lupton:
                im = np.arcsinh(im / norm[ib])
            ax.imshow(im, cmap=cmap_img, vmin=vmin[ib],
                      vmax=vmax[ib], interpolation='none')
        for ib, b in enumerate(bandlist):
            ax = fig.add_subplot(nbands, rows,
                                 3 + ib * rows + i * 2)
            if label and ib == nbands - 1:
                ax.set_xlabel('residual {}'.format(iid), fontsize=labelsize)
            ticksoff(ax)
            im = img[2][ib][::-1]
            if lupton:
                im = np.arcsinh(im / norm[ib])
            ax.imshow(im, cmap=cmap_res, norm=norm_res,
                      vmin=-vrange[ib], vmax=vrange[ib],
                      interpolation='none')
        if nonparam > 0 and img[3] is not None:
            for ib, b in enumerate(bandlist):
                ax = fig.add_subplot(nbands, rows,
                                     4 + ib * rows + i * 2)
                if label and ib == nbands - 1:
                    ax.set_xlabel('nonparim {}'.format(iid), fontsize=labelsize)
                ticksoff(ax)
                im = img[3][ib][::-1]
                if lupton:
                    im = np.arcsinh(im / norm[ib])
                ax.imshow(im, cmap=cmap_img, vmin=vmin[ib],
                          vmax=vmax[ib], interpolation='none')
            for ib, b in enumerate(bandlist):
                ax = fig.add_subplot(nbands, rows,
                                     5 + ib * rows + i * 2)
                if label and ib == nbands - 1:
                    ax.set_xlabel('datasub {}'.format(iid), fontsize=labelsize)
                ticksoff(ax)
                im = img[0][ib][::-1] - img[3][ib][::-1]
                if lupton:
                    im = np.arcsinh(im / norm[ib])
                ax.imshow(im, cmap=cmap_img, vmin=vmin[ib],
                          vmax=vmax[ib], interpolation='none')
            for ib, b in enumerate(bandlist):
                ax = fig.add_subplot(nbands, rows,
                                     6 + ib * rows + i * 2)
                if label and ib == nbands - 1:
                    ax.set_xlabel('nonparres {}'.format(iid), fontsize=labelsize)
                ticksoff(ax)
                im = img[2][ib][::-1] - img[3][ib][::-1]
                if lupton:
                    im = np.arcsinh(im / norm[ib])
                ax.imshow(im, cmap=cmap_res, norm=norm_res,
                          vmin=-vrange[ib], vmax=vrange[ib],
                          interpolation='none')
    if name is None:
        name = os.path.splitext(os.path.basename(fns[0]))[0]
    pyplot.tight_layout(h_pad=0.02, w_pad=0.02)
    fig.savefig('plots/images_{}.pdf'.format(name))
    pyplot.close('all')


def plotcolimg(fns, rgb='Hzg', ids=None, name=None, label=False,
               desaturate=True, pedestal=0):
    nbands = len(bands)
    fns = np.atleast_1d(fns)
    nid = len(fns)
    beta = 2.5
    scales = np.array((0.04, 0.055, 0.2))
    # offsets not so necessary now have nice desaturation feature working
    offsets = np.array([75.0, 40.0, 8.0]) * 0.5
    rows = 1 + 2 * nid
    fig = pyplot.figure(figsize=(15.0 / nbands * rows, 15))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0.0, wspace=0.0)
    for i, fn in enumerate(fns):
        if ids is not None:
            iid = ids[i]
        else:
            iid = os.path.splitext(os.path.basename(fn))[0]
        img = fit_images(fn, rgb)
        img[0] = [img[0][j] - offsets[j] for j in range(3)]
        img[1] = [img[1][j] - offsets[j] for j in range(3)]
        img[2] = [img[2][j] + scales[j] * 2 * offsets.mean() for j in range(3)]
        if i == 0:
            ax = fig.add_subplot(nbands, rows, 1 + i * 2)
            ticksoff(ax)
            ax.set_xlabel('image', fontsize=labelsize)
            colimg = RGBImage(*img[0], scales=scales, beta=beta,
                              desaturate=desaturate, pedestal=pedestal).img
            pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, rows, 2 + i * 2)
        ticksoff(ax)
        if label:
            ax.set_xlabel('model {}'.format(iid), fontsize=labelsize)
        colimg = RGBImage(*img[1], scales=scales, beta=beta,
                          desaturate=False, pedestal=pedestal).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, rows, 3 + i * 2)
        ticksoff(ax)
        if label:
            ax.set_xlabel('residual {}'.format(iid), fontsize=labelsize)
        colimg = RGBImage(*img[2], scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
    if name is None:
        name = os.path.splitext(os.path.basename(fns[0]))[0]
    fig.savefig('plots/colimages_{}.pdf'.format(name))
    pyplot.close('all')


def plotnonparamcolimg(fns, rgb='Hzg', ids=None, name=None, label=False,
                       desaturate=True, pedestal=0, beta=2.5,
                       scales=np.array([0.04, 0.055, 0.2]),
                       offsets=np.array([75.0, 40.0, 8.0]) * 0.5):
    nbands = len(bands)
    fns = np.atleast_1d(fns)
    nid = len(fns)
    fig = pyplot.figure(figsize=(15.0/nbands * (1+nid*2), 15))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.0, wspace=0.0)
    original_iid = None
    for i, fn in enumerate(fns):
        if ids is not None:
            iid = ids[i]
        else:
            iid = os.path.splitext(os.path.basename(fn))[0]
        img = fit_images(fn, rgb)
        # Second row, results with nonparam
        img = fit_images(iid, rgb)
        img[0] = [img[0][j] - offsets[j] for j in range(3)]
        img[1] = [img[1][j] - offsets[j] for j in range(3)]
        img[2] = [img[2][j] + scales[j]*2*offsets.mean() for j in range(3)]
        ax = fig.add_subplot(nbands, 1+2*nid, 1+2*nid+2+i*2)
        ticksoff(ax)
        colimg = RGBImage(*img[1], scales=scales, beta=beta,
                          desaturate=False, pedestal=pedestal).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, 1+2*nid, 1+2*nid+3+i*2)
        ticksoff(ax)
        colimg = RGBImage(*img[2], scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        # Third row, nonparam diagnostics
        nonparam = nonparam_images(iid, rgb)
        datasub = [img[0][j] - nonparam[j] for j in range(3)]
        nonparam = [nonparam - offsets[j] for j in range(3)]
        datasub = [datasub[j] - offsets[j] for j in range(3)]
        ax = fig.add_subplot(nbands, 1+2*nid, 2+4*nid+2+i*2)
        ticksoff(ax)
        ax.set_xlabel('nonparam %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*nonparam, scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, 1+2*nid, 2+4*nid+3+i*2)
        ticksoff(ax)
        ax.set_xlabel('datasub %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*datasub, scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
    fig.savefig('plots/nonparamcolimages_%s.pdf'%name)
    pyplot.close('all')


def ticksoff(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def plotres(fns, field, func=None, norm=None, ids=None, wlfunc=None):
    fns = np.atleast_1d(fns)
    res = [fit_results(f) for f in fns]
    nid = len(fns)
    # mec = ['black', None] * (1+nid//2)
    # mfc = ['white', 'black'] * (1+nid//2)
    # color = ['grey', 'grey'] * (1+nid//2)
    mec_func = ['DeepSkyBlue', 'DarkGreen', 'Orange']
    mfc_func = ['DeepSkyBlue', 'white', 'white']
    color = ['DeepSkyBlue', 'DarkGreen', 'Orange']
    mec_nofunc = ['MediumPurple', 'MediumSeaGreen', 'Pink']
    mfc_nofunc = ['MediumPurple', 'MediumSeaGreen', 'Pink']
    wlfunc = np.resize(wlfunc, nid)
    ymin, ymax = (1e99, -1e99)
    for i, fn in enumerate(fns):
        x = w + 100 * (1 + i // 2) * (-1)**i
        if func is not None and func[i] is not None:
            mec = mec_func
            mfc = mfc_func
            f = func[i][field]
            if norm is not None:
                f /= norm[i][field]
            plotfunc(f, wlfunc=wlfunc[i], color=color[i])
        else:
            mec = mec_nofunc
            mfc = mfc_nofunc
        r = res[i][field]
        if norm is not None:
            r /= norm[i][field](x)
        pyplot.errorbar(x, r, res[i][field + '_ERR'], color=mec[i],
                        marker=marker[i // 2], mec=mec[i],
                        markerfacecolor=mfc[i], linestyle='',
                        label=iid)
        ymin = min(ymin, (res[i][field] - res[i][field + '_ERR']).min())
        ymax = max(ymax, (res[i][field] + res[i][field + '_ERR']).max())
    yrange = ymax - ymin
    ymin -= 0.05 * yrange
    ymax += 0.05 * yrange
    pyplot.ylim(ymin, ymax)


def plotfunc(func, wlfunc=None, color='red', label=''):
    dx = (xlim[1] - xlim[0]) / 1000.0
    x = np.arange(xlim[0], xlim[1] + dx / 2.0, dx)
    if wlfunc is None:
        xfunc = x
    else:
        xfunc = wlfunc(x)
    y = func(xfunc)
    return pyplot.plot(x, y, '-', color=color, label=label, alpha=0.5)


def fit_results(fn, bands=None):
    """Get the galfitm results table from the specified filename(s)

    If `bands` is supplied, it should be a list of band id strings and fn
    should include a format placeholder `{}`, which will be replaced by each
    band id in turn. In this case a list of results tables is returned.
    """
    if bands is None:
        r = pyfits.getdata(fn, 'final_band')
    else:
        r = np.concatenate([pyfits.getdata(fn.format(b), 'final_band')
                              for b in bands])
    return r


def fit_images(fn, bands=None, zoom=None,
               extensions=['input', 'model', 'residual']):
    """Get the images from the specified galfit(m) filename(s)

    `bands` should be a list of band id strings.
    If fn includes a format placeholder `{}`, this will be replaced by each
    band id in turn. Otherwise all bands are assumed to be in a single file.
    """
    # auto-discovery of bands for galfitm files
    multiband = True
    if '{}' not in fn:
        with pyfits.open(fn) as p:
            names = [x.name for x in p]
        ext = extensions[0].upper()
        if ext in names:
            multiband = False
        else:
            ext = ext + '_'
            found_bands = [x.name.replace(ext, '') for x in p
                           if x.name.startswith(ext)]
            if bands is None:
                bands = found_bands
    out = []
    if not multiband:
        for ext in extensions:
            try:
                hdu = [pyfits.getdata(fn, ext)]
            except KeyError:
                hdu = None
            out.append(hdu)
    else:
        for ext in extensions:
            try:
                if '{}' in fn:
                    hdu = [pyfits.getdata(fn.format(b), ext)
                           for b in bands]
                else:
                    hdu = [pyfits.getdata(fn, '{}_{}'.format(ext, b))
                           for b in bands]
            except KeyError:
                hdu = None
            out.append(hdu)
    if zoom is not None:
        for ib, b in enumerate(bands):
            for i, xx in enumerate(out):
                if xx is not None:
                    shape = np.array(out[i][ib].shape)
                    crop = shape * (1 - 1 / zoom) / 2
                    crop = crop.round().astype(np.int)
                    crop = crop.clip(0, shape // 2 - 1)
                    icrop = [crop[0]] * 2
                    jcrop = [crop[1]] * 2
                    if xx is not None:
                        out[i][ib] = out[i][ib][icrop[0]:-icrop[1],
                                                jcrop[0]:-jcrop[1]]
    return out


def nonparam_images(fn, bands=None):
    return fit_images(fn, bands, extensions=['nonparam'])[0]


def fit_func(f):
    fn = 'fits/%s/fit%s.fits'%(f,f)
    if os.path.exists(fn):
        r = {}
        d = pyfits.getdata('fits/%s/fit%s.fits'%(f,f), 'fit_info')[0]
        ref = d.field('refwlband')
        low = d.field('lowdwlband') + ref
        high = d.field('highdwlband') + ref
        d = pyfits.getdata('fits/%s/fit%s.fits'%(f,f), 'final_cheb')
        for n in d.names:
            r[n] = Chebyshev(d.field(n), (low, high))
    else:
        r = None
    return r


def make_bands_plot(fig, subplot=111, ylabel='', top=True, bottom=True):
    ax1 = fig.add_subplot(*subplot)
    ax2 = ax1.twiny()
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlim)
    if top:
        ax2.set_xlabel('wavelength, \AA')
    else:
        ax2.set_xticklabels([])
    ax2.set_xlim(xlim)
    ax1.set_xticks(w)
    if bottom:
        ax1.set_xticklabels(['${}$'.format(i) for i in bands])
    else:
        ax1.set_xticklabels([])
    pyplot.setp(ax1.get_xticklabels(), va='baseline')
    pyplot.setp(ax1.get_xaxis().get_major_ticks(), pad=20.)
    pyplot.setp(ax1.get_yaxis().get_major_ticks(), pad=8.)
    ax2.xaxis.labelpad = 12
    return ax1


class Sersic:
    # currently doesn't handle uncertainties
    def __init__(self, mag, re, n, ar=1.0, pa=0.0,
                 mag_err=None, re_err=None, n_err=None, ar_err=None, pa_err=None, xc_err=None, yc_err=None):
        self.mag = mag
        self.re = re
        self.n = n
        self.ar = ar
        self.pa = pa
        self.mag_err = mag_err
        self.re_err = re_err
        self.n_err = n_err
        self.ar_err = ar_err
        self.pa_err = pa_err
    def __call__(self, r):
        return self.mu_r(r)
    def mu_r(self, r):
        # Returns the surface brightess at specified major axis radius,
        # within annular ellipses corresponding to the shape of each component individualy
        # Taking, e.g. colours, this currently assumes major axes of components align
        # to be more generally correct need to account for AR, PA, XC, YC,
        # and either select specific vector, or properly compute azimuthal average
        mag = self.mag
        re = self.re
        n = self.n
        bn = self.bn()
        mue = mag + 5.0*log10(re) + 2.5*log10(2.0*pi*n*gamma(2.0*n)*exp(bn)/power(bn, 2.0*n))
        mu = mue + 2.5 * bn / log(10) * (power(r/re, 1.0/n) - 1.0)
        return mu
    def bn(self):
        return gammaincinv(2.0*self.n, 0.5)
    # These need testing
    def I_el(self, r_m, ar_m, pa_m=0):
        return quad(self.I_el_theta, 0, 2*pi, args=(r_m, ar_m, pa_m))[0] / (2*pi)
    def mu_el(self, r_m, ar_m, pa_m=0):
        return -2.5*np.log10(self.I_el(r_m, ar_m, pa_m))
    def mu_el_theta(self, theta, r_m, ar_m, pa_m=0):
        x = r_m * np.cos(theta - pa_m)
        y = ar_m * r_m * np.sin(theta - pa_m)
        r_c = np.sqrt(x**2 + self.ar**2 * y**2)
        return self.mu_r(r_c)
    def I_el_theta(self, theta, r_m, ar_m, pa_m=0):
        return 10**(-0.4*self.mu_el_theta(theta, r_m, ar_m, pa_m))


def plotprof(id=('A1', 'A2'), name='0'):
    print(name, ':', id)
    color = [cm.gist_rainbow(i) for i in np.linspace(1.0, 0.0, 9)]
    func, remax = make_funcs(id)
    fig = pyplot.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.0, wspace=0.0)
    rmax = remax*3.0001
    r = np.arange(rmax/10000.0, rmax, rmax/100.0)
    for i, iid in enumerate(id):
        print(i)
        for j in range(len(func[i])):
            for k, band in enumerate(bands):
                if k == 0:
                    label = "%s\_%i"%(iid, j)
                else:
                    label = ""
                pyplot.plot(r, func[i][j][k](r), linestyle=linestyle[i],
                            marker=None, color=color[k], label=label)
    pyplot.legend(loc='upper right', numpoints=1, prop={'size': 16})
    pyplot.xlabel('$r_{\mathrm{e}}$')
    pyplot.ylabel('$\mu$')
    #fig.gca().invert_yaxis()
    pyplot.xlim(0.0, rmax)
    pyplot.ylim(26, 16)
    fig.savefig('plots/profiles_%s.pdf'%name)


def plotcolprof(id=('A1', 'A2'), name='0'):
    # need to decide and implement consistent annuli in which to determine colour
    # would be nice to plot lines for input model too
    # normalised at remax and offset for display purposes
    print(name, ':', id)
    offset = 0.5
    color = [cm.gist_rainbow(i) for i in np.linspace(1.0, 0.0, 9)]
    func, remax = make_funcs(id)
    fig = pyplot.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.0, wspace=0.0)
    rmax = remax*3.0001
    r = np.arange(rmax/10000.0, rmax, rmax/100.0)
    for i, iid in enumerate(id):
        print(i)
        for k in range(len(bands)-1):
            f1 = f2 = f1max = f2max = 0.0
            for j in range(len(func[i])):
                if k == 0:
                    #label = "%s_%i_%s-%s"%(iid, j, bands[k], bands[k+1])
                    #label = "%s_%i"%(iid, j)
                    label = "%s"%iid
                else:
                    label = ""
                # to use elliptically averaged surface brightnesses will need
                # to supply multi-component fits with single-Sersic info
                f1 += 10**(-0.4*func[i][j][k](r))
                f2 += 10**(-0.4*func[i][j][k+1](r))
                f1max += 10**(-0.4*func[i][j][k](remax))
                f2max += 10**(-0.4*func[i][j][k+1](remax))
            colour = -2.5*np.log10(f1/f2)
            colour_remax = -2.5*np.log10(f1max/f2max)
            colour -= colour_remax
            colour += offset*k
            pyplot.hlines([offset*k], 0.0, rmax, colors='grey')
            pyplot.plot(r, colour, linestyle=linestyle[i],
                        marker=None, color=color[k], label=label)
    pyplot.legend(loc='upper right', numpoints=1, prop={'size': 16})
    pyplot.xlabel('$r_{\mathrm{e}}$')
    pyplot.ylabel('Colour')
    pyplot.ylim(-1*offset, (len(bands)+1) * offset)
    pyplot.xlim(0.0, rmax)
    fig.savefig('plots/colprofiles_%s.pdf'%name)


def make_funcs(id):
    res = [fit_results(i) for i in id]
    func = []
    for i, iid in enumerate(id):
        func.append([])
        remax = 0
        compno = 0
        while True:
            compno += 1
            field = 'COMP%i_MAG'%compno
            if field not in res[i].dtype.names:
                break
            mag = res[i][field]
            re, n, ar, pa, xc, yc =  [res[i]['COMP%i_%s'%(compno, par)] for par in
                                      ('Re', 'n', 'AR', 'PA', 'XC', 'YC')]
            func[i].append([])
            for k, band in enumerate(bands):
                func[i][compno-1].append(Sersic(mag[k], re[k], n[k], ar[k], pa[k], xc[k], yc[k]))
            remax = max(remax, re.max())
    return func, remax


if __name__ =='__main__':
    plot_all()
