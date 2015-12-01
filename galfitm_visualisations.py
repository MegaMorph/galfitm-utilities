#!/usr/bin/env python

from glob import glob
import os
import numpy
import pyfits
import matplotlib
from matplotlib import pyplot
from matplotlib import cm
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

rescolmap = matplotlib.colors.LinearSegmentedColormap.from_list('rescol', ('blue', 'black', 'white', 'red'), N=256, gamma=1.0)
labels = {'MAG': '$m$', 'Re': '$R_e$', 'n': '$n$', 'AR': '$b/a$', 'PA': '$\\theta$'}
marker = ['o', '^', 's', 'D', 'x', '+', '*']
linestyle = [':', '-', '.-', '.-.']

bands = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K']
w = numpy.array([3543,4770,6231,7625,9134,10305,12483,16313,22010], numpy.float)
xlim = (2000, 23000)
#varlist_std = ('MAG', 'Re', 'n', 'AR', 'PA')
varlist_std = ('MAG', 'Re', 'n')


def plot(id=('A2', 'A1'), compno=1, name='0', show_func=False,
         varlist=varlist_std, ylim=ylim_std, sim=sim_std):
    print name, ':', id
    res = [fit_results(i) for i in id]
    if show_func:
        func = [fit_func(i) for i in id]
    else:
        func = None
    nvar = len(varlist)
    fig = pyplot.figure(figsize=(5, 15))
    fig.subplots_adjust(bottom=0.05, top=0.94, left=0.2, right=0.95, hspace=0.075)
    for i, v in enumerate(varlist):
        ax = make_bands_plot(fig, (5, 1, i+1), labels[v], i==0, i==nvar-1)
        if v in sim.keys():
            pyplot.plot(w, sim[v], '-k', alpha=0.75)
        plotres(res, id, 'COMP%i_%s'%(compno, v), func)
        if v in sim.keys():
            pyplot.plot(w, sim[v], 'xk', markersize=10.0, alpha=0.75)
        pyplot.ylim(ylim[v])
        if i==0:
            pyplot.legend(loc='lower right', numpoints=1, prop={'size': 16})
    fig.savefig('plots/illustration_%s.pdf'%name)
    pyplot.close('all')
    if compno==1:
        plotimg(id, name)
        plotcolimg(id, name)
        plotprof(id, name)
        plotcolprof(id, name)
        npid = [j for j in id if j[0] == 'N' and j[-1] in 'nm']
        if len(npid) > 0:
            plotnonparamcolimg(npid, name)


def plotimg(fn, name=None):
    cmap_img = pyplot.cm.gray
    cmap_res = rescolmap
    norm_res = None
    nbands = len(bands)
    nid = len(id)
    fig = pyplot.figure(figsize=(15.0/nbands * (1+nid*2), 15))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.0, wspace=0.0)
    for i, iid in enumerate(id):
        img = fit_images(iid)
        if i == 0:
            vmin = []
            vmax = []
            vrange = []
            for ib, b in enumerate(bands):
                ax = fig.add_subplot(nbands, 1+2*len(id), 1+ib*(1+nid*2)+i*2)
                if ib==nbands-1:
                    ax.set_xlabel('image', fontsize=labelsize)
                ticksoff(ax)
                vmin.append(scoreatpercentile(img[0][ib].ravel(), 0.1))
                vmax.append(scoreatpercentile(img[0][ib].ravel(), 99.9))
                vrange.append(scoreatpercentile(img[2][ib].ravel(), 99.9) - scoreatpercentile(img[2][ib].ravel(), 0.1))
                pyplot.imshow(img[0][ib][::-1], cmap=cmap_img, vmin=vmin[ib], vmax=vmax[ib], interpolation='nearest')
                ax.set_ylabel('$%s$'%b, fontsize=labelsize)
        for ib, b in enumerate(bands):
            ax = fig.add_subplot(nbands, 1+2*len(id), 2+ib*(1+nid*2)+i*2)
            if ib==nbands-1:
                ax.set_xlabel('model %s'%iid, fontsize=labelsize)
            ticksoff(ax)
            pyplot.imshow(img[1][ib][::-1], cmap=cmap_img, vmin=vmin[ib], vmax=vmax[ib], interpolation='nearest')
        for ib, b in enumerate(bands):
            ax = fig.add_subplot(nbands, 1+2*len(id), 3+ib*(1+nid*2)+i*2)
            if ib==nbands-1:
                ax.set_xlabel('residual %s'%iid, fontsize=labelsize)
            ticksoff(ax)
            pyplot.imshow(img[2][ib][::-1], cmap=cmap_res, norm=norm_res, vmin=-vrange[ib], vmax=vrange[ib], interpolation='nearest')
    if name is None:
        name = os.path.splitext(os.path.basename(fn))[0]
    fig.savefig('plots/images_%s.pdf'%name)
    pyplot.close('all')


def plotcolimg(id, name='0', rgb='Hzg', desaturate=True, pedestal=0):
    nbands = len(bands)
    nid = len(id)
    beta = 2.5
    scales = numpy.array((0.04, 0.055, 0.2))
    # offsets not so necessary now have nice desaturation feature working
    offsets = numpy.array([75.0, 40.0, 8.0]) * 0.5
    fig = pyplot.figure(figsize=(15.0/nbands * (1+nid*2), 15))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.0, wspace=0.0)
    for i, iid in enumerate(id):
        img = fit_images(iid, rgb)
        img[0] = [img[0][j] - offsets[j] for j in range(3)]
        img[1] = [img[1][j] - offsets[j] for j in range(3)]
        img[2] = [img[2][j] + scales[j]*2*offsets.mean() for j in range(3)]
        if i == 0:
            ax = fig.add_subplot(nbands, 1+2*nid, 1+i*2)
            ticksoff(ax)
            ax.set_xlabel('image', fontsize=labelsize)
            colimg = RGBImage(*img[0], scales=scales, beta=beta,
                              desaturate=desaturate, pedestal=pedestal).img
            pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, 1+2*nid, 2+i*2)
        ticksoff(ax)
        ax.set_xlabel('model %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*img[1], scales=scales, beta=beta,
                          desaturate=False, pedestal=pedestal).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, 1+2*nid, 3+i*2)
        ticksoff(ax)
        ax.set_xlabel('residual %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*img[2], scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
    fig.savefig('plots/colimages_%s.pdf'%name)
    pyplot.close('all')


def plotnonparamcolimg(id, name='0', rgb='Hzg', desaturate=True, pedestal=0):
    nbands = len(bands)
    nid = len(id)
    beta = 2.5
    scales = numpy.array((0.04, 0.055, 0.2))
    # offsets not so necessary now have nice desaturation feature working
    offsets = numpy.array([75.0, 40.0, 8.0]) * 0.5
    fig = pyplot.figure(figsize=(15.0/nbands * (1+nid*2), 15))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.0, wspace=0.0)
    original_iid = None
    for i, iid in enumerate(id):
        # First row, results without nonparam
        if original_iid != iid[:-1]:
            original_iid = iid[:-1]
            print original_iid
            img = fit_images(original_iid, rgb)
            img[0] = [img[0][j] - offsets[j] for j in range(3)]
            img[1] = [img[1][j] - offsets[j] for j in range(3)]
            img[2] = [img[2][j] + scales[j]*2*offsets.mean() for j in range(3)]
            if i == 0:
                ax = fig.add_subplot(nbands, 1+2*nid, 1+i*2)
                ticksoff(ax)
                ax.set_title('image', fontsize=labelsize)
                colimg = RGBImage(*img[0], scales=scales, beta=beta,
                                  desaturate=desaturate, pedestal=pedestal).img
                pyplot.imshow(colimg, interpolation='nearest', origin='lower')
            ax = fig.add_subplot(nbands, 1+2*nid, 2+i*2)
            ticksoff(ax)
            ax.set_title('model %s'%original_iid, fontsize=labelsize)
            colimg = RGBImage(*img[1], scales=scales, beta=beta,
                              desaturate=desaturate, pedestal=pedestal).img
            pyplot.imshow(colimg, interpolation='nearest', origin='lower')
            ax = fig.add_subplot(nbands, 1+2*nid, 3+i*2)
            ticksoff(ax)
            ax.set_title('residual %s'%original_iid, fontsize=labelsize)
            colimg = RGBImage(*img[2], scales=scales, beta=beta,
                              desaturate=desaturate).img
            pyplot.imshow(colimg, interpolation='nearest', origin='lower')
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
        img = nonparam_images(iid, rgb)
        #img[0] = [img[0][j] + scales[j]*2*offsets.mean() for j in range(3)]
        img[1] = [img[1][j] - offsets[j] for j in range(3)]
        ax = fig.add_subplot(nbands, 1+2*nid, 2+4*nid+2+i*2)
        ticksoff(ax)
        ax.set_xlabel('nonparam %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*img[0], scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
        ax = fig.add_subplot(nbands, 1+2*nid, 2+4*nid+3+i*2)
        ticksoff(ax)
        ax.set_xlabel('datasub %s'%iid, fontsize=labelsize)
        colimg = RGBImage(*img[1], scales=scales, beta=beta,
                          desaturate=desaturate).img
        pyplot.imshow(colimg, interpolation='nearest', origin='lower')
    fig.savefig('plots/nonparamcolimages_%s.pdf'%name)
    pyplot.close('all')


def ticksoff(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def plotres(res, id, field, func=None, norm=None):
    nid = len(id)
    #mec = ['black', None] * (1+nid//2)
    #mfc = ['white', 'black'] * (1+nid//2)
    #color = ['grey', 'grey'] * (1+nid//2)
    mec_func = ['DeepSkyBlue' ,'DarkGreen', 'Orange']
    mfc_func = ['DeepSkyBlue', 'white', 'white']
    color = ['DeepSkyBlue', 'DarkGreen', 'Orange']
    mec_nofunc = ['MediumPurple', 'MediumSeaGreen', 'Pink']
    mfc_nofunc = ['MediumPurple', 'MediumSeaGreen', 'Pink']
    ymin, ymax = (1e99, -1e99)
    for i, iid in enumerate(id):
        if nid%2 == 0:
            x = w + 100 * (1+i//2) * (-1)**i
        else:
            x = w + 100 * (1+i//2) * (-1)**i
        if func is not None and func[i] is not None:
            mec = mec_func
            mfc = mfc_func
            f = func[i][field]
            if norm is not None:
                f /= norm[i][field]
            plotfunc(f, wlfunc=wlfuncs.get(iid), color=color[i])
        else:
            mec = mec_nofunc
            mfc = mfc_nofunc
        r = res[i][field]
        if norm is not None:
            r /= norm[i][field](x)
        pyplot.errorbar(x, r, res[i][field+'_ERR'], color=mec[i],
                        marker=marker[i//2], mec=mec[i], markerfacecolor=mfc[i], linestyle='',
                        label=iid)
        ymin = min(ymin, (res[i][field]-res[i][field+'_ERR']).min())
        ymax = max(ymax, (res[i][field]+res[i][field+'_ERR']).max())
    yrange = ymax - ymin
    ymin -= 0.05 * yrange
    ymax += 0.05 * yrange
    pyplot.ylim(ymin, ymax)

def plotfunc(func, wlfunc=None, color='red', label=''):
    dx = (xlim[1] - xlim[0]) / 1000.0
    x = numpy.arange(xlim[0], xlim[1]+dx/2.0, dx)
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
        r = numpy.concatenate([pyfits.getdata(fn.format(b), 'final_band')
                              for b in bands])
    return r


def fit_images(fn, bands):
    """Get the images from the specified galfit(m) filename(s)

    `bands` should be a list of band id strings.
    If fn includes a format placeholder `{}`, this will be replaced by each
    band id in turn. Otherwise all bands are assumed to be in a single file.
    """
    # TODO: add auto-discovery of bands for galfitm files
    if '{}' in fn:
        original = [pyfits.getdata(fn, 'input_{}'.format(b)) for b in bands]
        model = [pyfits.getdata(fn, 'model_{}'.format(b)) for b in bands]
        residual = [pyfits.getdata(fn, 'residual_{}'.format(b)) for b in bands]
    else:
        original = [pyfits.getdata(fn.format(b), 'input_x') for b in bands]
        model = [pyfits.getdata(fn.format(b), 'model_x') for b in bands]
        residual = [pyfits.getdata(fn.format(b), 'residual_x') for b in bands]
    return [original, model, residual]


def nonparam_images(f, bands=bands):
    fn = 'fits/%s/fit%s.fits'%(f,f)
    r = None
    if os.path.exists(fn):
        nonparam = [pyfits.getdata(fn, 'nonparam_%s'%b) for b in bands]
        datasub = [pyfits.getdata(fn, 'datasub_%s'%b) for b in bands]
    else:
        nonparam = [pyfits.getdata('fits/%s/fit%s%s.fits'%(f,f, b), 'nonparam_x') for b in bands]
        datasub = [pyfits.getdata('fits/%s/fit%s%s.fits'%(f,f, b), 'datasub_x') for b in bands]
    return [nonparam, datasub]

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
        ax1.set_xticklabels(['$'+i+'$' for i in bands])
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
        return -2.5*numpy.log10(self.I_el(r_m, ar_m, pa_m))
    def mu_el_theta(self, theta, r_m, ar_m, pa_m=0):
        x = r_m * numpy.cos(theta - pa_m)
        y = ar_m * r_m * numpy.sin(theta - pa_m)
        r_c = numpy.sqrt(x**2 + self.ar**2 * y**2)
        return self.mu_r(r_c)
    def I_el_theta(self, theta, r_m, ar_m, pa_m=0):
        return 10**(-0.4*self.mu_el_theta(theta, r_m, ar_m, pa_m))


def plotprof(id=('A1', 'A2'), name='0'):
    print name, ':', id
    color = [cm.gist_rainbow(i) for i in numpy.linspace(1.0, 0.0, 9)]
    func, remax = make_funcs(id)
    fig = pyplot.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.0, wspace=0.0)
    rmax = remax*3.0001
    r = numpy.arange(rmax/10000.0, rmax, rmax/100.0)
    for i, iid in enumerate(id):
        print i
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
    print name, ':', id
    offset = 0.5
    color = [cm.gist_rainbow(i) for i in numpy.linspace(1.0, 0.0, 9)]
    func, remax = make_funcs(id)
    fig = pyplot.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.0, wspace=0.0)
    rmax = remax*3.0001
    r = numpy.arange(rmax/10000.0, rmax, rmax/100.0)
    for i, iid in enumerate(id):
        print i
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
            colour = -2.5*numpy.log10(f1/f2)
            colour_remax = -2.5*numpy.log10(f1max/f2max)
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
