# galfitm-utilities
Utilities for galfitm (mostly python)

##galfitm-visualisations

Work in progress...

    import galfitm_visualisations as gv

### plotimg

A couple of examples...

Figures are output in a folder named `plots` in the folder from which the code
is run, so you may need to create this first.

Create a pdf figure of the input, model and residual from a single-band `galfitm` output
fits file.  Automatic scaling should be reasonable, but you can also specify
various options manually.

    gv.plotimg('imgblock.fits')

To include nonparametric output, 

    gv.plotimg('imgblock.fits', nonparam=1)

If you have several different fits you want to compare, say single-sersic, bulge+disk, bulge+disk+bar:

    gv.plotimg(('ss.fits', 'bd.fits', 'bdb.fits'))

If two of these have nonparametric images enabled:

    gv.plotimg(('ss.fits', 'bd.fits', 'bdb.fits'), nonparam=2)

If you have performed multi-band fits, then just specify the band labels you want to
plot:

    gv.plotimg(('ss.fits', 'bd.fits', 'bdb.fits'), bands=('g', 'r', 'i'))

To make an equaivalent figure based on single-band fits, you can specify the
location of the band name in the filename with `{}`:

    gv.plotimg(('ss_{}.fits', 'bd_{}.fits', 'bdb_{}.fits'), bands=('g', 'r', 'i'))

This functionality can be repurposed to make a single figure for several different galaxies.
Say you have a list of object IDs in `ids`, corresponding to a set of normal
(named `fits/1C_sersic/1C_{}.fits`) and nonparametric
(`fits/1C_sersic/1C_{}.fits`) `galfitm` output FITS files for single-band fits
to SDSS galaxies.
Then you can could create a nice figure with:

    gv.plotimg(('fits/1C_sersic/1C_{}.fits', 'fits/1C_sersic/1C_{}_nonparam.fits'),
                      bands=ids, zoom=2, uservmin=-10, uservmax=1000, usernorm=20,
                      uservrange=1000, nonparam=1, label=True, ids=('normal', 'nonparam'), 
                      name='1C_sersic')
                      
