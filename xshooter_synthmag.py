#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import os
from scipy.integrate import simps
from astropy.io import fits
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 50000

#%%
#%%
class Filter():
    def __init__(self, name='(unnamed)'):
        super(Filter, self).__init__()
        self.name = name
        self.wave = None
        self.thru = None

    def read(self, file):
        [self.wave, self.thru] = np.loadtxt(file).transpose()

    def extend(self, min, max, res):
        wavemin = self.wave.min()
        wavemax = self.wave.max()
        if (min < wavemin):
            w = np.arange(min, wavemin, res)
            self.wave = np.hstack((w, self.wave))
            self.thru = np.hstack((np.zeros(len(w)), self.thru))

        if (max > wavemax):
            w = np.arange(wavemax, max, res) + res
            self.wave = np.hstack((self.wave, w))
            self.thru = np.hstack((self.thru, np.zeros(len(w))))


def getrad(lum,temp):
    sb = 5.67e-5 #grams s^-3 kelvin^-4
    lsun = 3.8e33 #erg/s 
    l = lsun*(10**lum) #luminosity from isochrone is in log(L/lsun)
    t = np.round(10**temp) #teff from isochrone is in log(teff)
    radius = np.sqrt(l/(4*np.pi*sb*t**4))
    return radius

###
## ok so I think that we want to actually compute colors - don't need that radius dependence!
###
def synthmag(spectra, filte,lum,temp, dered=False):
    '''
    useful resource: L. Casagrande, Don A. VandenBerg
    Monthly Notices of the Royal Astronomical Society, Volume 444, Issue 1, 11 October 2014, 
    Pages 392–419, https://doi.org/10.1093/mnras/stu1476

    pass in a filter (filte), which is just a dataframe/structured array with columns of wavelength and throughput. Here, 
    the format is filte.wave for wavelength dimension, filte.thru for throughput dimension. You also pass in a spectrum 
    with columns of wavelength and flux (here self.wave, self.flux). Ensure your flux units are flam, and your wavelength
    units are angstroms. You also pass in a luminosity (given as log L/Lsun) and temperature (log Teff) that you get from 
    either divine knowledge or an isochrone. These are used to compute the radius of the star, which scales the flux 
    (Radius of the star / 10 pc)^2 scales the flux such that we get out absolute magnitudes.
    '''
    if dered == False:
        flux = spectra.data['FLUX']
    elif dered == True:
        flux = spectra.data['FLUX_DR']
    flux = np.where(np.isnan(flux), 0, flux)
    #xshooter spectra have wavelenth in nm, flux in erg/s/cm^2/Å, multiply nm by 10 to go to angstrom
    spec = pysynphot.spectrum.ArraySourceSpectrum(wave=spectra.data['WAVE']*10, flux=flux, keepneg=True, fluxunits='flam')
    filt = pysynphot.spectrum.ArraySpectralElement(filte.wave, filte.thru, waveunits='angstrom')
    #normalising spectra
    #getting bounds of integral
    lam = spec.wave[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
    T = np.interp(lam,filt.wave,filt.throughput)
    T = np.where(T<.001, 0, T)
    R = getrad(lum,temp)
    #1/(3.08567758128*10**(19))**2 is just 1/10pc^2 in cm! (1/(3.086e19)**2)
    s = spec.flux[(spec.wave<=filt.wave.max())&(spec.wave>=filt.wave.min())]
    s = s*(R/3.086e19)**2
    #interpolating to get filter data on same scale as spectral data
    #doin classic integral to get flux in bandpass
    stzp = 3.631e-9

    a = -2.5*np.log10((simps(s*T*lam,lam)/(stzp*simps(T*lam,lam))))
    b = -2.5*np.log10((simps(T*lam,lam)/simps(T/lam,lam)))
    return a+b+18.6921#+.075
#%%

#%%

# %%
def synthcolor(spectra, filte1, filte2, dered=False):
    '''
    L. Casagrande, Don A. VandenBerg
    Monthly Notices of the Royal Astronomical Society, Volume 444, Issue 1, 11 October 2014, 
    Pages 392–419, https://doi.org/10.1093/mnras/stu1476

    pass in a filter (filte), which is just a dataframe/structured array with columns of wavelength and throughput. Here, 
    the format is filte.wave for wavelength dimension, filte.thru for throughput dimension. You also pass in a spectrum 
    with columns of wavelength and flux (here self.wave, self.flux). Ensure your flux units are flam, and your wavelength
    units are angstroms. You also pass in a luminosity (given as log L/Lsun) and temperature (log Teff) that you get from 
    either divine knowledge or an isochrone. These are used to compute the radius of the star, which scales the flux 
    (Radius of the star / 10 pc)^2 scales the flux such that we get out absolute magnitudes.
    '''
    if dered == False:
        flux = spectra.data['FLUX']
    elif dered == True:
        if 'FLUX_SC' in spectra.data.names:
            flux = spectra.data['FLUX_SC'] 
        elif 'FLUX_DR' in spectra.data.names:
            flux = spectra.data['FLUX_DR'] ## different data labels  -
    flux = np.where(np.isnan(flux), 0, flux)
    #xshooter spectra have wavelenth in nm, flux in erg/s/cm^2/Å, multiply nm by 10 to go to angstrom
    spec = pysynphot.spectrum.ArraySourceSpectrum(wave=spectra.data['WAVE']*10, flux=flux, keepneg=False, fluxunits='flam') #changed keepneg to False
    filt1 = pysynphot.spectrum.ArraySpectralElement(filte1.wave, filte1.thru, waveunits='angstrom')
    filt2 = pysynphot.spectrum.ArraySpectralElement(filte2.wave, filte2.thru, waveunits='angstrom')
    #normalising spectra
    #getting bounds of integral
    lam1 = spec.wave[(spec.wave<=filt1.wave.max())&(spec.wave>=filt1.wave.min())]
    T1 = np.interp(lam1,filt1.wave,filt1.throughput)
    T1 = np.where(T1<.001, 0, T1)
    s1 = spec.flux[(spec.wave<=filt1.wave.max())&(spec.wave>=filt1.wave.min())]

    #interpolating to get filter data on same scale as spectral data
    #doin classic integral to get flux in bandpass
    stzp = 3.631e-9

    a1 = -2.5*np.log10((simps(s1*T1*lam1,lam1)/(stzp*simps(T1*lam1,lam1))))
    b1 = -2.5*np.log10((simps(T1*lam1,lam1)/simps(T1/lam1,lam1)))

    lam2 = spec.wave[(spec.wave<=filt2.wave.max())&(spec.wave>=filt2.wave.min())]
    T2 = np.interp(lam2,filt2.wave,filt2.throughput)
    T2 = np.where(T2<.001, 0, T2)
    s2 = spec.flux[(spec.wave<=filt2.wave.max())&(spec.wave>=filt2.wave.min())]

    a2 = -2.5*np.log10((simps(s2*T2*lam2,lam2)/(stzp*simps(T2*lam2,lam2))))
    b2 = -2.5*np.log10((simps(T2*lam2,lam2)/simps(T2/lam2,lam2)))

    return (a1+b1+18.6921) - (a2+b2+18.6921)

#%%
# %%
