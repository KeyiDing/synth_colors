from pfsspec.obsmod.filter import Filter
import pandas as pd
import numpy as np


filt_nb515 = Filter()
filt_nb515.read('/scratch/ceph/dobos/data/pfsspec/subaru/filters/v2/HSC-NB515.txt')


filt_g = Filter()
filt_g.read('/home/cfilion1/nb515/pfs_isochrones/hsc_g.txt')


filt_i = Filter()
filt_i.read('/home/cfilion1/nb515/pfs_isochrones/hsc_i.txt')

qe = pd.read_csv('/home/cfilion1/nb515/pfs_isochrones/qe.txt', sep='\s+')
dewar = pd.read_csv('/home/cfilion1/nb515/pfs_isochrones/dewar.txt', sep='\s+')
popt2 = pd.read_csv('/home/cfilion1/nb515/pfs_isochrones/popt2.txt', sep='\s+')
mirror = pd.read_csv('/home/cfilion1/nb515/pfs_isochrones/mirror.txt', sep='\s+', comment='#', header=None,
                    names=['wavelength(nm)','a','b','c','d']) 
mirror['mean_transmission'] = (mirror.a + mirror.b + mirror.c + mirror.d)/4
mirror['#wavelength[angstrom]'] = mirror['wavelength(nm)']*10



def transmission(filt, qe, dewar, popt2, mirror):
    #interpolating quantum efficiency on the filter wavelength range so it's same length as filter
    lam_qe = qe['#wavelength[angstrom]'][(qe['#wavelength[angstrom]']<=filt.wave.max())&
                                  (qe['#wavelength[angstrom]']>=filt.wave.min())]
    q = qe['QE'][(qe['#wavelength[angstrom]']<=filt.wave.max())&
                                  (qe['#wavelength[angstrom]']>=filt.wave.min())]
    qe_filt = np.interp(filt.wave,lam_qe,q)
    
    #interpolating dewar window on filter wavelength range
    lam_dew = dewar['#wavelength[angstrom]'][(dewar['#wavelength[angstrom]']<=filt.wave.max())&
                                  (dewar['#wavelength[angstrom]']>=filt.wave.min())]
    thru = dewar['transmittance'][(dewar['#wavelength[angstrom]']<=filt.wave.max())&
                                  (dewar['#wavelength[angstrom]']>=filt.wave.min())]
    dew_filt = np.interp(filt.wave,lam_dew,thru)
    
    #interpolating popt2 on filter wavelength range
    lam_popt = popt2['#wavelength[angstrom]'][(popt2['#wavelength[angstrom]']<=filt.wave.max())&
                                  (popt2['#wavelength[angstrom]']>=filt.wave.min())]
    thru = popt2['transmittance'][(popt2['#wavelength[angstrom]']<=filt.wave.max())&
                                  (popt2['#wavelength[angstrom]']>=filt.wave.min())]
    popt2_filt = np.interp(filt.wave,lam_popt,thru)
    
    #interpolating mirror on filter wavelength range
    lam_mir = mirror['#wavelength[angstrom]'][(mirror['#wavelength[angstrom]']<=filt.wave.max())&
                                  (mirror['#wavelength[angstrom]']>=filt.wave.min())]
    thru = mirror['mean_transmission'][(mirror['#wavelength[angstrom]']<=filt.wave.max())&
                                  (mirror['#wavelength[angstrom]']>=filt.wave.min())]
    mirror_filt = np.interp(filt.wave,lam_mir,thru)/100
    
    #multiplying everything together
    full_trans = filt.thru*qe_filt*dew_filt*popt2_filt*mirror_filt
    #new_filter = pd.DataFrame(columns=['wave','thru'])
    #new_filter.thru = full_trans
    #new_filter.wave = filt.wave
    data_array = np.array([filt.wave, full_trans]).T
    return pd.DataFrame(data=data_array, columns=['#wavelength[angstrom]','transmittance'])

'''
#how to run:
g = transmission(filt_g, qe, dewar, popt2, mirror)
g.to_csv('/home/cfilion1/nb515/pfs_isochrones/hsc_g_full_throughput.txt',sep="\t", index=False)

filter_hsc_g = Filter()
filter_hsc_g.read('/home/cfilion1/nb515/pfs_isochrones/hsc_g_full_throughput.txt')


i = transmission(filt_i, qe, dewar, popt2, mirror)
i.to_csv('/home/cfilion1/nb515/pfs_isochrones/hsc_i_full_throughput.txt',sep="\t", index=False)

filter_hsc_i = Filter()
filter_hsc_i.read('/home/cfilion1/nb515/pfs_isochrones/hsc_i_full_throughput.txt')

nb = transmission(filt_nb515, qe, dewar, popt2, mirror)
nb.to_csv('/home/cfilion1/nb515/pfs_isochrones/hsc_nb515_full_throughput.txt',sep="\t", index=False)

filter_hsc_nb515 = Filter()
filter_hsc_nb515.read('/home/cfilion1/nb515/pfs_isochrones/hsc_nb515_full_throughput.txt')
'''