#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from xshooter_synthmag import Filter, synthcolor
from astropy.io import fits
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 50000
plt.rcParams.update({'font.size': 22})
#%%

#%%
#directory of where the spectra are
DATA_DIR = '/datascope/subaru/data/catalogs/xshooter/fits/'

#load the filters
filt_g = Filter()
filt_g.read('/datascope/subaru/data/pfsspec/subaru/hsc/filters/fHSC-g.txt')

filt_i = Filter()
filt_i.read('/datascope/subaru/data/pfsspec/subaru/hsc/filters/fHSC-i.txt')


filt_nb515 = Filter()
filt_nb515.read('/datascope/subaru/data/pfsspec/subaru/hsc/filters/fHSC-NB515.txt')

combined = pd.read_csv('/home/cfilion1/Galaxia/ursaminor_hsc_cleaned.csv')

   
#%%
#attempting to get /all/ stars



import glob
#get the table that gives the stellar parameters for the stars in xshooter dr2 
table = Vizier.get_catalogs('J/A+A/627/A138/tablea1')
stellar_params = table[0].to_pandas()

#get the table of stars that have comments/flags from dr2
tab2 = Vizier.get_catalogs('J/A+A/660/A34/table')
comment_flags = tab2[0].to_pandas()
#select only the stars that have flags! This table is now just a table of 'bad' stars
comment_flags = comment_flags[comment_flags.Com != ''] #finding stars with comments!


#get the set of xshooter merged and merged_scl files, these should each have extinction
#corrected spectra
merged_files = glob.glob("/datascope/subaru/data/catalogs/xshooter/fits/*merged.fits")
scl_files = glob.glob("/datascope/subaru/data/catalogs/xshooter/fits/*merged_scl.fits")
files = merged_files + scl_files

from astroquery.simbad import Simbad
customSimbad = Simbad()
#to learn what fields we can get from simbad, run below to lines:
#Simbad.get_field_description('otypes')
#Simbad.list_votable_fields()
#We want to get information on stellar types, metallicity, etc
customSimbad.add_votable_fields('otype','otypes', 'mk','fe_h','main_id')

#table from verro et al xshooter dr3 paper with Mdwarf parameters -
#this info is transcribed!
#important note - the star name from the xshooter dr3 table is different from
#the HNAME from the xshooter file, the dr3 table name should be the correct one
mdwarf_params = pd.read_csv('adtl_mdwarf_params.txt',sep='\t')
#%%
#%%
name_list = np.array([])
g_i_color = np.array([])
nb_g_color = np.array([])
fehs = np.array([])
teffs = np.array([])
loggs = np.array([])
flag = np.array([])
values = np.array([])
sp_type = np.array([])
sec_sp_type = np.array([])
var_flag = np.array([])
np_var_flag = np.array([])
mira_lpv_flag = np.array([])
#labels that lpv or mira stars have in simbad, 'micheck' is primary name and 'micheck_sec'
#is for set of secondary labels. note I'm being generous here by including candidate and '?' stars
micheck = np.array(['Mira', 'LongPeriodV*_Candidate','LongPeriodV*'])
micheck_sec = np.array(['LP?','Mi*','LP*','Mi?'])

#simbad labels for all variable type stars that are in the x-shooter spectra
mainvar_set = np.array(['PulsV*', 'Variable*', 'Mira', 'delSctV*','RSCVnV*', 'SB*', 
                'LongPeriodV*_Candidate','LongPeriodV*', 'gammaDorV*', 'alf2CVnV*',
                'RRLyrae', 'RVTauV*', 'ClassicalCep','EllipVar', 'Type2Cep', 'RCrBV*'])
secvar_set = np.array(['LP?','Mi*','dS*','LP*','Mi?'])
xshoot_name = np.array([])

#loop through files
for i in range(len(files)):
    fn = os.path.join(DATA_DIR, files[i])
    hdus = fits.open(fn, memmap=False)
    xshoot_name = np.append(xshoot_name,hdus[0].header['XSL_ID'])
    # the new m-dwarf stars that are added in x-shooter dr3 all have 'M' in their x-shooter id.
    # these stars arent in the dr2 catalogue and need to be handled separately.
    if 'M' in hdus[0].header['XSL_ID']:
        # searching simbad for info on the star, using the star's name in the mdwarf dr3 table! theres a lot going on here:
        # I need to get the name for the star from the mdwarf dr3 table using the info in the xshooter spectra
        # header (hdus[0].header, remembering that for some reason the hdu HNAMES and names in the mdwarf dr3 table 
        # don't match), so I am finding the star in the mdwarf dr3 table that has the same right ascension as the 
        # xshooter spectra we have selected - the RA value should be identical. Finally, I search simbad using the
        # name of the star given in the mdwarf dr3 table (column 'Star').
        resm = customSimbad.query_object(mdwarf_params[(mdwarf_params.RA.str.contains(hdus[0].header['RA_FULL']))|
                        (mdwarf_params.RA==hdus[0].header['RA_FULL'])].Star.values[0])
        #add the type label that we get from simbad to the array of type names. 'OTYPE' is main name on simbad
        sp_type = np.append(sp_type, resm['OTYPE'][0])
        #add the secondary type label to our array of type names. 'OTYPES' is secondary name on simbad
        sec_sp_type = np.append(sec_sp_type, resm['OTYPES'][0])
        #check if this type is a variable using the array of names of variable types we defined above.
        # if it is variable, flag it!
        if resm['OTYPE'][0] in mainvar_set:
                var_flag = np.append(var_flag, 1)
        #if the main type is not variable, check if the secondary type is
        else:
                #check if secondary names are in the array of secondary names of variable types we defined above.
                #note 'OTYPES' is secondary name, and the 'OTYPES' is a string with (usually) a few types
                # here we are checking if any part of the 'OTYPES' for this star is in the list of secondary names
                if any(param in resm['OTYPES'][0] for param in secvar_set):
                        var_flag = np.append(var_flag, 1)
                #if niether of the main and secondary types are variable, no flag! not a variable star
                else:
                        var_flag = np.append(var_flag, 0)
        #similar to what we just did above, but now checking if the type is a mira or lpv. if main or secondary
        #type indicate that the star is/could be an lpv or mira, we flag it
        if resm['OTYPE'][0] in micheck:
                mira_lpv_flag = np.append(mira_lpv_flag, 1)
        else: 
                if any(param in resm['OTYPES'][0] for param in micheck_sec):
                        mira_lpv_flag = np.append(mira_lpv_flag, 1)
                else:
                        mira_lpv_flag = np.append(mira_lpv_flag, 0)
        #add both the name from the mdwarf dr3 table and the name from the xshooter spectra header to
        #our array of stellar names
        name_list = np.append(name_list, str(hdus[0].header['HNAME'])+' or '+ str(mdwarf_params[(mdwarf_params.RA.str.contains(hdus[0].header['RA_FULL']))|
                                (mdwarf_params.RA==hdus[0].header['RA_FULL'])].Star.values[0]))
        #find logg from the mdwarf dr3 table, add it to our array of logg info
        #(note I take the mean of logg, feh etc in case there are multiple obsevations/measurements)
        logg = mdwarf_params[(mdwarf_params.RA.str.contains(hdus[0].header['RA_FULL']))|
                                (mdwarf_params.RA==hdus[0].header['RA_FULL'])].logg.mean()
        loggs = np.append(loggs, logg)
        #some of the stars in the mdwarf dr3 table have '<NA>' values in their teff estimate. here, we
        #find teff from the mdwarf dr3 table and check if it is an actual value, or if it is '<NA>'.
        #if it is an actual value, we add it to the array of teffs, if it is not we just add 'nan' to the
        #array of teffs instead
        teff = mdwarf_params[(mdwarf_params.RA.str.contains(hdus[0].header['RA_FULL']))|
                                (mdwarf_params.RA==hdus[0].header['RA_FULL'])].Teff
        if teff.astype(str).str.contains('<NA>').any():
                teffs = np.append(teffs,np.NaN)
        else:
                teffs = np.append(teffs,teff.mean())
        #find logg from the mdwarf dr3 table, add it to our array of logg info
        feh = mdwarf_params[(mdwarf_params.RA.str.contains(hdus[0].header['RA_FULL']))|
                                (mdwarf_params.RA==hdus[0].header['RA_FULL'])]['[Fe/H]e'].mean()
        fehs = np.append(fehs, feh)
        #compute the synthetic g-i and nb - g color, add each color to the respective array of
        # synthetic colors
        g_i_color = np.append(g_i_color, synthcolor(hdus[1], filt_g, filt_i, dered=True))
        nb_g_color = np.append(nb_g_color, synthcolor(hdus[1], filt_nb515, filt_g, dered=True))
        #setting flags - all of the mdwarf stars should have values, so set values flag to zero
        #similarly, these stars don't have flags in their spectra
        values = np.append(values, 0)
        flag = np.append(flag, 0)
    #if there isn't 'M' in the xshooter id, the star is not a new mdwarf from the dr3 catalogue and we can
    #proceed using xshooter dr2 table 
    else:
        #the HNAME in the xshooter spectra headers for the remaining stars do match the names given in
        #the dr2 table, so no need to do anything fancy here
        #add the name of the star to our array of names
        name_list = np.append(name_list, hdus[0].header['HNAME'])
        #query simbad for info on this star, using the HNAME
        res = customSimbad.query_object(hdus[0].header['HNAME'])
        #interestingly, not /all/ stars are in simbad. if nothing is found in simbad,
        #add 'NaN' to the array of main and secondary types, set the mira and variable flags to zero
        if type(res) == type(None): 
                sp_type = np.append(sp_type, np.NaN)
                sec_sp_type = np.append(sec_sp_type, np.NaN)
                var_flag = np.append(var_flag, 0)
                mira_lpv_flag = np.append(mira_lpv_flag, 0)
        #if we get a result from simbad, add the 'OTYPE' to our array of types, and 'OTYPES' to secondary types array
        else:
                sp_type = np.append(sp_type, res['OTYPE'][0])
                sec_sp_type = np.append(sec_sp_type, res['OTYPES'][0])
                #again checking if the main type is variable, then checking if secondary type is variable. Set 
                #flags accordingly
                if res['OTYPE'][0] in mainvar_set:
                        var_flag = np.append(var_flag, 1)
                        print(hdus[0].header['HNAME'], 'is variable')
                else:

                        if any(param in res['OTYPES'][0] for param in secvar_set):
                                var_flag = np.append(var_flag, 1)
                                print(hdus[0].header['HNAME'], 'is variable')
                        else:
                                var_flag = np.append(var_flag, 0)
                # checking if mira or LPV, same as above
                if res['OTYPE'][0] in micheck:
                        mira_lpv_flag = np.append(mira_lpv_flag, 1)
                else: 
                        if any(param in res['OTYPES'][0] for param in micheck_sec):
                                mira_lpv_flag = np.append(mira_lpv_flag, 1)
                        else:
                                mira_lpv_flag = np.append(mira_lpv_flag, 0)
        #find the star in the table of stellar parameters from xshooter dr2
        #here, we use the fact that the HNAMES should match. first we check that this star is in the table
        #of stellar parameters at least once (some stars are not in this catalogue, see 'else' below)
        if len(stellar_params[(stellar_params['HNAME'].str.contains(hdus[0].header['HNAME']))
                |(stellar_params['HNAME']==hdus[0].header['HNAME'])]) > 0:
                #take the mean of the logg estimates for this star, add it to our array of logg 
                logg = stellar_params[(stellar_params['HNAME'].str.contains(hdus[0].header['HNAME']))
                        |(stellar_params['HNAME']==hdus[0].header['HNAME'])].logg.mean()
                loggs = np.append(loggs, logg)
                #find the teff estimate for this star
                teff = stellar_params[(stellar_params['HNAME'].str.contains(hdus[0].header['HNAME']))
                |(stellar_params['HNAME']==hdus[0].header['HNAME'])].Teff
                #some stars have '<NA>' values for Teff. If the Teff is '<NA>', we set add nan to
                #the array of Teff. if it is a real value, we add the average of the Teff estimates
                #to the aray of Teff
                #note that I should improve this, should at least check if there is a non-<NA> estimate from a second observation
                if teff.astype(str).str.contains('<NA>').any(): 
                        print(teff)
                        teffs = np.append(teffs,np.NaN)
                else:
                        teffs = np.append(teffs,teff.mean())
                #take the mean of the feh estimates for this star, add it to our array of feh 
                feh = stellar_params[(stellar_params['HNAME'].str.contains(hdus[0].header['HNAME']))
                        |(stellar_params['HNAME']==hdus[0].header['HNAME'])]['__Fe_H_'].mean()
                fehs = np.append(fehs, feh)
                #compute the synthetic g-i and nb - g color, add each color to the respective array of
                # synthetic colors
                g_i_color = np.append(g_i_color, synthcolor(hdus[1], filt_g, filt_i, dered=True))
                nb_g_color = np.append(nb_g_color, synthcolor(hdus[1], filt_nb515, filt_g, dered=True))
                #set values flag to zero 
                values = np.append(values, 0)
                #check if there are comments for this star. if so, flag it
                if len(comment_flags[comment_flags['Star'].str.contains(hdus[0].header['HNAME'])])>0:
                        flag = np.append(flag, 1)
                else:
                        flag = np.append(flag, 0)
        #if there is no entry for this star in the dr2 table of stellar parameters, compute the color,
        #add NaN to the feh, teff, logg arrays. Set values flag to 1
        else:
                g_i_color = np.append(g_i_color, synthcolor(hdus[1], filt_g, filt_i, dered=True))
                nb_g_color = np.append(nb_g_color, synthcolor(hdus[1], filt_nb515, filt_g, dered=True))
                fehs = np.append(fehs, np.NaN)
                teffs = np.append(teffs, np.NaN)
                loggs = np.append(loggs, np.NaN)
                values = np.append(values, 1)
                if len(comment_flags[comment_flags['Star'].str.contains(hdus[0].header['HNAME'])])>0:
                        flag = np.append(flag, 1)
                else:
                        flag = np.append(flag, 0)
#%%
#note that some spectra have divide by zero errors - these stars will have one or both
#colors equal to inf 
# %%
df = pd.DataFrame()
df['gi'] = g_i_color
df['nbg'] = nb_g_color
df['logg'] = loggs
df['feh'] = fehs
df['teff'] = teffs
df['name'] = name_list
df['xshoot_name'] = xshoot_name
df['main_type'] = sp_type
df['secondary_type'] = sec_sp_type
df['flag'] = flag
df['var_flag'] = var_flag
df['mira_lpv_flag'] = mira_lpv_flag
df['fits_flag'] = values
df.to_csv('xshooter_synthcolor.csv', index=False)
#%%
df[(df.nbg>5)] #looking at the stars with divide by zero errors-
#%%
print(len(df[(df.var_flag>0)&(df.logg<1)]), 'log(g) < 1 giants flagged as variable ',
        len(df[(df.var_flag<1)&(df.logg<1)]), 'log(g) < 1 non-variable')
#counting number of red giants that are miras, lpvs 
print(len(df[(df.logg<4)&(df.gi>2)]), ' total giants with g - i > 2')
print(len(df[(df.logg<4)&(df.gi>2)&(df.var_flag>0)]), ' of these very red giants are variable')
print(len(df[(df.logg<4)&(df.gi>2)&(df.mira_lpv_flag>0)]), ' of these very red giants are mira or LPV')
#%%
len(df[(df.var_flag>0)]), len(df[df.mira_lpv_flag>0])
#%%
#from https://ui.adsabs.harvard.edu/abs/2002AJ....124.1706V/abstract
#we have an observational estimate of the radii of a few Miras at different points in their periods
#which we can use to estimate how much log(g) changes, as log(g) \propto log(1/r^2)
#here we have theoretical predictions for R changes https://ui.adsabs.harvard.edu/abs/2011MNRAS.418..114I/abstract
#this paper also gives log(g) vs phase https://articles.adsabs.harvard.edu//full/1998IAPPP..73...59P/0000072.000.html
#but isn't cited at all / seems to exist as an island, hard to gauge accuracy since they
#fit models to estimate log(g)
# ex one star has changes of ~260 to ~330 Rsun, another star from ~440 to ~410 rsun,
#model predictions from ~0.85 'parent radii' to ~1.4 'parent radii'
#and so on-
#over all, find that miras/LPVs should have log(g) change of like less than 1 but can be like
#delta log(g) of ~.3 or ~.4
#%%
# %%
#reading in HSC UMi data to plot as background
combined = pd.read_csv('/home/cfilion1/Galaxia/ursaminor_hsc_cleaned.csv')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24,10))
ax1.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)
#plotting synthetic colors for the x-shooter stars colored by logg, first plot non-variables
c1 = ax1.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['logg'], vmin=0, vmax=5., cmap='rainbow')
#then plot variables that are not lpv/mira 
g = ax1.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['logg'], 
                vmin=0, vmax=5., cmap='rainbow', 
                marker='s', label = 'Variable')
#finally, plot lpvs/miras
q = ax1.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['logg'], 
                vmin=0, vmax=5., cmap='rainbow', 
                marker='*', s=100, label='LPV or Mira')
g.set_facecolor('none')
g.set_edgecolor('k')
q.set_facecolor('none')
q.set_edgecolor('k')
ax1.legend()
ax1.set_xlim(-0.75, 4)
ax1.set_ylim(0.8,-.8)
ax1.set_xlabel('$(g-i)_0$')
ax1.set_ylabel('$(NB515 - g)_0$')

ax2.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)
#same as above, but color-coding by feh
c2 = ax2.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag<1)]['feh'], vmin=-3.5, vmax=1, cmap='jet')

gg = ax2.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag<1)]['feh'], 
                vmin=-3.5, vmax=1, cmap='jet', marker='s', label='Variable')
qq = ax2.scatter(df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['gi'], 
                df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['nbg'], 
                c=df[(df.flag<1)&(df.fits_flag<1)&(df.var_flag>0)&(df.mira_lpv_flag>0)]['feh'], 
                vmin=-3.5, vmax=1, cmap='jet', marker='*', s=100, label='Mira or LPV')
gg.set_facecolor('none')
gg.set_edgecolor('k')
qq.set_facecolor('none')
qq.set_edgecolor('k')
ax2.legend()
ax2.set_xlim(-0.75, 4)
ax2.set_ylim(0.8,-.8)
ax2.set_xlabel('$(g-i)_0$')
ax2.set_ylabel('$(NB515 - g)_0$')

ax1.set_title('Colored by log(g)')
ax2.set_title('Colored by [Fe/H]')
fig.colorbar(c1, ax=ax1, orientation = 'vertical', label='log(g)')
fig.colorbar(c2, ax=ax2, orientation = 'vertical', label='[Fe/H]')
plt.subplots_adjust(wspace=.2)
plt.savefig('/home/cfilion1/xshooter_cc_diagram.png',bbox_inches = 'tight',
    pad_inches = 0.1) 
#%%

#%%
#plotting teff vs color, teff vs logg
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24,10))
c1 = ax1.scatter(np.log10(df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['teff'].astype(float)), 
        df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['gi'],
        c=df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['logg'],
        cmap='copper')
cc = ax1.scatter(np.log10(df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['teff'].astype(float)), 
        df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['gi'],
        c=df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['logg'],
        cmap='copper', marker='s')
cc.set_facecolor('none')
fig.colorbar(c1, ax=ax1, orientation = 'vertical', label='log(g)')
ax1.set_xlabel('log(Teff)')
ax1.set_xlim(np.log10(21000), np.log10(2100))
ax1.set_ylabel('g-i')
ax1.set_ylim(-1.0,4.5)

c2 = ax2.scatter(np.log10(df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['teff'].astype(float)), 
        df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['logg'],
        c=df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag<1)]['nbg'],
        cmap='rainbow')
aa = ax2.scatter(np.log10(df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['teff'].astype(float)), 
        df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['logg'],
        c=df[(df.flag<1)&(df.fits_flag<1)&(df.nbg<99)&(df.var_flag>0)]['nbg'],
        cmap='rainbow', marker='s')
aa.set_facecolor('none')
fig.colorbar(c2, ax=ax2, orientation = 'vertical', label='NB515-g')
ax2.set_xlabel('log(Teff)')
ax2.set_xlim(np.log10(21000), np.log10(2100))
ax2.set_ylabel('log(g)')
ax2.set_ylim(5.5,-.2)
plt.subplots_adjust(wspace=.1)
plt.savefig('/home/cfilion1/giteff_loggteff.png')
#%%
#%%
#looking at stars with feh > 0
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24,10))
ax1.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)
c1 = ax1.scatter(df[(df.feh>0)]['gi'], df[(df.feh>0)]['nbg'], 
                c=df[(df.feh>0)]['logg'], vmin=0, vmax=5., cmap='rainbow')
ax1.set_xlim(-0.75, 4)
ax1.set_ylim(0.8,-.8)
ax1.set_xlabel('$(g-i)_0$')
ax1.set_ylabel('$(NB515 - g)_0$')

ax2.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)

c2 = ax2.scatter(df[(df.feh>0)]['gi'], df[(df.feh>0)]['nbg'], 
                c=df[(df.feh>0)]['feh'], vmin=0, vmax=1, cmap='rainbow')

ax2.set_xlim(-0.75, 4)
ax2.set_ylim(0.8,-.8)
ax2.set_xlabel('$(g-i)_0$')
ax2.set_ylabel('$(NB515 - g)_0$')


fig.colorbar(c1, ax=ax1, orientation = 'vertical', label='log(g)')
fig.colorbar(c2, ax=ax2, orientation = 'vertical', label='[Fe/H]')
plt.subplots_adjust(wspace=.2)
plt.savefig('/home/cfilion1/metalrich_xshooter_cc_diagram.png')

#looking at stars with feh < -2
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24,10))
fig.suptitle('Stars with [Fe/H] < -2')
ax1.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)
c1 = ax1.scatter(df[(df.feh<-2)]['gi'], df[(df.feh<-2)]['nbg'], 
                c=df[(df.feh<-2)]['logg'], vmin=0, vmax=5., cmap='rainbow')
ax1.set_xlim(-0.75, 4)
ax1.set_ylim(0.8,-.8)
ax1.set_xlabel('$(g-i)_0$')
ax1.set_ylabel('$(NB515 - g)_0$')

ax2.scatter(combined.gpsf-combined.ipsf,
        combined.npsf-combined.gpsf, c='grey',s=.1,rasterized=True)

c2 = ax2.scatter(df[(df.feh<-2)]['gi'], df[(df.feh<-2)]['nbg'], 
                c=df[(df.feh<-2)]['feh'], vmin=-3.5, vmax=-2, cmap='rainbow')

ax2.set_xlim(-0.75, 4)
ax2.set_ylim(0.8,-.8)
ax2.set_xlabel('$(g-i)_0$')
ax2.set_ylabel('$(NB515 - g)_0$')


fig.colorbar(c1, ax=ax1, orientation = 'vertical', label='log(g)')
fig.colorbar(c2, ax=ax2, orientation = 'vertical', label='[Fe/H]')
plt.subplots_adjust(wspace=.2)
plt.savefig('/home/cfilion1/metalpoor_xshooter_cc_diagram.png')
#%%
#%%
# %%
#OGLEII DIA BUL-SC22 1319 is mira variable, no nb mag
#SHV 0534578-702532 is mira variable
#SHV 0517337-725738 is carbon star, candidate lpv
#SHV 0518161-683543 is carbon star, also lpv/maybe mira
#OGLEII DIA BUL-SC03 3941 is mira variable

# %%

#%%