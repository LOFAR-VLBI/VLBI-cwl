from scipy.stats import skewnorm

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import leastsq
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import os
import bdsf 

deg2rad=np.pi/180.0
rad2deg=180.0/np.pi

def time_smearing2(delta_T, delta_Theta, resolution, verbose=False):
    """
    Calculate the flux reduction due to time averaging (time smearing).
    Parameters:
        delta_T (float): Time averaging interval (seconds)
        delta_Theta (float): Angular distance from phase center (degrees)
        resolution (float): Synthesized beam resolution (degrees)
        verbose (bool): If True, print detailed info
    Returns:
        float: Fraction of flux remaining after time smearing
    """
    Reduction = 1 - 1.22E-9 * (delta_Theta / resolution) ** 2.0 * delta_T ** 2.0
    if verbose:
        print(f"[time_smearing2] At radius {delta_Theta} deg and resolution {resolution} deg, the source will have {Reduction:.4f} of its flux if data smoothed to {delta_T} sec.")
    return Reduction

def bandwidth_smearing2(resolution, freq, delta_Theta, delta_freq, verbose=False):
    """
    Calculate the flux reduction due to frequency averaging (bandwidth smearing).
    Parameters:
        resolution (float): Synthesized beam resolution (degrees)
        freq (float): Central frequency (Hz)
        delta_Theta (float): Angular distance from phase center (degrees)
        delta_freq (float): Frequency averaging interval (Hz)
        verbose (bool): If True, print detailed info
    Returns:
        float: Fraction of flux remaining after bandwidth smearing
    """
    beta = (delta_freq / freq) * (delta_Theta / resolution)
    gamma = 2 * (np.log(2) ** 0.5)
    Reduction = ((np.pi ** 0.5) / (gamma * beta)) * (scipy.special.erf(beta * gamma / 2.0))
    if verbose:
        print(f"[bandwidth_smearing2] At radius {delta_Theta} deg, resolution {resolution} deg, frequency {freq} Hz: flux fraction = {Reduction:.4f} if data smoothed in freq to {delta_freq} Hz.")
    return Reduction


def fit_and_report_skewed_gaussian(data, label):
    """
    Fit a skewed Gaussian (skewnorm) to the log of the data, return median and std in linear space.
    """
    data = np.array(data)
    data = data[data > 0]
    logdata = np.log(data)
    # Fit skewed normal to log(data)
    params = skewnorm.fit(logdata)
    a, loc, scale = params
    # Median and std in log space
    median_log = skewnorm.median(a, loc, scale)
    std_log = skewnorm.std(a, loc, scale)
    # Convert back to linear space
    median = np.exp(median_log)
    std = np.exp(std_log)
    print(f"{label}: Skewed Gaussian fit median = {median:.3f}, std = {std:.3f} (log-space median = {median_log:.3f}, std = {std_log:.3f})")
    return median, std, params


def run_bdsf_fast(infile, cthresh_isl, cthresh_pix, catprefix='mosaic'):
    """
    Run PyBDSF source finding on a radio image and save output catalogues and images.

    This function processes a given FITS image using PyBDSF with specified island and pixel thresholds. It generates and saves:
      - Source catalogue in FITS and DS9 region formats
      - RMS map
      - Residual image (Gaussian residuals)
      - Island mask image

    Parameters
    ----------
    infile : str
        Path to the input FITS image to process.
    cthresh_isl : float
        Island threshold for source detection (in sigma).
    cthresh_pix : float
        Pixel threshold for source detection (in sigma).
    catprefix : str, optional
        Prefix for output catalogue and image filenames (default: 'mosaic').

    Outputs
    -------
    Writes the following files to disk (with names based on catprefix):
      - .cat.fits : Source catalogue (FITS format)
      - .cat.reg  : Source catalogue (DS9 region format)
      - .rms.fits : RMS map
      - .resid.fits : Residual image
      - .pybdsfmask.fits : Island mask image
    """
    restfrq=144000000.0
    print('Running PyBDSF and saving catalogue and rms image')
    img = bdsf.process_image(infile, thresh_isl=cthresh_isl, thresh_pix=cthresh_pix, rms_box=(150,15), rms_map=True, mean_map='zero', ini_method='intensity', adaptive_rms_box=True, adaptive_thresh=150, rms_box_bright=(60,15), group_by_isl=False, group_tol=10.0, output_opts=True, output_all=True, atrous_do=False, atrous_jmax=4, flagging_opts=True, flag_maxsize_fwhm=0.5,advanced_opts=True, blank_limit=None, frequency=restfrq)
    img.write_catalog(outfile=catprefix +'.cat.fits',catalog_type='srl',format='fits',correct_proj='True')
    img.export_image(outfile=catprefix +'.rms.fits',img_type='rms',img_format='fits',clobber=True)
    img.export_image(outfile=catprefix +'.resid.fits',img_type='gaus_resid',img_format='fits',clobber=True)
    img.export_image(outfile=catprefix +'.pybdsfmask.fits',img_type='island_mask',img_format='fits',clobber=True)
    img.write_catalog(outfile=catprefix +'.cat.reg',catalog_type='srl',format='ds9',correct_proj='True')

    return f'{catprefix}.cat.fits', f'{catprefix}.rms.fits'


def radial_model_quadratic(t, a, b, c):
    return a + b * t + c * t**2

def sepn(r1,d1,r2,d2):
    """
    Calculate the separation between 2 sources, RA and Dec must be
    given in radians. Returns the separation in radians
    """
    # NB slalib sla_dsep does this
    # www.starlink.rl.ac.uk/star/docs/sun67.htx/node72.html
    cos_sepn=np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    sepn = np.arccos(cos_sepn)
    # Catch when r1==r2 and d1==d2 and convert to 0
    sepn = np.nan_to_num(sepn)
    return sepn

def smearing_corrected_sourcesizes(highrescat,highresimagerms,fitsmearing,verbose=False):
    
    # Find some things from image
    with fits.open(highresimagerms) as f:
        racent = f[0].header['CRVAL1']
        deccent = f[0].header['CRVAL2']
        beammaj = f[0].header['BMAJ']
        beammin = f[0].header['BMIN']
        average_beam = np.sqrt(beammaj * beammin)
    """
    Apply smearing corrections to source sizes and fluxes in a high-resolution radio source catalogue.

    This function reads a high-resolution source catalogue and its associated RMS image, computes the radial distance of each source from the image center, and applies corrections for time and bandwidth smearing. Optionally, it fits a radial correction model to the data and applies it. The corrected catalogue can be saved, and diagnostic plots are generated to visualize the corrections and their effects on flux ratios.

    Returns the peak and standard deviation of a skewed Gaussian fitted to the source Total/Peak flux.

    Parameters
    ----------
    highrescat : str
        Path to the high-resolution source catalogue FITS file.
    highresimagerms : str
        Path to the high-resolution image RMS FITS file.
    fitsmearing : bool
        If True, fit a radial correction model to the data; otherwise, use theoretical smearing corrections.
    verbose : bool, optional
        If True, print detailed progress and save diagnostic plots. Default is False.

    Returns
    -------
    median_tc : float
        Median of the total-to-peak flux ratio for the corrected catalogue.
    std_tc : float
        Standard deviation of the total-to-peak flux ratio for the corrected catalogue.
    """
    catdata = Table.read(highrescat)


    if verbose:
        print(f'Image centre: RA={racent}, DEC={deccent}')
        print(f'Beam major axis: {beammaj}, Beam minor axis: {beammin}, Average beam: {average_beam}')
        print('Adjusting Peak flux for time- and bandwidth-smearing (assuming 4s and 4ch/sb)')


    # Plot radial correction
    separation = sepn(catdata['RA']*deg2rad,catdata['DEC']*deg2rad,racent*deg2rad,deccent*deg2rad)*rad2deg

    ideal_smearing = np.zeros_like(separation)
    for i in range(len(separation)):
        t_smear = time_smearing2(2.0,separation[i], average_beam)
        b_smear = bandwidth_smearing2(average_beam, 144E6, separation[i], 195.3125E3/16.0)
        ideal_smearing[i] = t_smear*b_smear
        #if verbose:
        #    print(f'Source {i}: Time smearing = {t_smear}, Bandwidth smearing = {b_smear}, Ideal smearing = {ideal_smearing[i]}')


    if fitsmearing:
        plt.figure(figsize=(8,6))

        radbins = np.arange(0.0,1.5,0.2)
        xvals = []
        yvals = []
        raddeletes = []

        totsmears = []
        for i in range(0,len(radbins)-1):
            binmin = radbins[i]
            binmax = radbins[i+1]
            sourcemin = np.where(separation > binmin)
            sourcemax = np.where(separation < binmax)
            meetscriteria = np.intersect1d(sourcemin,sourcemax)
            print('Bin: %s deg - %s deg contains %s sources'%(binmin,binmax,len(meetscriteria)))
            if len(meetscriteria) == 0:
                raddeletes.append(i)
                continue
            medval = np.median(catdata['Total_flux'][meetscriteria]/catdata['Peak_flux'][meetscriteria])
            medval = np.percentile(catdata['Total_flux'][meetscriteria]/catdata['Peak_flux'][meetscriteria],10)
            plt.plot((binmin+binmax)/2.0,medval,'g+',markersize=10)
            plt.plot(separation[meetscriteria],catdata['Total_flux'][meetscriteria]/catdata['Peak_flux'][meetscriteria],'b.',alpha=0.1)
            xvals.append((binmin+binmax)/2.0)
            yvals.append(medval)

            bandsmear = bandwidth_smearing2(average_beam, 144E6, (binmin+binmax)/2.0, 195.3125E3/4.0)
            timesmear = time_smearing2(4.0, (binmin+binmax)/2.0, average_beam)
            totsmears.append(bandsmear*timesmear)

        radbins = np.delete(radbins,raddeletes)
        # Fit and apply radial correction
        # Try multiple models for the radial fit

        models = [
            (radial_model_quadratic, [1, 0.1, 0.01], 'quadratic')
        ]

        for model, p0, label in models:
            try:
                popt, _ = scipy.optimize.curve_fit(model, xvals, yvals, p0=p0, maxfev=10000)
                plt.plot(xvals, model(np.array(xvals), *popt), label=label)
            except Exception as e:
                print(f"Fit failed for {label}: {e}")
        plt.scatter(xvals, yvals, color='k', label='data')
        print(xvals,yvals)
        plt.ylim(ymin=0.8, ymax=2.0)
        plt.xlabel('Distance from pointing centre (deg)')
        plt.ylabel('Total/Peak flux')
        plt.plot(xvals, 1.0/np.array(totsmears), 'r--', linewidth=2, label='Ideal smearing')
        plt.legend()
        plt.tight_layout()
        plt.savefig('radial-correction.png')
        plt.close()
        plt.cla()

    for i in range(0,len(catdata)):
        #corval = model(separation[i],xfit[0],xfit[1])/np.min(model(np.array(xvals),xfit[0],xfit[1])) # To normalise to the value at min X
        if fitsmearing:
            corval = radial_model_quadratic(separation[i],*popt)
        else:
            timesmear = time_smearing2(4.0, separation[i], average_beam)
            freqsmear = bandwidth_smearing2(average_beam, 144E6, separation[i], 195.3125E3/4.0)
            corval = 1/(timesmear*freqsmear)
        newSI = catdata[i]['Peak_flux']*corval
        catdata[i]['Peak_flux'] = newSI
        newSI = catdata[i]['E_Peak_flux']*corval
        catdata[i]['E_Peak_flux'] = newSI

    if verbose:
        radcorcat = highrescat.replace('.fits', '_radcat.fits')
        if not os.path.exists(radcorcat):
            catdata.write(radcorcat, overwrite=False)
        else:
            print(f'Not outputing {radcorcat} as it already exists')
        radbins = np.arange(0.0,1.5,0.2)

        # Check corrections applied ok
        catdata_cor = Table.read(radcorcat)
        separation = sepn(catdata_cor['RA'] * deg2rad, catdata_cor['DEC'] * deg2rad, racent * deg2rad, deccent * deg2rad) * rad2deg
        xvals = []
        yvals = []
        for i in range(0, len(radbins) - 1):
            binmin = radbins[i]
            binmax = radbins[i + 1]
            print('Bin: %s - %s' % (binmin, binmax))
            sourcemin = np.where(separation > binmin)
            sourcemax = np.where(separation < binmax)
            meetscriteria = np.intersect1d(sourcemin, sourcemax)
            if len(meetscriteria) == 0:
                break
            medval = np.median(catdata_cor['Total_flux'][meetscriteria] / catdata_cor['Peak_flux'][meetscriteria])
            plt.plot((binmin + binmax) / 2.0, medval, 'g+', markersize=10)
            plt.plot(separation[meetscriteria], catdata_cor['Total_flux'][meetscriteria] / catdata_cor['Peak_flux'][meetscriteria], 'b.', alpha=0.1)
            xvals.append((binmin + binmax) / 2.0)
            yvals.append(medval)

        models = [
            (radial_model_quadratic, [1, 0.1, 0.01], 'quadratic')
        ]

        for model, p0, label in models:
            try:
                popt, _ = scipy.optimize.curve_fit(model, xvals, yvals, p0=p0, maxfev=10000)
                plt.plot(xvals, model(np.array(xvals), *popt), label=label)
            except Exception as e:
                print(f"Fit failed for {label}: {e}")

        plt.ylim(ymin=0.8, ymax=2.0)
        plt.xlabel('Distance from pointing centre (deg)')
        plt.ylabel('Total/Peak flux')
        plt.savefig('radial-correction-corrected.png')
        plt.close()
        plt.cla()

    

    # Plot histograms of flux ratios for original and corrected catalogues
    # Read original and corrected catalogues
    catdata_cor = Table.read(radcorcat)
    # Compute ratios
    total_ratio_orig = catdata['Total_flux'] / catdata['Peak_flux']
    total_ratio_cor = catdata_cor['Total_flux'] / catdata_cor['Peak_flux']
    # Plot
    plt.figure(figsize=(8,6))
    # Filter out non-positive values for log bins
    all_ratios = np.concatenate([
        total_ratio_orig[total_ratio_orig > 0],
        total_ratio_cor[total_ratio_cor > 0]
    ])
    min_bin = all_ratios.min()
    max_bin = all_ratios.max()
    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
    # Fit and plot for each histogram
    legend_entries = []
    # Uncomment if you want to show original as well
    #center_to, std_to, params_to = fit_and_report_skewed_gaussian(total_ratio_orig, 'Total/Err (orig)')
    median_tc, std_tc, params_tc = fit_and_report_skewed_gaussian(total_ratio_cor, 'Total/Err (corr)')

    # Plot histograms
    #plt.hist(total_ratio_orig[total_ratio_orig > 0], bins=bins, color='red', alpha=0.5, label=f'Total/Err (orig)')

    n_tc, bins_tc, _ = plt.hist(total_ratio_cor[total_ratio_cor > 0], bins=bins, color='red', alpha=0.5, label=None, histtype='step', linestyle='dashed', linewidth=2, hatch='//')


    # Mark the peak (mode) of the fitted skewed Gaussian with a vertical line
    a, loc, scale = params_tc
    # The mode of a skewnorm is at loc + scale * delta, where delta = skewnorm.mean(a, 0, 1) for standardized
    # But for skewnorm, the mode is not directly available, so we numerically find the maximum of the PDF
    from scipy.optimize import minimize_scalar
    def neg_pdf(logx):
        return -skewnorm.pdf(logx, a, loc, scale)
    res = minimize_scalar(neg_pdf, bounds=(np.log(min_bin), np.log(max_bin)), method='bounded')
    mode_log = res.x
    mode_tc = np.exp(mode_log)
    plt.axvline(mode_tc, color='red', linestyle='dashed', linewidth=2, label=f'Peak (mode): {mode_tc:.2f}')

    # Plot the fitted skewed Gaussian PDFs on top of the histograms, normalized to counts in each bin
    xfit = np.logspace(np.log10(min_bin), np.log10(max_bin), 200)
    # For each bin, compute the expected count in that bin from the PDF
    pdf_tc = skewnorm.pdf(np.log(xfit), *params_tc)
    # Normalize PDF so that the area under the curve matches the total number of counts
    area_tc = np.sum(n_tc * np.diff(bins_tc))
    pdf_tc = pdf_tc * area_tc / np.trapz(pdf_tc, xfit)
    plt.plot(xfit, pdf_tc, color='red', linestyle='solid', linewidth=2, label='Total/Peak (corr) fit')

    legend_entries.append(f'Total/Peak (corr): median={median_tc:.2f}, std={std_tc:.2f}')

    plt.xscale('log')
    plt.xlabel('Total Flux / Peak Flux (log scale)')
    plt.ylabel('Number of sources')
    plt.legend(legend_entries + ['Total/Peak (corr) fit', 'Peak/Peak (corr) fit'])
    plt.title('Total/Peak Histograms')
    plt.tight_layout()
    plt.savefig('Flux_TotaltoPeak_Histograms.png')
    plt.close()
    return mode_tc, std_tc

def compare_with_external_catalog(highrescat, highresimagerms, externalcat, externalscaling, verbose=False):
    """
    Compare an external source catalogue to a high-resolution catalogue within a given image region.

    This function filters the external catalogue to sources within 3 degrees of the image pointing centre and inside the image footprint.
    It then matches these sources to the high-resolution catalogue within a 3 arcsec radius, and compares their fluxes.
    Optionally, it saves the filtered catalogue and plots flux comparisons if verbose is True.

    Parameters
    ----------
    highrescat : str
        Path to the high-resolution catalogue FITS file (e.g., 'mosaic.cat.fits').
    highresimagerms : str
        Path to the high-resolution image RMS FITS file (e.g., 'mosaic.rms.fits').
    externalcat : str
        Path to the external catalogue FITS file (e.g., LoTSS or other pybdsf catalogue).
    externalscaling : float
        Scaling factor to apply to the external catalogue fluxes before comparison (LoTSS catalogues are sometimes in mJy)
    verbose : bool, optional
        If True, print progress and plot flux comparisons. Default is False.

    Returns
    -------
    fraction_matched : float
        Fraction of external catalogue sources (with sufficient S/N) matched within 3 arcsec to the high-res catalogue.
    median_tot_fluxratio : float
        Median ratio of total fluxes for matched sources (filtered_cat / highrescat_open).
    """



    with fits.open(highresimagerms) as hdul:
        wcs = WCS(hdul[0].header)
        data = hdul[0].data
        header = hdul[0].header
        # If data has more than 2 dimensions, take the first 2D slice
        if data is not None and data.ndim > 2:
            image_2d = data[0, 0, :, :]
        else:
            image_2d = data
        ny, nx = image_2d.shape
        centre_ra = header['CRVAL1']
        centre_dec = header['CRVAL2']


    centre_coord = SkyCoord(ra=centre_ra, dec=centre_dec, unit='deg', frame='icrs')

    # Get the four corners of the image in pixel coordinates for a 4D WCS (RA, DEC, FREQ, STOKES)
    # For each corner, build a 4D pixel coordinate: (x, y, 0, 0)
    corner_pixels = [
        (0, 0, 0, 0),         # lower-left
        (nx-1, 0, 0, 0),      # lower-right
        (nx-1, ny-1, 0, 0),   # upper-right
        (0, ny-1, 0, 0)       # upper-left
    ]
    corner_pixels = list(zip(*corner_pixels))  # unzip for pixel_to_world_values
    sky_corners = wcs.pixel_to_world_values(*corner_pixels)
    # Unwrap RA corners to [0, 360] to match catalogue
    ra_corners, dec_corners = sky_corners[0], sky_corners[1]
    ra_corners = ra_corners % 360

    if verbose:
        print(f'Image centre: RA={centre_ra}, DEC={centre_dec}')
        print(f'Image corners (RA, Dec):')
        for ra, dec in zip(ra_corners, dec_corners):
            print(f'  RA={ra}, DEC={dec}')


    # Open the catalogue and extract RA/Dec
    cat = Table.read(externalcat)
    ra = cat['RA']  # or the correct column name for RA
    dec = cat['DEC']  # or the correct column name for Dec

    # Create SkyCoord objects for catalogue
    cat_coords = SkyCoord(ra=ra, dec=dec, frame='icrs')



    # Filter catalogue to within 3 degrees of pointing centre
    sep = cat_coords.separation(centre_coord)
    mask = sep < 3 * u.deg
    cat_coords = cat_coords[mask]
    image_poly = Polygon(zip(ra_corners, dec_corners))
    # Find sources within the image region and save to a new catalogue
    in_region_idx = []
    if verbose:
        iterator = enumerate(tqdm(cat_coords, desc='Cutting external catalogue to within 3 degrees of pointing'))
    else:
        iterator = enumerate(cat_coords)
    for i, coord in iterator:
        point = Point(coord.ra.deg, coord.dec.deg)
        if image_poly.contains(point):
            in_region_idx.append(i)
    filtered_cat = cat[mask][in_region_idx]
    if verbose:
        # Save the filtered catalogue
        output_filtered = 'filtered_external_catalogue.fits'
        if os.path.exists(output_filtered):
            print(f'Not outputing {output_filtered} as it already exists')
        else:
            filtered_cat.write(output_filtered, overwrite=False)


    # For each source in filtered_cat, check if it is within 3" of a source in mosaic.cat.fits. 
    highrescat_open = Table.read(highrescat)
    filtered_coords = SkyCoord(ra=filtered_cat['RA'], dec=filtered_cat['DEC'], unit='deg', frame='icrs')
    mosaic_coords = SkyCoord(ra=highrescat_open['RA'], dec=highrescat_open['DEC'], unit='deg', frame='icrs')
    idx, sep2d, _ = filtered_coords.match_to_catalog_sky(mosaic_coords)
    radius = 3 * u.arcsec
    not_matched = sep2d >= radius
    arcsec2deg = 1 / 3600.0
    # Open the mosaic.rms.fits map
    notfound = 0
    found = 0
    tot_ratios = []
    for i in range(len(filtered_cat)):
        # Get pixel coordinates for this source
        ra = filtered_cat['RA'][i]
        dec = filtered_cat['DEC'][i]
        # Use reference values for STOKES and FREQ axes
        stokes = wcs.wcs.crval[2]
        freq = wcs.wcs.crval[3]
        xpix, ypix, _, _ = wcs.world_to_pixel_values(ra, dec, stokes, freq)
        xpix = int(round(float(xpix)))
        ypix = int(round(float(ypix)))

        sep = sepn(ra*deg2rad,dec*deg2rad,centre_ra*deg2rad,centre_dec*deg2rad)*rad2deg

        t_smear = time_smearing2(1.0,sep, 1.0*arcsec2deg)
        b_smear = bandwidth_smearing2(1.0*arcsec2deg, 144E6, sep, 195.3125E3/16.0)
        totsmears = t_smear*b_smear

        # Check bounds
        if (0 <= xpix < data.shape[-1]) and (0 <= ypix < data.shape[-2]):
            rms_val = data[..., ypix, xpix] if data.ndim > 2 else data[ypix, xpix]
        else:
            rms_val = float('nan')

        rms_val = rms_val/totsmears

        if filtered_cat['Peak_flux'][i]*externalscaling < 20.0*rms_val:
            continue

        if not_matched[i]:
            notfound +=1
            if verbose:
                print(f"No match for source {i}: Total_flux={filtered_cat['Total_flux'][i]}, Peak_flux={filtered_cat['Peak_flux'][i]}, RMS={rms_val}")
        else:
            found+=1
            tot_ratios.append(filtered_cat['Total_flux'][i]*externalscaling/highrescat_open['Total_flux'][idx[i]])
            if verbose:
                print(f"Match found for source {i}: Total_flux={filtered_cat['Total_flux'][i]}, Peak_flux={filtered_cat['Peak_flux'][i]}, RMS={rms_val}")


    # Calculate and print the fraction of sources matched within 3 arcsec
    fraction_matched = found / (found + notfound) if (found + notfound) > 0 else 0
    median_tot_fluxratio = np.median(tot_ratios) if tot_ratios else 0
    if verbose:
        print(f"Number of external catalogue sources within the image region: {len(in_region_idx)} but only {found + notfound} with Peak_flux > 20 x highres smeared RMS")
        print(f"Fraction of {found + notfound} sources matched within 3 arcsec: {fraction_matched:.4f} ({found}/{found + notfound})")

        print(f"Median total flux ratio for matched sources: {median_tot_fluxratio:.4f}")

        # Plotting Total_flux and Peak_flux comparisons for matched sources
        matched_total_flux = [filtered_cat['Total_flux'][i] for i in range(len(filtered_cat)) if not_matched[i] == False]
        matched_highres_total_flux = [highrescat_open['Total_flux'][idx[i]] for i in range(len(filtered_cat)) if not_matched[i] == False]
        matched_peak_flux = [filtered_cat['Peak_flux'][i] for i in range(len(filtered_cat)) if not_matched[i] == False]
        matched_highres_peak_flux = [highrescat_open['Peak_flux'][idx[i]] for i in range(len(filtered_cat)) if not_matched[i] == False and 'Peak_flux' in highrescat_open.colnames]

        plt.figure(figsize=(8,6))
        # Total flux scatter
        plt.scatter(matched_highres_total_flux, np.array(matched_total_flux)*externalscaling, color='red', label='Total Flux', alpha=0.6)
        # Median line for total flux
        if matched_total_flux:
            median_total = np.median(np.array(matched_total_flux)*externalscaling/(np.array(matched_highres_total_flux)))
            xvals = np.logspace(np.log10(min(matched_highres_total_flux)), np.log10(max(matched_highres_total_flux)), 100)
            plt.plot(xvals, median_total * xvals, color='red', linestyle=':', label=f'Median Total Flux Ratio (external/highres): {median_total:.4f}')

        # Peak flux scatter (if available)
        if matched_highres_peak_flux:
            plt.scatter(matched_highres_peak_flux, np.array(matched_peak_flux)*externalscaling, color='green', label='Peak Flux', alpha=0.6)
            median_peak = np.median(np.array(matched_peak_flux)*externalscaling/(np.array(matched_highres_peak_flux)))
            xvals_peak = np.logspace(np.log10(min(matched_highres_peak_flux)), np.log10(max(matched_highres_peak_flux)), 100)
            plt.plot(xvals_peak, median_peak * xvals_peak, color='green', linestyle=':', label=f'Median Peak Flux Ratio (external/highres): {median_peak:.4f}')
        plt.semilogx()
        plt.semilogy()
        plt.xlabel('Highres Catalogue Flux')
        plt.ylabel('Filtered Catalogue Flux')
        plt.title('Matched Source Flux Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('matched_source_flux_comparison.png')

    return(fraction_matched, median_tot_fluxratio)


# Do you want to make a catalogue

# Example usage


externalcat = 'LoTSS_DR3_v0.3.srl.fits'
highresimage = 'ELAIS_L798074_1.2image-MFS-image-pb.fits'

#highrescat, highresrmsimage = run_bdsf_fast(highresimage, 16.0, 20.0)

highrescat = 'mosaic.cat.fits'
highresrmsimage = 'mosaic.rms.fits'

#mode_fluxratio, std_fluxratio = smearing_corrected_sourcesizes(highrescat,highresrmsimage,True,verbose=True)

#fraction_matched, median_tot_fluxratio = compare_with_external_catalog(highrescat, highresrmsimage, externalcat, 1E-3, verbose=True)

