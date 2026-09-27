"""
Why is the r50 ratio 1.06 instead of 1.00?

The PSF-matching check convolves the source PSF with the generated kernel and
compares the half-light radius of the result with that of the target PSF. The
ratio must be 1.0 for a correct kernel (Aniano et al. 2011, Sect. 5). The
pipeline measures ~1.06, which sits outside the 0.95-1.05 criterion.

This script tests the leading hypothesis - that PyPHER's Wiener regularisation
parameter (r, default 1e-4) smooths the kernel and therefore widens the result
- and rules in or out two alternatives along the way.

Three blocks:

  1. SYNTHETIC CONTROL. Two Gaussians of known width. For Gaussians the exact
     kernel is analytic, so a correct method MUST return ratio = 1.000. If it
     does not, the problem is in the method and not in the real PSFs. This is
     what makes the rest interpretable.

  2. REGULARISATION SWEEP. Vary r over several orders of magnitude on both the
     synthetic pair and a real one. If the ratio moves with r, regularisation
     is the cause; if not, it is innocent and we look elsewhere. Negative power
     is tracked alongside, because lowering r buys fidelity at the cost of
     ringing - the trade-off is the result.

  3. RADIAL PROFILE. Overlay the profile of (source * kernel) against the
     target. The ratio is one number; the profile shows WHERE they diverge,
     which distinguishes discretisation from smoothing from truncation.

Usage:
    python test_pypher_regularisation.py
Edit the CONFIG block below to point at your own PSF files.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import fftconvolve

# ------------------------------------------------------------------ CONFIG --
# Convolution grid of the survey being tested (arcsec/px). For PHANGS-JWST
# this is the native MIRI scale; check the preflight output for your run.
GRID_ARCSEC = 0.111

# Regularisation values to sweep. PyPHER's default is 1e-4.
R_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# Real PSF pair. Prefer a pair whose required blur is SMALL: the kernel is then
# narrow and most sensitive to regularisation, so the effect shows up clearly.
# f1130w -> f2100w (FWHM 0.357" -> 0.643") is a better probe than
# f200w -> f2100w (0.062" -> 0.643"), where the kernel is wide and forgiving.
PSF_DIR = Path("~/Research/AsTrovello/Input/PHANGS-JWST/PSF_CLEAN").expanduser()
REAL_SOURCE = PSF_DIR / "PSF_MIRI_in_flight_opd_filter_F1130W.fits"
REAL_TARGET = PSF_DIR / "master_PSF_MIRI_in_flight_opd_filter_F2100W.fits"

# Synthetic pair, in arcsec FWHM. Chosen to mimic the real case.
SYN_FWHM_SOURCE = 0.357
SYN_FWHM_TARGET = 0.643
SYN_SIZE = 201          # odd, as PyPHER requires

# Outputs go under the repository, not the directory the script is run from.
REPO_ROOT = Path("~/Research/AsTrovello").expanduser()
OUT_DIR = REPO_ROOT / "Output" / "checks" / "pypher_regularisation"

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# --------------------------------------------------------------- utilities --
def gaussian_psf(n_pix, fwhm_arcsec, scale_arcsec):
    """Unit-sum 2D Gaussian on an n_pix square grid."""
    sigma_px = fwhm_arcsec * FWHM_TO_SIGMA / scale_arcsec
    c = n_pix // 2
    y, x = np.indices((n_pix, n_pix)) - c
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma_px ** 2))
    return (g / g.sum()).astype(np.float64)


def write_psf(path, data, scale_arcsec):
    """Write a PSF with the PIXSCALE keyword PyPHER reads."""
    hdr = fits.Header()
    hdr['PIXSCALE'] = (scale_arcsec, 'arcsec/px')
    fits.PrimaryHDU(data.astype(np.float32), header=hdr).writeto(
        path, overwrite=True)


def enclosed_radius(data, scale_arcsec, fraction=0.5):
    """Radius enclosing `fraction` of the flux, in arcsec.

    Sorts pixels by distance from the peak and interpolates where the
    cumulative flux crosses the requested fraction. Profile-shape independent,
    unlike a Gaussian FWHM fit.
    """
    peak = np.array(np.unravel_index(np.nanargmax(data), data.shape))
    y, x = np.indices(data.shape)
    r = np.hypot(y - peak[0], x - peak[1]).ravel()
    v = np.nan_to_num(data.ravel())
    order = np.argsort(r)
    cum = np.cumsum(v[order])
    if cum[-1] <= 0:
        return np.nan
    cum = cum / cum[-1]
    return float(np.interp(fraction, cum, r[order])) * scale_arcsec


def negative_power(kernel):
    """Fraction of the kernel's absolute power that is negative, in percent.

    Relates to Aniano's W- by  pct = W- / (1 + 2 W-) * 100, since
    integral(|K|) = 1 + 2 W- for a unit-sum kernel.
    """
    total = np.sum(np.abs(kernel))
    if total <= 0:
        return np.nan
    return 100.0 * abs(np.sum(kernel[kernel < 0])) / total


def aniano_D(psf_target, convolved):
    """D = integral |psi_B - K * psi_A|  (Aniano et al. 2011, Eq. 20).

    Both inputs normalised to unit sum first, so D is comparable across runs.
    A perfect kernel gives D = 0.
    """
    a = psf_target / np.nansum(psf_target)
    b = convolved / np.nansum(convolved)
    return float(np.nansum(np.abs(a - b)))


def radial_profile(data, scale_arcsec, n_bins=60):
    """Median flux in concentric annuli. Returns (radii_arcsec, profile)."""
    peak = np.array(np.unravel_index(np.nanargmax(data), data.shape))
    y, x = np.indices(data.shape)
    r = np.hypot(y - peak[0], x - peak[1])
    r_max = min(data.shape) // 2
    edges = np.linspace(0, r_max, n_bins + 1)
    centres, prof = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() > 0:
            centres.append(0.5 * (lo + hi) * scale_arcsec)
            prof.append(np.nanmedian(data[m]))
    return np.array(centres), np.array(prof)


def run_pypher(src_path, tgt_path, out_path, r_value):
    """Call PyPHER and return the kernel, or None if it failed."""
    # pypher refuses to overwrite, so clear any leftover from a previous run
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    out_path.with_suffix('.log').unlink(missing_ok=True)

    cmd = ["pypher", str(src_path), str(tgt_path), str(out_path),
           "-r", f"{r_value:.3e}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("    [!] pypher is not on PATH. Activate the environment that "
              "has it, or install with: pip install pypher")
        return None
    if res.returncode != 0 or not Path(out_path).exists():
        print(f"    [!] pypher failed for r={r_value:.0e}")
        if res.stderr:
            print("        " + res.stderr.strip().splitlines()[-1])
        return None
    return fits.getdata(out_path).astype(np.float64)


def evaluate(psf_source, psf_target, kernel, scale):
    """All four metrics for one kernel."""
    kernel = kernel / kernel.sum()            # unit sum, as the pipeline does
    convolved = fftconvolve(psf_source, kernel, mode='same')
    r50_conv = enclosed_radius(convolved, scale, 0.5)
    r50_tgt = enclosed_radius(psf_target, scale, 0.5)
    r80_conv = enclosed_radius(convolved, scale, 0.8)
    r80_tgt = enclosed_radius(psf_target, scale, 0.8)
    return {
        'ratio_r50': r50_conv / r50_tgt if r50_tgt > 0 else np.nan,
        'ratio_r80': r80_conv / r80_tgt if r80_tgt > 0 else np.nan,
        'neg_pct': negative_power(kernel),
        'D': aniano_D(psf_target, convolved),
        'convolved': convolved,
    }


def sweep(label, src_path, tgt_path, psf_source, psf_target, scale, workdir):
    """Run the regularisation sweep for one pair and print the table."""
    # pypher writes a .log next to the kernel and does not create the folder
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"{'r':>10} {'r50 ratio':>11} {'r80 ratio':>11} "
          f"{'neg power':>11} {'D':>11}")
    print("-" * 74)

    results = {}
    for r in R_VALUES:
        out = workdir / f"kernel_r{r:.0e}.fits"
        kernel = run_pypher(src_path, tgt_path, out, r)
        if kernel is None:
            continue
        m = evaluate(psf_source, psf_target, kernel, scale)
        results[r] = m
        print(f"{r:>10.0e} {m['ratio_r50']:>11.4f} {m['ratio_r80']:>11.4f} "
              f"{m['neg_pct']:>10.2f}% {m['D']:>11.3e}")
    return results


def verdict(res_syn, res_real):
    """Interpret the two sweeps."""
    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")

    if not res_syn:
        print("The synthetic control produced nothing - check the PyPHER call.")
        return

    default = 1e-4
    syn_def = res_syn.get(default)
    if syn_def is not None:
        ratio = syn_def['ratio_r50']
        print(f"\n1. Synthetic control at the default r=1e-4: ratio = {ratio:.4f}")
        if abs(ratio - 1.0) < 0.01:
            print("   The method is exact where it should be. A 1.06 on the real")
            print("   PSFs therefore comes from something specific to those PSFs,")
            print("   not from PyPHER or from the way r50 is measured.")
        else:
            print("   The method does NOT recover the analytic answer on a case")
            print("   where it must. The discrepancy is in the method itself -")
            print("   regularisation, sampling, or the r50 measurement - and not")
            print("   a property of the real PSFs.")

    ratios = [m['ratio_r50'] for m in res_syn.values()]
    spread = max(ratios) - min(ratios)
    print(f"\n2. Spread of the r50 ratio across r (synthetic): {spread:.4f}")
    if spread > 0.01:
        print("   The ratio MOVES with r: regularisation is a real contributor.")
        print("   Note the negative-power column - lowering r buys fidelity at")
        print("   the cost of ringing, and the trade-off is the actual result.")
    else:
        print("   The ratio barely moves with r: regularisation is NOT the")
        print("   cause. Look instead at sampling (how many pixels across the")
        print("   source PSF on this grid) and at kernel truncation.")

    if res_real:
        ratios_r = [m['ratio_r50'] for m in res_real.values()]
        print(f"\n3. Real pair, ratio range: "
              f"{min(ratios_r):.4f} to {max(ratios_r):.4f}")
        in_crit = [r for r, m in res_real.items() if 0.95 <= m['ratio_r50'] <= 1.05]
        if in_crit:
            print(f"   Values of r meeting the 0.95-1.05 criterion: "
                  f"{[f'{r:.0e}' for r in in_crit]}")
        else:
            print("   No value of r brings the real pair inside 0.95-1.05.")
            print("   Changing r alone will not satisfy the criterion; the grid")
            print("   change (kernels on the native grid) has to be tested next.")


# ------------------------------------------------------------------- main ---
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workdir = OUT_DIR / "kernels"
    workdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Block 1 + 2a: synthetic control ----------------
    psf_syn_src = gaussian_psf(SYN_SIZE, SYN_FWHM_SOURCE, GRID_ARCSEC)
    psf_syn_tgt = gaussian_psf(SYN_SIZE, SYN_FWHM_TARGET, GRID_ARCSEC)

    p_src = OUT_DIR / "synthetic_source.fits"
    p_tgt = OUT_DIR / "synthetic_target.fits"
    write_psf(p_src, psf_syn_src, GRID_ARCSEC)
    write_psf(p_tgt, psf_syn_tgt, GRID_ARCSEC)

    # sanity: the analytic kernel must give exactly 1.000
    sigma_src = SYN_FWHM_SOURCE * FWHM_TO_SIGMA / GRID_ARCSEC
    sigma_tgt = SYN_FWHM_TARGET * FWHM_TO_SIGMA / GRID_ARCSEC
    sigma_k = np.sqrt(sigma_tgt ** 2 - sigma_src ** 2)
    k_exact = gaussian_psf(SYN_SIZE, sigma_k * GRID_ARCSEC / FWHM_TO_SIGMA,
                           GRID_ARCSEC)
    m_exact = evaluate(psf_syn_src, psf_syn_tgt, k_exact, GRID_ARCSEC)
    print(f"{'=' * 74}")
    print("SANITY: analytic kernel on the synthetic pair")
    print(f"{'=' * 74}")
    print(f"   r50 ratio = {m_exact['ratio_r50']:.4f}   (must be 1.000)")
    print(f"   D         = {m_exact['D']:.3e}")
    print(f"   source PSF sampling: {SYN_FWHM_SOURCE / GRID_ARCSEC:.2f} px/FWHM")
    if SYN_FWHM_SOURCE / GRID_ARCSEC < 2:
        print("   [!] the source PSF is UNDERSAMPLED on this grid (< 2 px/FWHM);")
        print("       the convolved core is then dominated by the pixel itself,")
        print("       which is a candidate explanation for a biased r50.")

    res_syn = sweep("BLOCK 2a - regularisation sweep, SYNTHETIC pair",
                    p_src, p_tgt, psf_syn_src, psf_syn_tgt,
                    GRID_ARCSEC, workdir / "syn")

    # ---------------- Block 2b: real pair ----------------
    res_real = {}
    if REAL_SOURCE.exists() and REAL_TARGET.exists():
        psf_real_src = fits.getdata(REAL_SOURCE).astype(np.float64)
        psf_real_tgt = fits.getdata(REAL_TARGET).astype(np.float64)

        scale_src = fits.getheader(REAL_SOURCE).get('PIXSCALE', GRID_ARCSEC)
        scale_tgt = fits.getheader(REAL_TARGET).get('PIXSCALE', GRID_ARCSEC)
        if abs(scale_src - scale_tgt) / scale_tgt > 0.02:
            print(f"\n[!] the two real PSFs are on DIFFERENT grids "
                  f"({scale_src} vs {scale_tgt}). The kernel would be invalid; "
                  f"skipping the real pair.")
        else:
            res_real = sweep(
                f"BLOCK 2b - regularisation sweep, REAL pair\n"
                f"  {REAL_SOURCE.name}\n  -> {REAL_TARGET.name}",
                REAL_SOURCE, REAL_TARGET, psf_real_src, psf_real_tgt,
                scale_src, workdir / "real")

            # ---------------- Block 3: radial profile ----------------
            k_def = fits.getdata(workdir / "real" / "kernel_r1e-04.fits")
            m_def = evaluate(psf_real_src, psf_real_tgt, k_def.astype(np.float64),
                             scale_src)
            r_c, p_c = radial_profile(m_def['convolved'], scale_src)
            r_t, p_t = radial_profile(psf_real_tgt, scale_src)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.semilogy(r_t, np.abs(p_t), label='target PSF', lw=2)
            ax1.semilogy(r_c, np.abs(p_c), '--', label='source * kernel', lw=2)
            ax1.set_xlabel('radius (arcsec)')
            ax1.set_ylabel('flux (log)')
            ax1.set_title('Radial profile, r = 1e-4')
            ax1.legend()
            ax1.grid(alpha=0.3)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_prof = np.abs(p_c) / np.abs(p_t)
            ax2.plot(r_t, ratio_prof, lw=2)
            ax2.axhline(1.0, color='k', ls=':', lw=1)
            ax2.set_xlabel('radius (arcsec)')
            ax2.set_ylabel('(source * kernel) / target')
            ax2.set_title('Ratio per radius - where the kernel errs')
            ax2.set_ylim(0, 2)
            ax2.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(OUT_DIR / 'radial_profile.png', dpi=140)
            print(f"\nRadial profile saved to {OUT_DIR / 'radial_profile.png'}")
            print("  core off but wings on  -> discretisation")
            print("  wings off but core on  -> truncation or regularisation")
            print("  uniformly wide         -> global smoothing")
    else:
        print(f"\n[!] real PSF files not found; skipping blocks 2b and 3.")
        print(f"    looked for: {REAL_SOURCE}")

    verdict(res_syn, res_real)


if __name__ == "__main__":
    main()