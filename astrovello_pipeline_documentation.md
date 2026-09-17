# AsTrovello 2.0 — Pipeline Documentation

**Multi-survey datacube construction for spatially-resolved SED fitting**

This document describes the AsTrovello 2.0 pipeline module by module and
function by function. For each stage it states *what* the code does, *why* it
does it that way, and *which literature* supports the choice. It is written to
serve both as a maintenance reference and as the methodological basis for the
dissertation chapter on data reduction.

> **Bibliographic caveat.** Two references were read directly from the PDFs and
> their quotations are verbatim and verified: **Aniano et al. (2011)** and
> **Boucaud et al. (2016)**. All *other* references in Section 12 were compiled
> from memory, without access to a citation database — their titles and
> existence are reliable, but year, volume and page should be verified on ADS
> before citing.

---

## Table of contents

1. [Overview and design principles](#1-overview-and-design-principles)
2. [The survey drivers (`drivers.py`)](#2-the-survey-drivers-driverspy)
3. [PSF measurement and master selection](#3-psf-measurement-and-master-selection)
4. [PSF matching and kernel generation](#4-psf-matching-and-kernel-generation)
5. [Convolution](#5-convolution)
6. [Alignment / reprojection (`alignment_2_0.py`)](#6-alignment--reprojection-alignment_2_0py)
7. [Unit conversion (`units_2_0.py`)](#7-unit-conversion-units_2_0py)
8. [Datacube assembly (`cube_2_0.py`, `mask_2_0.py`)](#8-datacube-assembly-cube_2_0py-mask_2_0py)
9. [Preflight validation (`preflight.py`)](#9-preflight-validation-preflightpy)
10. [The CLI orchestrator (`astrovello_cli_2.0.py`)](#10-the-cli-orchestrator-astrovello_cli_20py)
11. [Configuration (`config.py`)](#11-configuration-configpy)
12. [Annotated bibliography](#12-annotated-bibliography)

---

## 1. Overview and design principles

AsTrovello ingests imaging of a galaxy from several surveys (PHANGS-HST,
PHANGS-JWST, S4G/Spitzer) and produces a single spatially-registered datacube
in Jy/pixel, on a common pixel grid and at a common angular resolution. That
cube is the input to the segmentation and SED-fitting stages (Capivara).

The pipeline runs in four stages, in this order:

1. **PSF matching + convolution** — degrade every band to the resolution of
   the worst-resolved band (the *master*).
2. **Alignment** — reproject every band onto the master's pixel grid.
3. **Unit conversion** — convert every band to Jy/pixel.
4. **Datacube** — stack the bands, subtract sky, mask, and write the cube.

Three design principles run through the whole codebase, and most of the
non-obvious code exists to enforce them:

- **Information travels with the data.** Pixel scale, native pixel area,
  binning factor, PSF-matching provenance — all are read from the FITS header
  at the point of use, never assumed from a constant elsewhere. Header keys
  such as `NATPXAR`, `BINFACT`, `PSFTARGT`, `PSFMATCH`, `PSFRESID` carry these
  decisions forward.
- **Verify the transformation, not just that it ran.** Each stage that
  transforms the data has a check that the transformation did what it should —
  most importantly the kernel-reproduces-target test (Section 3) and the
  grid-consistency guard (Section 5).
- **Derive from the data; configure only what cannot be derived.** Pixel
  scales come from the WCS; PSF widths from the images; the master and the
  binning factor from those widths. `config.py` holds only survey-specific
  facts that cannot be measured (file-naming conventions, unit strings,
  reference PSF scales).

The order convolution → alignment is deliberate and is the single most
important architectural decision. It is justified quantitatively in Section 6.

---

## 2. The survey drivers (`drivers.py`)

Each survey stores its data differently: different filename conventions,
different flux units, different invalid-pixel conventions, different science
HDU. The driver pattern isolates all of this behind a common interface, so the
rest of the pipeline is survey-agnostic. Adding a survey means writing a driver,
not editing the pipeline.

### `BASE_Driver`

The abstract base. It defines the interface every survey must implement and
provides the survey-dispatch logic (`get_survey`, which maps a file path to the
survey that owns it). Methods that must be survey-specific
(`get_invalid_mask`, `convolve`, `convert2Jansky`) raise `NotImplementedError`
so that an incomplete driver fails loudly rather than silently misbehaving —
the preflight (Section 9) checks for exactly this.

Key properties on the base:

- `get_convolution_bin_factor` — the pre-convolution binning factor for this
  survey (Section 5). Reads `convolution_bin_factor` from config, default 1.
- `get_hdu_sci_position` — which FITS HDU holds the science image. JWST mosaics
  use HDU 1; HST and S4G use HDU 0. This is read only when opening the
  *original* mosaics; every file the pipeline itself writes uses HDU 0.

### `PHANGS_Driver` (HST)

- **Units.** PHANGS-HST drizzled mosaics are in electrons/s. `convert2Jansky`
  multiplies by `PHOTFNU` (the header keyword giving the Jy-per-(electron/s)
  conversion for one native pixel) and then rescales to the actual pixel area
  read from the WCS (see Section 7 for why the two-step form is necessary).
- **Invalid pixels.** Off-footprint pixels are marked with exact 0
  (`get_invalid_mask` returns `img_data == 0`). This is safe because
  electrons/s is essentially never exactly zero on a real pixel.
- **`convolve`.** Described in Section 5.

### `PHANGS_JWST_Driver`

- **Units.** JWST (NIRCam + MIRI) mosaics are in **MJy/sr**, a surface
  brightness — *not* electrons/s. `convert2Jansky` therefore uses the S4G-style
  route: multiply by the pixel solid angle from the WCS. This is
  grid-independent, so it stays correct after reprojection. Using `PHOTFNU`
  here would be wrong (JWST headers do not even carry it).
- **Invalid pixels.** PHANGS-JWST level-3 mosaics mark off-footprint pixels
  with exact 0, like HST (confirmed empirically on the NGC 1087 MIRI mosaics:
  0 NaN, ~1.5M exact zeros forming the tilted-frame border). Isolated interior
  zeros (<0.03% of pixels) are handled by the border-connectivity logic in
  `convolve`.
- **Two pixel scales.** NIRCam (~0.031–0.063″/px) and MIRI (~0.11″/px) differ,
  so `get_pixel_scale` indexes by filter.

### `S4G_Driver` (Spitzer/IRAC)

- **Units.** S4G `.phot` mosaics are in MJy/sr; same conversion as JWST.
- **SIP distortion.** S4G headers carry real 3rd-order SIP distortion
  coefficients but omit the `-SIP` suffix on `CTYPE`. `get_sip` returns True so
  that the alignment stage re-attaches the suffix, making the header
  self-consistent with the coefficients that are already present.
- **Native vs mosaic scale.** The IRAC detector is 1.221″/px, but the `.phot`
  mosaics are drizzled to 0.75″/px. This distinction is the origin of the
  original unit bug and is why pixel scales are read from the WCS, never from
  the config constant.

> **Reference.** The driver/adapter separation is standard software practice;
> the survey-specific facts come from the instrument documentation: WFC3 Data
> Handbook (STScI) for `PHOTFNU`; IRAC Instrument Handbook (IPAC) and
> Muñoz-Mateos et al. (2015, ApJS) for the S4G photometric mosaics; the
> PHANGS-JWST data release (Williams et al. 2024) for the MIRI/NIRCam mosaics.

---

## 3. PSF measurement and master selection

PSF matching degrades every band to the resolution of the worst-resolved band.
Choosing that band correctly is the first decision, and it is not trivial.

### `get_fwhm(data)`

FWHM in pixels from a 2D Gaussian fit (`astropy.modeling`). Fast, but it
responds to the *core* of the PSF and underestimates the true width by ~25% for
profiles with broad wings such as the IRAC PRF. **It is used only for an order
of magnitude, never to rank two similar PSFs** — a warning to that effect is in
its docstring.

### `get_half_light_radius(data, pixel_scale, fraction)`

The radius enclosing a given fraction of the flux (default 50%; the master
criterion uses 80%, `r80`). Profile-shape-independent: it sorts pixels by
distance from the peak, accumulates the flux, and interpolates where the
cumulative curve crosses the requested fraction. A robust background (median of
the outer annulus) is subtracted first, otherwise the cumulative sum grows with
area and the radius loses meaning.

### `calculate_half_light_radii(...)` and master selection

The master is the band with the largest `r80`, not the largest Gaussian FWHM.
This matters because **PSF matching is dominated by the wings, not the core**.
The two IRAC channels differ by only ~4%; the Gaussian FWHM ordered them
*backwards* (it reported channel 1 as wider, when literature and r80 both give
channel 2 as wider). Choosing the wrong master turns a smoothing kernel into a
sharpening (deconvolution) kernel — measured as 31% negative kernel power
versus 0% for the correct choice.

> **Reference.** The use of enclosed-energy radius over Gaussian FWHM for
> IR PSFs with extended wings follows Aniano et al. (2011, PASP 123, 1218),
> the canonical reference for convolution kernels between space-based
> instruments. The extended-wing structure of the IRAC PRF is documented in the
> IRAC Instrument Handbook (IPAC).

**Why r80 and not r100:** r100 would include unbounded background noise (the
cumulative curve never truly converges); r50 barely differs from the FWHM and
misses the wings. r80 sits where the signal still dominates the noise but the
wings are already included — consistent with the 70–90% range used by Aniano
et al. (2011).

---

## 4. PSF matching and kernel generation

> **Note.** The method described in this section follows Section 4 of
> Aniano et al. (2011, PASP 123, 1218). The choice of enclosed-energy radius
> over Gaussian FWHM (Section 3 of this document) is motivated by the extended
> wings of IR PSFs, also discussed there.

### Foreword — validation of the kernels

PSF matching is a delicate operation, and a badly built kernel produces an
incorrect convolution without raising any error. Several diagnostics can
detect this. The tests implemented in AsTrovello were adapted from the
performance metrics of Aniano et al. (2011, PASP 123, 1218, Sect. 5) and are
described below. They are reported by `inspect_kernel` at convolution time and
by `diagnostico_etapa1c.py` as a standalone check.

#### 1. The kernel must reproduce the target PSF

Convolving the source PSF with the kernel must yield the target PSF. Expressed
as a ratio of enclosed-energy radii:

$$\frac{R50_{\text{convolved}}}{R50_{\text{target}}} = 1$$

where $R50_{\text{convolved}}$ is the radius enclosing 50% of the energy of
$\Psi_A \star K$, and $R50_{\text{target}}$ is the same quantity for $\Psi_B$.
The same test can be run with $R_{80}$, which is preferable when the PSF has
significant power in the wings:

$$\frac{R80_{\text{convolved}}}{R80_{\text{target}}} = 1$$

**Aniano et al. (2011), Sect. 5 ("Kernel Performance")**:

> *"For each generated kernel, we compute $\Psi_A \star K\{A \Rightarrow B\}$
> the convolution of $\Psi_A$ and $K\{A \Rightarrow B\}$, and compare it with
> $\Psi_B$. **For a perfect kernel, both quantities should coincide at all
> radii.**"*

> *"One measure of kernel performance is its accuracy in redistribution of PSF
> power. We define*
> $$D = \int\!\!\int |\Psi_B - K\{A \Rightarrow B\} \star \Psi_A|\, dx\, dy \tag{20}$$
> *A kernel with perfect performance will have $D = 0$."*

Our ratio test is a scalar reduction of the same comparison: $D$ integrates the
difference at every radius, while the radius ratio summarises it in one number
that is directly readable. **Measured: 0.08 before the grid correction, ~1.06
after** — a factor ~12 shortfall in the applied blur, later traced to the
kernel being generated on a different grid from the one it was applied to.

#### 2. The kernel must have small negative power

A smoothing kernel is almost entirely positive. Negative lobes indicate that
the kernel is sharpening (deconvolving) rather than blurring, which amplifies
noise and produces ringing near strong gradients. Aniano et al. quantify this
with $W_-$, the integral of the negative part of the kernel:

$$W_{\pm} = \frac{1}{2} \int\!\!\int \left( |K\{A \Rightarrow B\}| \pm K\{A \Rightarrow B\} \right) dx\, dy \tag{21}$$

**Aniano et al. (2011), Sect. 5**:

> *"A second quantitative measure of kernel performance is obtained by studying
> its negative values. […] Flux conservation requires that $W_+ = 1 + W_-$. In
> general, kernels will have $W_- > 0$. **Well-behaved kernels have small $W_-$
> values**: $W_-(K\{M24 \Rightarrow S250\}) = 0.07$. The integral of
> $|K\{A \Rightarrow B\}|$ is $[1 + 2W_-]$, so **a kernel with a large value of
> $W_-$ could potentially amplify image artifacts**."*

and, in Sect. 7, the usage threshold:

> *"**We do not recommend using any kernel with $W_- \gtrsim 1.2$.**"*

`inspect_kernel` reports the negative power as a percentage of the total
absolute power, which relates to $W_-$ by

$$\text{negative power (\%)} = \frac{W_-}{1 + 2W_-} \times 100$$

so the well-behaved example of Aniano et al. ($W_- = 0.07$) corresponds to
6.1% on our scale. **Measured: 0.00% for the five HST → IRAC kernels** (better
than their canonical example, as expected when the source PSF is far narrower
than the target), **versus ~31% for the IRAC1 ↔ IRAC2 pair in either
direction** ($W_- \approx 0.8$) — below their 1.2 limit, but already well
inside the problematic regime, and combined with the sub-pixel required blur
(criterion 3) it is why that pair is not matched.

#### 3. The required blur must be resolvable on the working grid

Matching two PSFs requires a kernel whose width adds in quadrature:

$$\text{required blur} = \sqrt{R80_{\text{target}}^{2} - R80_{\text{source}}^{2}}$$

If this is smaller than roughly two pixels of the convolution grid, the kernel
degenerates into a near-delta function and the Fourier division returns sinc
side lobes — ringing, not matching. The underlying condition is stated by
Aniano et al. in Fourier space, via the cutoff frequency
$k_{H,A} = \kappa_A \times 2\pi/\text{FWHM}_A$ at which $FT(\Psi_A)$ falls to
$5\times10^{-3}$ of its maximum (their Eq. 9):

> *"PSF Fourier transforms do not have significant power at (spatial)
> frequencies above the $2\pi/\text{FWHM}$. The high-frequency components of the
> FT will be small, **introducing large uncertainties when inverted**."*

When the two PSFs have similar widths their cutoff frequencies nearly coincide,
the regularising filter dominates, and the kernel becomes unreliable — the
regime Aniano et al. describe as depending on "the exact form of the filter".
Our real-space criterion is the equivalent statement, expressed in pixels of
the working grid so it answers directly whether the operation fits.

**Measured for IRAC1 → IRAC2:** required blur
$\sqrt{2.1878^2 - 2.0864^2} = 0.658''$, which is 0.88 px on the 0.75″/px S4G
grid — below the 2 px limit. The pair is therefore left unmatched, with the
residual mismatch recorded in the `PSFRESID` header keyword.

#### 4. The kernel must be centred

The convolved band must not be shifted relative to the others; a sub-pixel
offset in the kernel becomes a per-region colour error, which is the hardest
kind to detect downstream. Aniano et al. (2011), Sect. 7, require the kernel to
be *"centered (to avoid shifts in the image)"*.

`inspect_kernel` reports the centroid offset from the array centre.
**Measured: ≤ 0.04 px for the HST kernels.** An earlier parity-crop bug (using
`data[:-1, :]` to force an odd size, which shifts the content instead of
recentring it) produced a +0.375″ offset — exactly half a pixel — on the
IRAC → 0.75″ path; `_recenter_odd` reduced it to +0.0000″.

### The core principle

A convolution kernel is valid **only on the grid of the PSFs used to build
it**. PyPHER (the tool used) returns the kernel on the grid of the input PSFs.
Since that kernel is then applied to a science image, the PSFs must be on the
grid where the convolution will happen — not on any detector's native grid.

Violating this was the original pipeline's central bug: kernels were built on
the target (IRAC) grid but applied on the HST native grid, so the effective
blur was ~12× too small and the HST bands never reached the target resolution.
The `diagnostico_etapa1c.py` script measured this directly: the ratio
r50(convolved source)/r50(target) was 0.08 before the fix and ~1.06 after.

### `clean_psf(...)`

Resamples a PSF onto the convolution grid. Takes `target_pixel_scale_arcsec`
**explicitly** — this is the enforcement mechanism for the core principle.

Resampling strategy (the important detail):

- **Large downscale (ratio ≥ 2):** the bulk goes through `block_reduce`
  (exact area averaging — the integral over the target pixel, which does not
  alias), and only the non-integer remainder goes through cubic interpolation
  (`scipy.ndimage.zoom`). Downscaling by interpolation alone would alias.
- **Upscale or small change:** cubic interpolation is safe because the input
  PRF is already oversampled.

Other steps: negatives from cubic interpolation are clipped; optional wing
truncation (default off — truncating the IRAC PRF at 16″ discards ~5% of the
flux and, after renormalisation, leaves the kernel too concentrated,
under-blurring); centring and odd-parity via `_recenter_odd`; normalisation to
unit sum; optional `output_size` padding so PyPHER receives equal-size arrays.

### `_recenter_odd(data)`

PyPHER requires odd-sized arrays. The naive way (drop the last row/column)
*shifts* the content by half a pixel instead of recentring it — and half a
pixel in the PSF becomes half a pixel of offset in the convolved band relative
to the others, i.e. a per-region colour error. This function instead computes
the centroid, sub-pixel-shifts it onto an integer index, and crops
symmetrically around it.

### `required_blur`, `psf_matching_resolvable`, `choose_bin_factor`, `report_convolution_grid`

These four functions formalise two independent quantitative limits.

- **`required_blur(source, target)`** — the kernel width needed to match two
  PSFs, added in quadrature: √(target² − source²).
- **`psf_matching_resolvable(...)`** — whether that blur fits on the grid.
  Below ~2 pixels the kernel degenerates into a near-delta and the Fourier
  division returns sinc side-lobes (ringing), not matching. This is why the
  IRAC1↔IRAC2 pair is *not* matched: the required blur is 0.658″ = 0.88 px on
  the 0.75″/px grid, unresolvable in either direction (~30% negative power both
  ways). Those bands are written unmatched with the residual recorded in
  `PSFRESID`.
- **`choose_bin_factor(...)`** — derives the pre-convolution binning factor
  (Section 5) from the target, subject to two constraints: the grid must sample
  the master PSF (≥ ~3 px/FWHM) and resolve the smallest required blur. The
  binning factor is a property of the *target*, not the source survey — a
  per-survey constant silently breaks when the survey combination changes.
- **`report_convolution_grid(...)`** — prints the consequences of the chosen
  grid per pair, feeding the preflight.

### `pypher_kernel_creation(...)`

Builds the PyPHER commands for one grid. Before emitting any command it
**verifies that every PSF is on the requested grid** (raises otherwise) — the
guard that prevents the original grid bug from ever returning silently. Given
`psf_widths`, it skips unresolvable pairs and returns them for the unmatched
route. Uses `clear_dir=False` inside the per-survey loop so the second survey
does not wipe the first survey's kernels.

### Literature mapping — PSF matching

This subsection maps each of our functions onto the section of the source
papers that prescribes the method. Quotations are verbatim.

#### The kernel is defined on the grid of the input PSFs

**Boucaud et al. (2016), Sect. 2.2, Algorithm 1** states the output grid
explicitly in its first line:

> *"inputs: h_a 2D array of size N_a × N_a and pixel scale p_a, h_b 2D array
> of size N_b × N_b and pixel scale p_b […] **output: k_{a,b} 2D array of size
> N × N and pixel scale p. N = N_b ; p = p_b***"

and the warping step that follows:

> *"if p_a ≠ p then rescale h_a to the pixel scale p"*

That is, **PyPHER returns the kernel on the pixel scale of the target PSF**
(`p = p_b`), rescaling the source PSF to match. This single line is the formal
statement of the bug that AsTrovello 1.x contained: the kernel was produced on
`p_b` (the IRAC grid) and then applied to an image on a different grid. Our
`clean_psf(..., target_pixel_scale_arcsec=...)` enforces that both PSFs — and
therefore `p` — are the *convolution* grid.

**Aniano et al. (2011), Sect. 7 ("Usage of the Kernels")** gives the same
requirement as an operational instruction:

> *"The kernels K{A ⇒ B} computed here are given on a 0.20″ grid. **Before
> performing an image convolution, the kernel K{A ⇒ B} should be resampled
> onto a grid with the same pixel size as the original image I_A.** The
> resampled kernels should be centered (to avoid shifts in the image) and
> normalized so that ∫∫K{A ⇒ B}(x,y)dxdy = 1 to ensure flux conservation. The
> flux in the image to be convolved should be in surface brightness units."*

Three of our design decisions are in that single paragraph: the kernel must be
on the image's grid; it must be centred to avoid shifts (our `_recenter_odd`);
and it must be normalised to unit sum (our `kernel_norm`). The last sentence —
surface brightness units — is also why `convert2Jansky` works through a genuine
surface brightness (Sect. 7 of this document).

#### Resampling PSFs onto a common grid

**Aniano et al. (2011), Sect. 4.2 ("Resample the PSFs")**:

> *"Each PSF comes in a grid of different pixel size. We transform each PSF
> into a grid of a common pixel size of 0.20″ […] The 0.20″ pixel size capture
> all the details on the instrumental and Gaussian PSFs. We also pad with 0 the
> resulting images into an odd-sized square array if needed."*

Our `clean_psf` does exactly this, with two differences justified by our
context: (i) the common grid is not a fixed 0.20″ but the convolution grid
derived per run (`choose_bin_factor`), because our targets vary; and (ii) for
large downscale factors we use `block_reduce` (exact area averaging) for the
integer part rather than pure interpolation, since our ratios reach 20× where
interpolation alone would alias. Note that Aniano et al. use 0.10″ instead of
0.20″ specifically "to regenerate the kernels from optical PSFs into IRAC
cameras" — i.e. they too refine the grid when the source PSF is much narrower
than the target, which is precisely the HST → IRAC case.

#### Centring the PSF

**Aniano et al. (2011), Sect. 4.3 ("Center the PSFs")**:

> *"To determine the image center, we smooth the image with a 5 pixel radius
> circular kernel, and locate the image maximum. In some PSFs the maximum value
> is achieved over a (small) ring. To avoid possible misidentification of the
> real image center, we take the centroid of all the pixels that satisfy
> [max(Ψ) − Ψ(x,y)]/max(Ψ) ≤ 5 × 10⁻⁴."*

Our `_recenter_odd` uses the flux centroid of the whole array rather than the
centroid of near-peak pixels, then performs a sub-pixel shift so the centroid
falls on an integer index. The motivation is the one Aniano et al. state in
Sect. 7 — *"centered (to avoid shifts in the image)"*. Our measured case: the
naive parity crop left a +0.375″ (half-pixel) offset on the IRAC→0.75″ path,
which the recentring reduced to +0.0000″.

#### Odd-sized arrays and truncation

**Aniano et al. (2011), Sect. 4.14 ("Final trim of the kernels")**:

> *"We trim each kernel to a smaller size (to speed up further convolution)
> such that it contains **99.9% of the total kernel energy**. Moreover, we use
> a square grid with an **odd number of pixels so that the kernel peaks in a
> single central pixel**."*

Two prescriptions we follow. The odd-array requirement is why `_recenter_odd`
exists. The 99.9% energy criterion is why our default is
`max_extent_arcsec = 0` (no truncation): our earlier 16″ truncation retained
only ~94.9% of the IRAC PRF energy — an order of magnitude more loss than
Aniano et al. permit — and after renormalisation left the kernel too
concentrated, under-blurring the image.

#### Kernel resampling to the application grid

**Aniano et al. (2011), Sect. 4.13 ("Resample the kernels")**:

> *"All the computed kernels are given in a grid of a common pixel size of
> 0.20″, but will be needed in grids of different pixel sizes. Again, we
> resample the kernels […]"*

Their workflow builds kernels on one grid and resamples the *kernel* to the
image grid. Ours builds the kernel directly on the convolution grid by
resampling the *PSFs* first. Both satisfy the Sect. 7 requirement; ours avoids
a second interpolation of the kernel itself.

#### The existence condition and the resolvability limit

**Aniano et al. (2011), Sect. 2** derives the kernel by Fourier division,

> *"K{A ⇒ B} = FT⁻¹( FT(Ψ_B) × 1/FT(Ψ_A) ) […] Equation (8) provides a
> condition for the existence of such a kernel […] **a condition for the
> existence of a kernel is that the Fourier components for which FT(Ψ_A) = 0
> should satisfy FT(Ψ_B) = 0. Informally speaking, this means that the PSF of
> camera A must be narrower than the PSF of camera B.**"*

and on the failure mode when that does not hold:

> *"In the cases of k_{H,A} < k_{H,B} (convolving into narrower PSFs), use of
> the filter f_A allows one to compute convolution kernels, but **their
> performance can be poor**."*

This is the formal basis for two of our choices. First, selecting the master by
the **widest** PSF (`calculate_half_light_radii`): the target must be the
broadest, or the operation becomes a deconvolution. Second, our
`psf_matching_resolvable` guard: Aniano et al. restrict their published kernels
to pairs with `FWHM_B ≥ FWHM_A / 1.35` (Sect. 7), i.e. they decline to publish
kernels that sharpen by more than 35%. Our IRAC1↔IRAC2 pair fails an even more
basic criterion — the required blur, √(r80_B² − r80_A²) = 0.658″, is 0.88 px on
the 0.75″/px grid, so the kernel cannot be represented at all. We therefore
skip it and record `PSFRESID`, in the same spirit as their refusal to publish
poorly-performing kernels.

#### Negative kernel power (our `inspect_kernel`) — the W⁻ statistic

**Aniano et al. (2011), Sect. 5** defines exactly the metric our
`inspect_kernel` reports:

> *"A second quantitative measure of kernel performance is obtained by studying
> its negative values. We define W± = ½ ∫∫ ( |K{A ⇒ B}| ± K{A ⇒ B} ) dxdy […]
> **Well-behaved kernels have small W⁻ values**: W⁻(K{M24 ⇒ S250}) = 0.07. The
> integral of |K{A ⇒ B}| is [1 + 2W⁻], so **a kernel with a large value of W⁻
> could potentially amplify image artifacts**. Additionally, a kernel with
> large W⁻ can generate areas of negative flux near point sources […]"*

and sets a usage threshold in Sect. 7:

> *"Kernels with FWHM_A ≳ FWHM_B (that have larger W⁻ values) tend to perform
> poorly and should be used with care. **We do not recommend using any kernel
> with W⁻ ≳ 1.2.**"*

Our `inspect_kernel` prints the negative power as a percentage of the total
absolute power, which is a normalised form of the same quantity. Measured
values: 0.00% for the five HST→IRAC kernels (well-behaved, as expected when the
source is far narrower), versus ~31% for the IRAC1→IRAC2 pair in either
direction — the signature Aniano et al. associate with kernels that must
"remove energy from some region to relocate to another region".

#### The kernel-reproduces-target test (our `diagnostico_etapa1c.py`)

**Aniano et al. (2011), Sect. 5** also defines the accuracy metric that our
validation test implements:

> *"One measure of kernel performance is its accuracy in redistribution of PSF
> power. We define D = ∫∫ |Ψ_B − K{A ⇒ B} ⋆ Ψ_A| dxdy. **A kernel with perfect
> performance will have D = 0** […] Good kernels have small D values:
> D(K{M24 ⇒ S250}) = 0.011."*

Our test is the same idea in a scalar form that is easier to read: we convolve
Ψ_A with K, then compare the half-light radius of the result to that of Ψ_B.
The ratio must be 1.0 for a correct kernel. Measured 0.08 before the grid fix
and ~1.06 after — a direct empirical demonstration that the kernel was not
performing the transformation it was supposed to.

#### Why PyPHER rather than the Aniano kernels

**Boucaud et al. (2016), Sect. 1 and 2.1** motivate the choice: the Aniano et
al. kernels are built from *circularised* PSFs (their Sect. 4.4 averages over
2¹⁴ rotations), whereas PyPHER preserves anisotropy —

> *"This method ensures all anisotropic features in the PSFs to [be
> preserved]"*

— and is regularised by a tunable Wiener filter,

> *"we use a technique called regularisation. We choose a ℓ₂ norm to have a
> linear estimator and use Fourier filtering; and penalise the high frequencies
> in which we expect the noise to dominate […] The optimal balance between the
> data and the penalisation is found by setting µ to the inverse of the
> signal-to-noise ratio (S/N) of the homogenised image."*

For our purpose the decisive advantage is that PyPHER accepts arbitrary PSF
pairs at run time (the Aniano et al. kernels are a fixed published library that
does not include WFC3 or NIRCam/MIRI), while implementing the same Fourier
inversion with an explicit regularisation parameter.

> **Other references for this section.** The instability of deconvolution and
> the origin of ringing: Goodman, *Introduction to Fourier Optics*; Bracewell,
> *The Fourier Transform and Its Applications* (the delta ↔ sinc pair — the
> mathematical reason a sub-2-pixel kernel produces side lobes). Area-exact
> resampling: Fruchter & Hook (2002, PASP 114, 144).

---

## 5. Convolution

### `bin_for_convolution(...)`

Bins the image before convolving, with a NaN-aware **mean**. Rationale: the
final product is resampled onto the master grid anyway, so convolving on HST's
native grid (0.0396″/px, ~144M pixels) carries ~25× more pixels than the
physics requires and exhausts memory. Binning 5× leaves 8.6 px/FWHM at the
target resolution — well above the Nyquist minimum of 2.

The **mean** (not sum) is deliberate: each binned pixel value still means
"electrons/s per *native* pixel", which is what the unit conversion expects.
The true native pixel area is written to `NATPXAR` so the later conversion does
not read the (now coarser) WCS area by mistake — this closes the unit bug with
a factor `bin²`. SIP coefficients, being in pixel units, are stripped.

> **Reference.** Area-exact averaging vs point sampling, and Nyquist sampling
> of the PSF: Fruchter & Hook (2002); the sampling theorem in any signal /
> Fourier text (Bracewell). The 2-px-per-FWHM Nyquist criterion for imaging:
> standard, discussed in Howell, *Handbook of CCD Astronomy*.

### `PHANGS_Driver.convolve(...)` (and the JWST/S4G variants)

The actual convolution, `astropy.convolution.convolve_fft`. Three subtleties:

1. **Border handling by connectivity.** Off-footprint zeros are marked NaN;
   the NaN regions *connected to the image border* are identified with
   `scipy.ndimage.label` and a border seed, zeroed before the FFT, and masked
   out afterward with a dilation the size of the kernel. This prevents the
   large border region from bleeding into the interior during convolution —
   and, as a side effect, speeds the FFT up because it no longer interpolates
   over a huge invalid region. Isolated interior NaNs are handled by
   `nan_treatment='interpolate'`.
2. **`normalize_kernel=False`.** The kernel is already normalised to unit sum
   in `create_convolvedFITS`; normalising again would be wrong for a kernel
   whose sum is not exactly 1 after truncation.
3. **`allow_huge=True`** because even binned images are large.

### `create_convolvedFITS(...)`

Orchestrates one band's convolution: `_prepare_image` (mark invalid → bin) →
normalise the kernel → `inspect_kernel` (prints sum, negative power, centroid
offset) → **grid-consistency guard** (raises if the kernel scale and the image
scale differ by >2%, the safety net against a returning grid bug) → convolve →
write with provenance (`PSFMATCH=True`, `PSFTARGT`, `PSFRESID=0`).

### `copy_as_convolved(...)`

Writes a band that is *not* PSF-matched (an unresolvable pair, or the master
itself) with the same preparation (NaN + binning) but no convolution, recording
`PSFMATCH=False` and the residual. Using this for the master (rather than a
plain copy) matters when the master's survey has a binning factor > 1: a plain
copy would leave it on a different grid from its own survey's convolved bands,
and the alignment step would then resample between grids.

### `diagnose_negatives`, `inspect_kernel`

Diagnostics. `inspect_kernel` prints the four metrics that each catch one of
the historical failure modes: size (wrong grid), sum (normalisation), negative
power (wrong master or unresolvable pair), centroid offset (parity-crop shift).
`diagnose_negatives` distinguishes border artefacts from genuine kernel
ringing by locating the negatives and comparing the worst one to the noise.

---

## 6. Alignment / reprojection (`alignment_2_0.py`)

### Why convolution comes before alignment

This is the key architectural decision, and it rests on the Nyquist criterion.
On the final 0.75″/px grid, the master (FWHM 1.72″) is sampled at 2.29 px/FWHM
— above Nyquist, so reprojecting by interpolation loses nothing. But that
guarantee **only holds if the image was already blurred to 1.72″ before being
resampled**. Resampling a still-sharp HST image (FWHM 0.08″) onto 0.75″/px
would give 0.11 px/FWHM — undersampled by ~17×, with severe aliasing. Hence:
convolve first, reproject second.

### `discover_convolved_files(...)`

Selects the files belonging to *one* configuration, keyed on the `PSFTARGT`
header (with the filename as fallback). Filename-only matching is not enough: a
leftover file from a previous run with a different master parses fine and would
be picked up silently. `strict=True` raises if files from another configuration
are present; `--allow_mixed` relaxes it. Parsing is anchored from the right so
galaxy names containing underscores still work.

### `reproject_to_reference(...)`

Reprojects one convolved band onto the master's pixel grid with
`reproject.reproject_interp`. Handles the S4G SIP inconsistency by re-attaching
the `-SIP` suffix when the driver requests it. Writes the output with the
reference WCS and a `COMMENT` recording that surface brightness is preserved
but flux-per-pixel is not strictly conserved by interpolation.

**On `reproject_interp` vs `reproject_adaptive`:** the interpolation choice was
verified not to cost SNR. The kernel's correlation area (~4200 native px) is
~12× larger than one output pixel (~350 native px), so the pixels within an
output pixel share essentially the same noise realisation and averaging them
does not reduce σ. `reproject_adaptive` remains a small improvement (flux
conservation ~1%, border handling) but is not a priority.

### Literature mapping — alignment and reprojection

Neither source paper is *about* reprojection (both are about kernels), but both
constrain it, and one of the constraints is the reason our pipeline convolves
before it reprojects.

#### Surface brightness, not flux per pixel

**Aniano et al. (2011), Sect. 7** is explicit about the units in which the
convolution must be performed:

> *"**The flux in the image to be convolved should be in surface brightness
> units.** After convolving the image I_A with the kernel K{A ⇒ B}, the
> resulting image will be expressed in the original image grid and **original
> surface brightness units**, but with PSF Ψ_B."*

This is the statement behind two of our design choices. First, the convolution
happens before the unit conversion and before reprojection, on data whose
values are a surface-brightness-like quantity ("electrons/s per native pixel"
for HST; MJy/sr for S4G and JWST) — which is why `bin_for_convolution` uses the
**mean**, not the sum: the mean preserves the per-native-pixel meaning, the sum
would not. Second, it is why `convert2Jansky` routes through a genuine surface
brightness (divide by the *native* area, multiply by the *current* area): a
quantity expressed per-pixel is not invariant under a change of grid, whereas a
surface brightness is.

#### Why convolution must precede reprojection

The kernel formalism itself requires it. **Aniano et al. (2011), Sect. 2**
defines the kernel by the relation Ψ_B = Ψ_A ⋆ K{A ⇒ B}, which holds on a
*continuous* field sampled finely enough to represent Ψ_A. If the image is
first resampled onto a grid too coarse to sample Ψ_A, the sampled image no
longer carries Ψ_A and the relation does not apply.

Quantitatively, in our case: the master (FWHM 1.72″) on the final 0.75″/px grid
is sampled at 2.29 px/FWHM, above the Nyquist limit of 2, so reprojecting an
*already-convolved* image by interpolation loses nothing. But an unconvolved
HST image (FWHM 0.08″) on that same grid would be at 0.11 px/FWHM —
undersampled by ~17×, with severe aliasing. Hence the ordering. The sampling
theorem itself is standard (Bracewell; Howell, *Handbook of CCD Astronomy*);
Aniano et al.'s own practice of refining the working grid to 0.10″ when going
"from optical PSFs into IRAC cameras" (their Sect. 4.2) is the same principle
applied to the PSFs rather than to the images.

#### Interpolation versus area-exact resampling

**Fruchter & Hook (2002, PASP 114, 144)** is the reference for what a
resampling operation does to flux and to noise: interpolation preserves values
(a surface-brightness-like behaviour) but does not strictly conserve summed
flux, and it correlates the noise between neighbouring output pixels. Our
`reproject_to_reference` records this in the output header
(`"Surface brightness preserved; flux per pixel not strictly conserved"`),
and it is why `bin_for_convolution` uses area-exact averaging rather than
interpolation for the pre-convolution downscale, where the factor is large.

The decision to keep `reproject_interp` (point sampling) rather than move to
`reproject_adaptive` (area-weighted) was tested rather than assumed: the
kernel's correlation area (~4200 native px) is ~12× larger than one output
pixel (~350 native px), so the native pixels inside an output pixel share
essentially the same noise realisation and averaging them does not reduce σ.
This is the correlated-noise effect Fruchter & Hook describe; the practical
consequence here is that the adaptive scheme would gain ~1% in flux
conservation and better border handling, but no SNR.

#### PSF anisotropy and the rotation of the field

**Boucaud et al. (2016), Algorithm 1** includes a step our pipeline does not
need, and it is worth recording why:

> *"for i in {a, b} do if α_i ≠ 0 then rotate h_i through an angle α_i end end"*

PyPHER rotates each PSF by the position angle of its instrument before building
the kernel, because it deliberately **preserves anisotropic features** rather
than circularising them (contrast Aniano et al., Sect. 4.4, who average over
2¹⁴ rotations to impose circular symmetry). In our runs the PSFs are supplied
in the detector frame and the mosaics are north-up, so α = 0 for all bands and
the step is a no-op. **If a future survey supplies mosaics at a non-zero
position angle, the corresponding `--angle_a` / `--angle_b` arguments must be
passed to PyPHER**, otherwise the anisotropic structure of the kernel will be
misaligned with the image. This is a latent requirement, not a current bug.

> **Other references for this section.** The `reproject` package (Robitaille et
> al.) for the interpolation, adaptive and exact algorithms. SIP distortion
> convention: Shupe et al. (2005, ADASS). Drizzle and correlated noise:
> Fruchter & Hook (2002).

---

## 7. Unit conversion (`units_2_0.py`)

### `convert2Jansky(fits_file, driver)`

Reads a pipeline-produced file (always HDU 0 at this stage) and dispatches to
the driver's `convert2Jansky`. The conversion is two-step by necessity:

```
surface_brightness = raw_value / native_pixel_area      # grid-independent
flux_per_pixel     = surface_brightness × actual_pixel_area
```

The reason: for HST, `PHOTFNU` converts electrons/s to Jy **assuming one native
pixel**. But by this stage the file has been reprojected onto another survey's
grid, so its pixels are no longer native. Multiplying `PHOTFNU` directly would
give "Jy per native pixel" mislabelled as the current pixel. Going through a
genuine surface brightness (using the *native* area) and back out (using the
*current* WCS area) is correct regardless of grid. This is the same two-step
form the surface-brightness surveys (S4G, JWST) use naturally.

> **Reference.** WFC3 Data Handbook (STScI) for `PHOTFNU` and the electrons/s →
> Jy conversion; the surface-brightness / flux distinction under resampling:
> Fruchter & Hook (2002).

---

## 8. Datacube assembly (`cube_2_0.py`, `mask_2_0.py`)

### `discover_jansky_files(...)`

Same header-first selection logic as `discover_convolved_files`, for the
Jy/pixel files. Parses the projection filenames (10 underscore-separated parts)
from the right.

### `create_data_cube(...)`

Assembles the final cube. Reads band names and order from the header
(`FILT001`…`FILTnnn`) so the cube is survey-combination-agnostic. Computes the
intersection footprint of all bands, crops to the smallest useful area *before*
any processing, performs sky subtraction band by band, builds a signal mask,
crops again by the mask with padding, and writes the 3D cube with a WCS and
`FILTnnn`/`PIXAREA` provenance in the header.

### `mask_2_0.py`

- `intersection_footprint_mask` — pixels valid in every band.
- `crop_to_mask_bbox` — bounding-box crop with padding.
- `sky_level` — robust background estimate per plane.
- `sum_images` — signal image for masking.
- `mask_after_sky_sub` — n-sigma signal mask.

Sky subtraction was verified correct: PHANGS products arrive already
background-subtracted (median ≈ 0), while IRAC had a positive pedestal
(zodiacal + instrumental) that had to be removed. The health signature is that
all bands converge to ~45–50% negative pixels after subtraction; more than 50%
would indicate over-subtraction.

> **Reference.** Sky estimation and aperture photometry: Howell, *Handbook of
> CCD Astronomy*. S4G background treatment: Muñoz-Mateos et al. (2015).

---

## 9. Preflight validation (`preflight.py`)

Validates the entire configuration *before* any processing, failing loudly on
anything that would produce an incomplete or incoherent cube, and warning on
legitimate scientific choices. Its existence embodies the second design
principle: check the assumption at the point of use.

Checks: band coverage (every science band has a PSF, a pivot wavelength and an
r80); driver completeness (no unimplemented abstract methods); convolution grid
per survey (master sampling and pair resolvability, via
`report_convolution_grid`); master-grid consistency (the master must end up on
its own survey's grid); and an estimate of the final cube size.

`--preflight_only` runs it and exits, so a new configuration can be validated
in seconds. It correctly distinguishes a hard error (a grid that undersamples
the master yet still tries to match pairs) from a benign warning (no pair
resolvable, so bands enter unmatched — the PHANGS-only case).

---

## 10. The CLI orchestrator (`astrovello_cli_2.0.py`)

`main()` runs the four stages. The supporting functions:

- `select_master(...)` — chooses the master by r80 and prints the ranking
  alongside the Gaussian FWHM, so the decision is visible in the log.
- `derive_bin_factors(...)` — derives the per-survey binning factor from the
  master (with `--bin_factor` override).
- `build_kernels(...)` — the per-survey kernel loop: master first (it fixes the
  array size for the others), then the source PSFs on the same grid and size,
  then the kernels; returns the unmatched bands.
- `rediscover_unmatched(...)` — when kernels already exist on disk (no
  `--create_kernel`), recovers which bands were left unmatched, so they do not
  silently vanish from the cube.
- `print_matching_summary(...)` — confirms every band took one of the three
  routes (matched, unmatched, master) and that they sum to the band count.

Guards worth noting: new kernels invalidate all convolved files on disk
(`force = force_convolution or create_kernel`); a band cannot be both matched
and unmatched (raises); missing kernels for the current master raise with a
clear message; unit-conversion failures are accumulated and raise rather than
letting a band disappear.

---

## 11. Configuration (`config.py`)

Holds only what cannot be derived from the data: per-survey file-naming
conventions, unit strings (`sci_unit`), reference PSF pixel scales,
`convolution_bin_factor` (fallback only — the CLI derives it), the science-HDU
index, and `PIVOT_WAVELENGTHS` per filter (used to order the cube by wavelength
and by the downstream RGB/plotting).

Pixel scales for the *science images* are **not** taken from here — they are
read from each file's WCS (`science_pixel_scale` in `utils_2_0.py`), because
the config constant is the PSF reference scale, which for S4G (1.221″) is not
the mosaic scale (0.75″). Conflating the two was the original unit bug.

---

## 12. Annotated bibliography

Verify all details on ADS before citing.

**PSF matching and convolution kernels**
- **Aniano, Draine, Gordon & Sandstrom (2011), PASP 123, 1218**
  *"Common-Resolution Convolution Kernels for Space- and Ground-Based
  Telescopes"* (arXiv:1106.5065). **Verified against the PDF.** The canonical
  reference for the convolution stage. Sections used in this document:
  - **Sect. 2** — kernel definition by Fourier division; existence condition
    (source PSF must be narrower than target); the low-pass filter f_A.
  - **Sect. 4.2** — resampling all PSFs to a common pixel grid.
  - **Sect. 4.3** — centring the PSF by centroid.
  - **Sect. 4.13** — resampling the kernel to the image grid.
  - **Sect. 4.14** — odd-sized array; trim retaining 99.9% of kernel energy.
  - **Sect. 5** — the D and W± performance metrics (our `inspect_kernel` and
    the r50 validation test).
  - **Sect. 7** — usage: kernel on the image grid, centred, unit-normalised,
    image in surface brightness units; the W⁻ ≲ 1.2 recommendation and the
    FWHM_B ≥ FWHM_A/1.35 restriction.
- **Boucaud, Bocchio, Abergel, Orieux, Dole & Hadj-Youcef (2016), A&A 596,
  A63** — *"Convolution kernels for multi-wavelength imaging"*; the PyPHER
  paper. **Verified against the PDF.** Sections used:
  - **Sect. 2.1** — the ill-posed inversion and the ℓ₂ Wiener regularisation;
    µ set to the inverse S/N of the homogenised image.
  - **Sect. 2.2, Algorithm 1** — the generation recipe, including the output
    grid convention `N = N_b ; p = p_b` and the PSF-warping (rotation,
    rescaling, padding) steps.

**Resampling, drizzle, sampling theory**
- **Fruchter & Hook (2002), PASP 114, 144** — Drizzle; area-exact resampling,
  flux vs surface-brightness conservation, correlated noise. Supports the
  binning-by-mean, the reprojection, and the convolve-then-reproject order.
- **Goodman, *Introduction to Fourier Optics*** — PSF, OTF, the sampling
  theorem, rigorously.
- **Bracewell, *The Fourier Transform and Its Applications*** — the
  delta ↔ sinc pair that explains kernel ringing; convolution theorem.

**Instrumentation and surveys**
- **Howell, *Handbook of CCD Astronomy*** — sampling, Nyquist, sky estimation,
  aperture photometry. Good for the fundamentals chapter.
- **IRAC Instrument Handbook (IPAC)** — the IRAC PRF, its oversampling and
  extended wings; confirms channel 2 wider than channel 1.
- **WFC3 Data Handbook (STScI)** — `PHOTFNU`, drizzled product scales.
- **Sheth et al. (2010, PASP)** — S4G survey.
- **Muñoz-Mateos et al. (2015, ApJS)** — S4G photometric pipeline and mosaics.
- **Lee et al. (2022, ApJS)** — PHANGS-HST.
- **Williams et al. (2024)** — PHANGS-JWST data release (verify authorship/year).
- **Anand et al. (2021, MNRAS)** — PHANGS distances (relevant for the
  distance-vs-redshift discussion in the SED-fitting stage).
- **Shupe et al. (2005, ADASS)** — the SIP distortion convention.

**SED fitting (downstream, for completeness)**
- **Conroy (2013, ARA&A 51, 393)** — SED modelling; the age–metallicity–dust
  degeneracies that make those parameters poorly constrained by broadband
  photometry.
- **Robotham et al. (~2020, MNRAS)** — ProSpect (verify year/authors).
- **Bruzual & Charlot (2003, MNRAS 344, 1000)** — BC03 stellar populations.
- **Dale et al. (2014, ApJ)** — dust re-emission templates.
