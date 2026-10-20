# AsTrovello — package adaptation

**Branch:** `v2-dev/package_adaptation` · **Date:** 2026-10-20
**Scope:** packaging and paths (Sects. 1–7), then three guard fixes in Check 2
found along the way (Sects. 8–10). No algorithm was touched. The cube is
bitwise unchanged (Sects. 3, 6, 8).

The goal: make the pipeline installable with `pip install`, and make it run
from **any directory on any machine**. Like v1.0, it should look for `Input/`
in the directory where the command is given and build `Output/` there. Before
this change, every entry point assumed this machine's layout: the CLI
used `Path.cwd().parents[1]`, so it had to be run from `src/astrovello`, and the
checks and forensic scripts hardcoded `~/Research/AsTrovello`.

The module renaming (`astrovello_cli_2.0.py` -> `astrovello_cli.py`,
`convolution_2_0.py` -> `convolution.py`, etc.) was done by the author before
this session. The changes below start from those new file names.

---

## 1. What changed

### 1.1 Project directory = current directory (the request)

| File | Before | After |
|---|---|---|
| `astrovello_cli.py` | `BASE_DIR = Path.cwd().parents[1]`, so it ran only from `src/astrovello` | `BASE_DIR = --root`, default `Path.cwd()`: the directory holding `Input/`. `Output/` is created there |
| `check2_psf_matching.py` | `INPUT_ROOT`, `REPO_ROOT` = `~/Research/AsTrovello`; `--input-root` | `--root`, default CWD; reads `<root>/Input`, writes `<root>/Output/checks/check2_psf_matching/` |
| `check3_flux_conservation.py` | `REPO_ROOT` = `~/Research/AsTrovello`; `--output-root` (but `CHECK_DIR` ignored it) | `--root`, default CWD; reads `<root>/Output`, creates `<root>/Output/checks/check3_flux_conservation/` |
| `diagnostico_etapa1.py`, `1b`, `1c` | `Path.cwd().parents[1]` | `Path.cwd()`; usage text now says `python -m astrovello.<script>` |
| `pypher_regularisation_test.py` | `~/Research/AsTrovello/...` | `Path.cwd() / "Input" / ...` and `Path.cwd() / "Output" / ...` |
| `test_sip_hypothesis.py` | `REPO_ROOT = ~/Research/AsTrovello` | `REPO_ROOT = Path.cwd()` (`--repo-root` kept) |

After the change, no `.py` file under `src/astrovello` contains `Research`, `/Users` or
`parents[1]` (grep).

The forensic scripts got only the path change. Their logic is history and
was left as is (CLAUDE.md Sect. 7: keep them runnable).

### 1.2 Imports

All pipeline modules used flat imports (`from config import ...`). Those work
only when the CWD is the module directory, and some still pointed at the
deleted `_2_0` names, so after the renaming the CLI did not import at all. They
are now package-relative:

```
astrovello_cli.py  from .config / .drivers / .convolution / .reprojection /
                   .units / .validation / .cube / .utils / .preflight
cube.py            from .mask
preflight.py       from .convolution / .drivers / .utils
units.py           from .drivers
utils.py           from .config
```

Docstrings that named old modules were updated too (`reprojection.py` header,
the `preflight.py` usage example).

**Consequence:** the modules can no longer be run as `python astrovello_cli.py`.
Use the `astrovello` command or `python -m astrovello`.

### 1.3 Package files

- **`__init__.py`**: the v1.x version imported `alignment`,
  `S4G2PHANGS_reproject`, `final_clean_psf`, ..., so `import astrovello`
  failed (E1). It is replaced by a docstring plus `__version__`, read from the
  installed metadata. It imports nothing else, so `import astrovello` and the
  CLI start-up stay cheap.
- **`__main__.py`** (new): `python -m astrovello` is the same as `astrovello`.
- **`pyproject.toml`**:
  - version `0.1.2` -> `2.0.0`;
  - description fixed (it read "galaxy morphology and non-parametric
    decomposition");
  - entry points: `astrovello`, `astrovello-check-psf`, `astrovello-check-flux`.
    `astrovello-batch` (-> `batch_runner`, which does not exist) was removed
    (E2);
  - dependencies cut to what the code imports: `numpy`, `astropy`, `scipy`,
    `pandas`, `photutils`, `reproject`, `pypher`. The packages removed were
    never imported in v2: `regions`, `scikit-image`, `tqdm`, `psutil`,
    `pyyaml`, `pyreadr`, `rasterio`, `shapely`, `dask`, `zarr`. `matplotlib`
    was kept at first for `pypher_regularisation_test.py`, then dropped when
    that script left the package (Sect. 7).
- **`.gitignore`**: added `build/` (setuptools writes it during a non-editable
  install).

### 1.4 Behaviour fixes that a package needs

- **PyPHER call without a shell.** `pypher_kernel_creation` returned
  `f"pypher {psf} {master} {kernel}"`, and the CLI ran it with
  `shell=True`. Any path with a space split into several arguments. That was
  harmless on this machine, but a package runs in the user's directories
  (`~/Google Drive/...`, `My Project/`). It now returns an argument list, run
  with `subprocess.run(cmd, check=True)` and printed with `shlex.join`.
  The docstring is updated.
- **Unconfigured directories in `Input/`.** Answering "Y" passed every
  subdirectory to `DRIVERS[survey]`, so a directory not named after a
  `SURVEY_CONFIG` key (e.g. `PHANGS` instead of `PHANGS-HST`) raised a bare
  `KeyError`. Now such directories are listed in a warning that also shows
  the expected names, and they are ignored. If none is configured, the run
  stops with an error.
- **Exit code.** A missing `Input/` used to print an error and `return`
  (exit 0). It now returns 1 and says to run from the directory holding
  `Input/` or to use `--root`. `__main__` uses `raise SystemExit(main())`.

### 1.5 Documentation

- **`README.md`**: rewritten for v2. It still described v1.x (reprojecting S4G
  onto the HST grid, no JWST, a Quick Start with functions that no longer
  exist; E7). It now covers what the pipeline does, installation, the project
  directory layout with real file patterns from `config.py`, modes, outputs,
  standalone checks, data sources and the module list.
- **`astrovello_pipeline_documentation.md`**: `_2_0` file names and their TOC
  anchors updated. A new paragraph opens Sect. 10 on the project directory,
  `--root`, the survey-directory naming rule and the check commands.
- **`CLAUDE.md`**: Sect. 1 rewritten (how to run as a package), module map
  renamed with a note that Sect. 4 cites the old names, status note on E1–E3
  and E7, and the packaging row of Sect. 5 marked done.

---

## 2. How to run now

```bash
conda activate capivara
cd ~/Research/AsTrovello            # once: pip install -e .
astrovello --galaxy ngc1087 --surveys PHANGS-HST,PHANGS-JWST --create_kernel
astrovello-check-psf
astrovello-check-flux --galaxy ngc1087
python scripts/forensics/diagnostico_etapa1c.py   # forensic scripts (Sect. 7)
```

Any directory works, as long as it holds `Input/<SURVEY>/galaxies/<galaxy>/`
and `Input/<SURVEY>/PSF/`, with survey names matching `SURVEY_CONFIG`.
`--root /path` runs on a project directory without `cd`.

**The `capivara` environment does not have the package installed yet.** All
tests below used a throwaway environment. Run `pip install -e .` from the repo
root once.

---

## 3. Verification

Test environment: a venv created from `capivara`'s Python 3.11.15 with
`--system-site-packages`. The package was installed **non-editable** with
`pip install --no-deps --no-build-isolation`, so the tests used the installed
copy, not the source tree. `capivara` was not modified.

| Test | Result |
|---|---|
| Import every module from the installed copy (19, via `pkgutil`) | all import |
| Console scripts created | `astrovello`, `astrovello-check-psf`, `astrovello-check-flux` |
| `astrovello` in an empty directory | clear error, exit 1 |
| `--preflight_only` in a new directory **with a space in its name**, `Input` symlinked | 3 surveys found, preflight complete, nothing written |
| `astrovello-check-psf` from that directory | PyPHER ran on paths with spaces; HST 5/5 PASS (r50 1.0000–1.0001) |
| `astrovello-check-flux --root <repo>` from another directory | ran; 14 bands, PASS |
| **Full run from scratch** (below) | exit 0, cube bitwise identical to reference |

**Full run.** A new directory, `fresh project/` (with a space), received copies
of only `Input/<SURVEY>/galaxies/ngc1087` and `Input/<SURVEY>/PSF` for the three
surveys. It had no `PSF_CLEAN` and no `Output`. The run was
`astrovello --galaxy ngc1087 --create_kernel`, selecting PHANGS-HST + PHANGS-JWST
(master `f2100w`, the dissertation configuration):

- `Input/*/PSF_CLEAN`, `Output/PSF_Kernels`, `convolved_fits/ngc1087`,
  `reprojected_files/ngc1087` and `datacubes/ngc1087` were all created inside
  the new directory;
- 12 PyPHER kernels built on the per-band grids (0.1981, 0.0307, 0.0630,
  0.1109"/px); in-pipeline Check 2: 12/12 PASS;
- in-pipeline Check 3: 12 bands, worst deviation 0.01%;
- cube `ngc1087_datacube_sci_1109x1202_Jy_per_pixel.fits`, compared with
  `Output/datacubes/ngc1087/` of the same name (the valid 2026-09-30 cube):

```
shape                  (13, 1202, 1109)  both
NaN pattern identical  True
max |diff|             0.0
bitwise equal          True
header keys differing  none   (DATE, HISTORY, COMMENT excluded from the comparison)
```

**Triage (CLAUDE.md Sect. 0):** no number in any cube changes. All changes in
this report are packaging. The HST+JWST stopping rule is unaffected.

### Not covered by these tests

- **Dependency resolution from PyPI.** The install used `--no-deps` on top of
  `capivara`'s packages. That the reduced list is enough comes from the grep of
  third-party imports, not from a clean `pip install` in an empty environment.
- **S4G in a full run.** S4G went through preflight only. Its full run would
  carry C4, which is unrelated to this change.
- **The two notes in the test output come from existing state, not from this
  change.**
  - `astrovello-check-psf` printed "no usable PSF_CLEAN" for PHANGS-JWST. This
    is the known B8 leftover: the standalone check expects one `master_*.fits`.
  - `astrovello-check-flux` on the repo's `Output/` reported 14 bands, with
    irac1 at 0.84%. That directory holds intermediates of more than one
    configuration (CLAUDE.md Sect. 1), and 0.84% is C4.

---

## 4. Open items and decisions for the author

None of these changes a cube.

1. ~~**Interactive survey prompt.**~~ Done: `--surveys`, see Sect. 6.
2. **`PSF_CLEAN` is written inside `Input/`.** The pipeline modifies the user's
   input directory. A package would usually treat `Input/` as read-only and
   write the cleaned PSFs under `Output/`. **Decision (author, 2026-10-20):
   keep it as it is.**
3. ~~**Forensic scripts ship inside the package.**~~ Done: moved to
   `scripts/forensics/`, notebooks to `notebooks/` (Sect. 7).
4. **GitHub install.** The README says
   `pip install git+https://github.com/andretrovello/AsTrovello.git`. That
   installs `main`, which is still v1.0, so it gives v2 only after this branch
   is merged.
5. **Still open from Sect. 4E:** E5 (identify the survey from
   `TELESCOP`/`INSTRUME` rather than the directory name), E6 (`tests/` has no
   tests; `tests/classes.py` is a stale copy of `drivers.py`), E8 (`--error`
   and `--valid_pixels_cut` declared but unimplemented).
6. **`pypher` on `PATH`.** The kernel stage calls the `pypher` executable, so it
   works only with the environment activated. Calling the venv's binary
   directly without activating would fail at that stage.
7. **Found in the author's own run (2026-10-20, `~/Research/astrovello_test`;
   cube bitwise identical to the reference).** The log reports
   "PHANGS-JWST: 5 distinct grids" for 3, because grid keys use 6 decimals and
   file names 4. Cosmetic, pre-existing since B8; recorded in CLAUDE.md
   Sect. 4c.

---

## 5. Files touched

```
M  pyproject.toml
M  .gitignore
M  README.md
M  CLAUDE.md
M  astrovello_pipeline_documentation.md
M  src/astrovello/__init__.py
A  src/astrovello/__main__.py
M  src/astrovello/astrovello_cli.py          (renamed by the author; edited here)
M  src/astrovello/convolution.py             (renamed by the author; edited here)
M  src/astrovello/cube.py                    (renamed by the author; import only)
M  src/astrovello/reprojection.py            (renamed by the author; docstring only)
M  src/astrovello/units.py                   (renamed by the author; import only)
M  src/astrovello/utils.py                   (renamed by the author; import only)
M  src/astrovello/preflight.py
M  src/astrovello/check2_psf_matching.py
M  src/astrovello/check3_flux_conservation.py
M  src/astrovello/validation.py              (Check 2, Sects. 8 and 10)
R  src/astrovello/{diagnostico_astrometria,diagnostico_etapa1,diagnostico_etapa1b,
   diagnostico_etapa1c,pypher_regularisation_test,test_sip_hypothesis}.py
   -> scripts/forensics/                     (paths edited; moved in Sect. 7)
R  src/astrovello/*.ipynb (4) -> notebooks/  (jwst_file_analysis.ipynb: root path)
```

`mask.py`, `drivers.py` and `config.py` were not edited. Nothing has been
committed.

---

## 6. Follow-up: `--surveys` (same day)

**Problem.** The CLI asked which surveys to combine with `input()`. This had
three consequences:

- batch runs stopped at the prompt;
- runs without a terminal (`nohup`, cluster, cron) died with `EOFError`;
- the choice that sets the master, and so the whole cube, was not part of the
  command line, so it could not be reproduced from the shell history.

The interactive "N" branch also kept duplicates (`PHANGS-HST, PHANGS-HST`
would gather the files twice).

**Change (`astrovello_cli.py`).** A new option, `--surveys`, takes a
comma-separated list or `all`. The prompt moved into two functions:

- `resolve_surveys(text, available)` parses the text: case and spaces are
  normalised, duplicates removed, and the result follows the order of
  `Input/`, so the same choice always gives the same list. It returns
  `(selected, missing)`.
- `select_surveys(available, requested)` decides what happens:
  - with `--surveys`, a name that is not available, or an empty list, is
    **fatal** (exit 1, with the available names). Dropping a mistyped survey
    would silently change the master;
  - without `--surveys` and without a terminal, the run stops with a message
    pointing to `--surveys`, not an `EOFError`;
  - without `--surveys` and with a terminal, it asks as before. An invalid
    answer asks again.

The selection is logged in every case: `>>> Surveys: [...] (from --surveys)`
or `(chosen interactively)`.

**`all` vs an explicit list.** `all` is accepted, and the list it resolved to
is logged. For the dissertation cube, prefer the explicit list: `all` changes
meaning if a directory is added to `Input/`.

**Verification** (installed copy, `--preflight_only` unless stated):

| Case | Result |
|---|---|
| parsing: list, lowercase with spaces, duplicate, `all`/`ALL`, typo, `all,S4G`, empty, `" , "` | as specified (typo, `all,S4G` and empty lists rejected) |
| `--surveys PHANGS-HST,PHANGS-JWTS` | error naming `PHANGS-JWTS` and listing the available ones, exit 1 |
| `--surveys ""` | error, exit 1 |
| no flag, `stdin` from `/dev/null` | "no terminal ... Pass them with --surveys", exit 1 |
| `--surveys all` | `['PHANGS-HST', 'PHANGS-JWST', 'S4G'] (from --surveys)`, exit 0 |
| `--surveys "phangs-jwst, phangs-hst"` | `['PHANGS-HST', 'PHANGS-JWST']` |
| interactive under a real pty (`expect`): `x`, `N`, `S4X`, `phangs-hst, phangs-jwst` | re-asks on `x`, errors and re-asks on `S4X`, accepts the last answer, exit 0 |
| **full run without a terminal**: fresh project directory (its `Output/` and `PSF_CLEAN` removed first), `--surveys PHANGS-HST,PHANGS-JWST --create_kernel < /dev/null` | exit 0; Check 3 worst deviation 0.01%; cube **bitwise identical** to the 2026-09-30 reference (same shape, NaN pattern, max diff 0.0, no header key differing) |

**Behaviour change to note.** Piping answers into the prompt (`printf 'N\n...'
| astrovello ...`) no longer works, because `stdin` is not a terminal. Use
`--surveys` instead.

Docs updated: README (usage and a batch loop), methodology doc Sect. 10,
CLAUDE.md Sect. 1. Triage: hardening/interface; no number in any cube changes.

---

## 7. Follow-up: forensic scripts and notebooks out of the package (same day)

**Problem.** The wheel shipped six development scripts as `astrovello.*`
modules:

- `diagnostico_astrometria.py` (stub);
- `diagnostico_etapa1.py`, `1b`, `1c`;
- `pypher_regularisation_test.py`;
- `test_sip_hypothesis.py`.

They answer questions from the author's own investigation (NGC 2903 default
coordinates, a hardcoded master name, the v1.x `Input/PHANGS` layout). For
another user they fail or mislead. Two of them also match pytest's default
patterns (`test_*.py`, `*_test.py`) and have import-time side effects:
`test_sip_hypothesis.py` sets a **global** `warnings.filterwarnings('ignore')`,
and `pypher_regularisation_test.py` imports `matplotlib.pyplot`. No pipeline
module imports any of them. The only reference was a comment in
`reprojection.py`.

**Change.** All moves were made with `git mv`, so history is kept
(`git log --follow`). File names are unchanged, so references by name in
CLAUDE.md, the dated reports and the methodology doc stay valid.

- `src/astrovello/{those six}` -> `scripts/forensics/`, outside `src/`, so they
  are not installed.
- `src/astrovello/*.ipynb` (4) -> `notebooks/`. They were never in the wheel;
  the move tidies the source package. `jwst_file_analysis.ipynb` used
  `Path.cwd().parents[1]` to reach the repo root from `src/astrovello`. From
  `notebooks/` that is `Path.cwd().parent`, and the line was changed
  accordingly. The other three notebooks have no local imports or relative
  paths.
- `test_sip_hypothesis.py`: `sys.path.insert(...)` + `from validation import`
  -> `from astrovello.validation import`, which needs the package installed.
- Usage text in the scripts: `python -m astrovello.<x>` ->
  `python scripts/forensics/<x>.py`, run from the project directory (they use
  `Path.cwd()`, Sect. 1.1).
- `reprojection.py:206` comment now gives the new path.
- `pyproject.toml`: `[tool.pytest.ini_options] testpaths = ["tests"]`;
  `matplotlib` dropped from the dependencies (nothing in `src/` imports it).
- `tests/` was not touched. It still holds two notebooks and the stale
  `classes.py` (E6).

**Verification.**

| Test | Result |
|---|---|
| wheel contents | 15 modules: `__init__`, `__main__`, the pipeline modules, `check2`, `check3`; no forensic script |
| `py_compile` of the six scripts | ok |
| `--help` of `diagnostico_etapa1`, `1b`, `1c`, `test_sip_hypothesis` from `scripts/forensics/` | ok |
| `from astrovello.validation import check_flux_conservation` (installed copy) | ok |
| `diagnostico_etapa1c.py` run for real from the repo root | runs; reports `PSF_CLEAN ausente em: ['PHANGS']`, which is its own v1.x layout, not the move |
| pytest `--collect-only` with `testpaths` | nothing imported from `scripts/` (0.00 s) |
| same, without `testpaths` | 0 tests (the scripts define no `test_*` functions), but both are imported (0.81 s): the side effects `testpaths` avoids |

Triage: packaging, no cube affected.

---

## 8. Follow-up: Check 2 no longer passes over kernels it cannot check (same day)

**Problem.** `validation.check_psf_matching` must find, for each kernel, the
source PSF it was built from. Nothing records that link, so the gate finds the
PSF **by name**: among the `PSF_CLEAN` files on the kernel's grid, it looks for
the one whose name contains the filter. When that failed, the band was
**skipped** and the gate went on. It raised only if *no* band at all could be
checked. There were three ways to fail:

- the kernel name could not be parsed;
- zero candidate PSFs (a PSF not named after its filter);
- several candidates, e.g. one F814W PSF per camera (WFC3/UVIS and ACS/WFC).
  `PSFSTD` matches both. `build_kernels` then keeps whichever comes last
  (`cleaned[filt] = out_psf`), so the kernel is built from an arbitrary PSF.

Reproduced with the real HST kernels and PSFs of the clean run plus a second
F814W PSF:

```
         f814w: 2 candidate source PSFs, skipped
      -> 4 pair(s) checked, all within criterion      (no error; run continues)
```

The criterion itself (r50 ratio 0.95–1.05) was always general. What was not
general was finding the PSF to test, which depends on file names that a user
of the package controls. This is the Invariant 8 pattern ("an unverifiable
guard is a hard error, not a fallback"). None of the author's runs was
affected: they always had 12/12 bands checked.

**Change (`validation.py`, `check_psf_matching` only).** Each such kernel is
recorded with its reason and printed as `NOT CHECKED`. After the loop, if any
exist (and `raise_on_fail`), the gate raises `ValidationError` listing each
one, with the candidate files, and says to keep one PSF per filter and re-run
with `--create_kernel` (which rebuilds `PSF_CLEAN` from scratch). The
criterion, the measurements and the output for checkable kernels are
unchanged. `raise_on_fail=False` (reporting only) still returns the results
without raising.

**Verification.**

| Test | Result |
|---|---|
| (a) two F814W PSFs, HST kernels of the clean run | `ValidationError`: `f814w: 2 candidate source PSFs (0.1981_PSFSTD_ACSWFC_F814W.fits, 0.1981_PSFSTD_WFC3UV_F814W.fits)` |
| (b) F555W PSF renamed so it no longer contains the filter | `ValidationError`: `f555w: 0 candidate source PSFs (none)` |
| (c) kernel named `f275w_kernel.fits` | `ValidationError`: name does not follow `kernel_<source>_to_<master>.fits` |
| (d) the four real grids of the clean HST+JWST run (0.1981, 0.0307, 0.0630, 0.1109) | 5 + 1 + 3 + 3 = 12 checked, same ratios as before (0.9999–1.0001) |
| **end to end**: new project, HST+JWST, `PSFSTD_ACSWFC_F814W.fits` added to `Input/PHANGS-HST/PSF`, `astrovello --surveys PHANGS-HST,PHANGS-JWST --create_kernel` | **exit 1** at Check 2; message as in (a); **0 convolved files** written |
| same project after removing the extra PSF, as the message says | exit 0; Check 2 12/12; Check 3 worst 0.01%; cube **bitwise identical** to the 2026-09-30 reference |

A first end-to-end attempt with HST only was not a test of this change. With
HST alone `f814w` becomes the master, and no HST pair needs a kernel, so Check 2
has nothing to check. It is recorded here so the result is not misread.

**Same pattern one level up:** the grid-level skip in the CLI. Fixed
next, in Sect. 9.

The methodology doc does not describe the in-pipeline gates at all (Check 2
or Check 3). It should, before the dissertation chapter cites them.

Triage: hardening; no cube affected.

---

## 9. Follow-up: the CLI no longer skips Check 2 for a whole grid (same day)

**Problem.** Before calling `check_psf_matching`, the CLI finds, for each survey
and grid, the reference the kernels are compared against: the cleaned master
PSF `PSF_CLEAN/master_<grid>_*.fits`. It skipped the gate in two places:

| Situation | Before |
|---|---|
| `Input/<survey>/PSF_CLEAN` absent | survey skipped, **no message** |
| grid without exactly one `master_<grid>_*` (0 or 2+) | `==> CHECK 2 skipped for ...`, all kernels of that grid skipped |

In both cases the run went on and convolved with those kernels unverified.
This was the B8 leftover in CLAUDE.md Sect. 4c, the same pattern as Sect. 8
for a whole grid or survey.

**Reachability.** It cannot happen with `--create_kernel`: `build_kernels`
deletes and rebuilds `PSF_CLEAN` and writes exactly one master per grid,
right before the gate. It can happen only when kernels are reused. That is
the case where Check 2 is there to catch stale kernels, and `PSF_CLEAN` may
have been deleted (it sits in `Input/` and looks like an intermediate) or not
copied along with `Output/`. The grid guard and the kernel sign check in
`create_convolvedFITS` stay active either way. What was lost is the test that
the kernel reproduces the master.

**Change (`astrovello_cli.py`, the Check 2 block).**

- The `if not psf_clean_dir.is_dir(): continue` was removed. A missing
  directory now falls into the master lookup, which finds nothing.
- A count other than one raises `ValidationError` with the survey, grid,
  directory, the masters found, and the number of kernels that would go
  unverified. It says to re-run with `--create_kernel`, or to use
  `--skip_checks` to accept the risk.
- `if not survey_kernels: continue` stays: a survey without kernels (S4G,
  where irac1 enters unmatched) has nothing to check, and still needs no
  `PSF_CLEAN`.

**Verification** (new project from Sect. 8: HST+JWST, kernels and `PSF_CLEAN`
from a complete run; `--mode conv_only` **without** `--create_kernel`, so
kernels are reused and convolved files already exist):

| Case | Result |
|---|---|
| `PSF_CLEAN` intact | exit 0; Check 2 runs on all four grids (5 + 1 + 3 + 3) |
| HST `PSF_CLEAN` moved away | **exit 1**: `Check 2 cannot run for PHANGS-HST on the 0.1981 arcsec/px grid: ... found 0 (none)` (previously: silent skip) |
| same, `--skip_checks` | exit 0, `==> CHECK 2 SKIPPED by --skip_checks` |
| a second `master_0.1109_*` file in the JWST `PSF_CLEAN` | **exit 1** after the three other grids pass: `... found 2 (master_0.1109_PSF_MIRI_in_flight_opd_filter_F2100W.fits, master_0.1109_PSF_old_run_F2100W.fits). Its 3 kernel(s) would be applied unverified. ...` |
| restored | exit 0, all four grids checked |

No full `--create_kernel` run was repeated for this change. On that path the
block only replaced a `print` + `continue` with a `raise`, and neither branch
is reached when exactly one master exists. Case 1 shows the gate running
unchanged on all four grids.

Triage: hardening; no cube affected.

---

## 10. Follow-up: dead fallback removed from Check 2 (same day)

**What it was.** `check_psf_matching` looks for the band's source PSF among
the `PSF_CLEAN` files named `<grid>_*` that contain the filter. If none was
found, it searched again in **every** file, of any grid and with or without
the prefix (`# PSFs written before the per-grid naming`). This was a
transition path for `PSF_CLEAN` directories written before B8 (2026-09-30,
commit `71251e9`).

**Why it no longer served.** Such a directory cannot reach the gate any more.
`--create_kernel` deletes and rewrites `PSF_CLEAN` in the per-grid naming.
When kernels are reused, the CLI calls the gate only after finding
`master_<grid>_*.fits` (per-grid naming), and since Sect. 9 it stops if it
does not. So the fallback ran only when the per-grid search had already
failed:

- in practice it then found nothing either, because each band is cleaned on
  one grid only. The outcome was the same as without it;
- in an artificial case (a PSF with the same filter but another grid's prefix
  put there by hand) it found that file, and the grid check below then
  raised with a misleading message pointing at the v1.x grid bug ("blur by a
  factor ~12 too little"), when the real problem was a missing PSF.

**Change.** The four lines were removed (`validation.py`). The rule is now
the single one stated in the comment above it.

**Verification.**

| Case | Result |
|---|---|
| two F814W PSFs | `f814w: 2 candidate source PSFs (...)`, unchanged |
| F555W present only with another grid's prefix (`0.0630_...F555W.fits`) | `f555w: 0 candidate source PSFs (none)`, the correct message |
| unparseable kernel name | unchanged |
| clean HST set | PASS, 5 bands |
| the four real grids of the clean run | 12 checked, all PASS |
| CLI reuse run (`--mode conv_only`, HST+JWST) | exit 0, Check 2 on 5 + 1 + 3 + 3 |

Triage: cosmetic (CLAUDE.md listed it as hardening, but it had no effect on
any reachable run); no cube affected.

---

## 11. Follow-up: unimplemented flags removed; validation gates documented (same day)

**`--error` and `--valid_pixels_cut` (E8).** Both were declared in the CLI and
listed by `astrovello --help`, but nothing read them. A user passing `--error`
would have believed an error cube was produced. They are removed. Passing
either is now an argparse error (`unrecognized arguments: --error`, exit 2).
Each should be re-added together with its implementation. Verified on the
installed copy: absent from `--help`, `--error` rejected, a normal
`--preflight_only` run unchanged.

**Methodology doc, Section 9.** Until now the doc did not describe the
in-pipeline gates at all. Section 9 is now "Validation: preflight and
in-pipeline gates". The number is kept, so references to the doc's Sect. 9
stay valid, and the TOC anchor is updated. It covers:

- the gate design rules: fixed criteria, measuring before transforming, and
  "cannot check = fail";
- Check 2: when it runs, the r50 criterion, the hard failures (including the
  rules from Sects. 8–9 of this report), negative power, the 12-band
  measurements of the clean run, the by-construction limitation, and its
  relation to the ~1.06 of `diagnostico_etapa1c.py`;
- Check 3: which bands, the area-weighted sky-aperture method, the 12-band
  measurements, and two limitations (it still skips, and passing 1% does not
  rule out a smaller systematic, as IRAC1 at 0.84% showed, C4);
- the two standalone commands and how they differ from the gates.

A pointer to Check 2 was added to the Section 4 foreword, and the location of
`diagnostico_etapa1c.py` updated there.

One existing false claim in the rewritten paragraph was corrected: the
preflight does **not** check "master-grid consistency"; `_check_master_grid`
only prints a reminder (CLAUDE.md Sect. D).

**New finding while writing it.** Check 3 has the same skip pattern removed
from Check 2 (bands reported "skipped" without failing; the gate passes if
none can be measured). Fixed next, in Sect. 12.

---

## 12. Follow-up: Check 3 no longer passes over bands it cannot measure (same day)

**Problem.** `check_flux_conservation` skipped in five places and the run went
on:

- a band with no usable source;
- a band with no aperture complete on both grids;
- a band with no aperture with enough signal;
- no band measured at all ("nothing could be measured", gate passed);
- `photutils` not importable (gate skipped).

The CLI also did not call the gate when the list of reprojected pairs was
empty.

**Why not only flip the skips to failures.** The apertures were the 8
brightest peaks of the image *before* reprojection. Their coverage on the
master's grid was checked only afterwards, and that grid usually covers less
sky. A band whose brightest peaks lie outside the master's footprint would
then fail for a reason created by the check, not by the data. The only way
out would have been `--skip_checks`, which also disables Check 2. This
happens in the author's own data: in NGC 1087, one of the 8 brightest peaks
of each MIRI band falls partly outside the master's grid.

**Change (`validation.py`, `check_flux_conservation`; one line in the CLI).**

1. Five times as many candidate peaks are taken. The 8 brightest whose
   aperture is complete on **both** grids are kept, and the 5% signal cut is
   applied to those. When the 8 brightest are all complete, the selection is
   identical to before (the greedy, brightness-ordered peak picking yields the
   same prefix).
2. A band with no source, no aperture inside both grids, or no signal is
   recorded and printed as `NOT CHECKED`. If any exists, the gate raises
   `ValidationError` listing the bands and reasons.
3. Empty `pairs` raises ("nothing to measure"), and so does a missing
   `photutils`. With `raise_on_fail=False` (reporting, used by
   `scripts/forensics/test_sip_hypothesis.py`) nothing raises, as before.
4. CLI: `if not args.skip_checks and reprojection_pairs:` became
   `if not args.skip_checks:`, so an empty list reaches the gate and fails.

**Verification.**

The 12 real bands of the clean HST+JWST run, before and after the change:

| Bands | Before | After |
|---|---|---|
| HST ×5, F200W, F335M, F360M | 8 apertures, values as logged | **identical** |
| F300M | 7 (8th fails the signal cut) | 7, identical |
| F770W, F1000W, F1130W | 7 apertures, ratio 1.0000 | 8 apertures, ratio 0.9999 (+0.01%) |

Worst deviation still 0.01%. All PASS.

Failure cases, built from the real F555W pair:

| Case | Result |
|---|---|
| (a) image after reprojection shifted 1° on the sky (no overlap) | `ValidationError`: `f555w: no aperture lies entirely inside both grids` |
| (b) band image all NaN | `ValidationError`: `f555w: no usable source found` |
| (c) empty list of pairs | `ValidationError`: nothing to measure |
| (d) one good band plus one with no overlap | `ValidationError` naming the bad band. **Before, the gate passed on the good band alone** |
| (e) same as (d), `raise_on_fail=False` | returns, 1 band measured |
| (f) `photutils` not importable | `ValidationError`: cannot run |

End to end: in the clean project, `reprojected_files/` and `datacubes/` were
deleted and `astrovello --surveys PHANGS-HST,PHANGS-JWST` was run (kernels and
convolved files reused). Result: exit 0, Check 3 12/12 with the values above,
cube **bitwise identical** to the 2026-09-30 reference. Check 3 only reads the
files, so the cube could not change, and this confirms it.

Methodology doc Sect. 9 updated: the design rule now holds for both gates,
the aperture selection is described, the MIRI 7 -> 8 change is explained, a
hard-failure table was added, and the limitation about skips was removed.
CLAUDE.md Sect. 4c entry marked fixed.

Triage: hardening; no cube affected.
