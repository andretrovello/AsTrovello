"""
diagnostico_etapa1.py — AsTrovello 2.0

Diagnostico SOMENTE LEITURA. Nao escreve, nao altera, nao reprocessa nada.
Responde a tres perguntas antes de qualquer correcao:

  1. O master (irac1) e realmente a PSF mais larga? Se nao, o kernel
     irac2 -> irac1 e uma DECONVOLUCAO.
  2. O header do S4G tem coeficientes SIP de verdade, ou o codigo esta
     forcando uma distorcao que nao existe?
  3. As bandas PHANGS e IRAC estao alinhadas depois da reprojecao?

COMO RODAR
----------
Salve em  /Users/andretrovello/Research/AsTrovello/src/astrovello/
(o mesmo diretorio de onde voce roda o astrovello_cli_2.0.py) e execute:

    conda activate capivara
    python diagnostico_etapa1.py --galaxy ngc2903

Sem dependencias novas: numpy + astropy apenas.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

SEP = "=" * 78


# ----------------------------------------------------------------------------
# Utilitarios
# ----------------------------------------------------------------------------
def load_2d(path):
    """Le a primeira extensao com dados; colapsa cubo 3D pela media."""
    with fits.open(path, ignore_missing_end=True, ignore_missing_simple=True) as hdu:
        data = next((h.data for h in hdu if h.data is not None), None)
    if data is None:
        raise ValueError(f"Sem dados em {path}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
        data = np.nanmean(data, axis=0)
    return data


def half_light_radius(data, pixscale, verbose=False):
    """
    Raio que contem 50% do fluxo, em arcsec. Independente de forma de perfil,
    ao contrario de um ajuste gaussiano.

    Subtrai um fundo robusto (mediana do anel externo) antes de acumular:
    sem isso a soma cumulativa cresce com a area e o "50%" perde sentido.
    Limita o raio a min(shape)/2 para nao entrar nos cantos.
    """
    d = np.nan_to_num(data, nan=0.0)
    cy, cx = np.unravel_index(np.argmax(d), d.shape)

    y, x = np.indices(d.shape)
    r = np.sqrt((x - cx) ** 2.0 + (y - cy) ** 2.0)
    r_max = min(d.shape) / 2.0

    outer = (r > 0.85 * r_max) & (r <= r_max)
    background = np.median(d[outer]) if outer.sum() > 20 else 0.0
    d = d - background

    inside = r <= r_max
    r_flat = r[inside].ravel()
    v_flat = d[inside].ravel()

    order = np.argsort(r_flat)
    cum = np.cumsum(v_flat[order])
    total = cum[-1]
    if total <= 0:
        return np.nan
    cum = cum / total

    r_half_px = float(np.interp(0.5, cum, r_flat[order]))

    if verbose:
        print(f"      centro=({cy},{cx})  fundo={background:.3e}  "
              f"r_max={r_max:.0f} px  r50={r_half_px:.2f} px")

    return r_half_px * pixscale


def centroid_com(sub):
    """Centro de massa de um recorte 2D. Retorna (dy, dx) em pixels do recorte."""
    s = np.nan_to_num(sub, nan=0.0)
    s = s - np.median(s)
    s[s < 0] = 0.0
    tot = s.sum()
    if tot <= 0:
        return np.nan, np.nan
    y, x = np.indices(s.shape)
    return float((s * y).sum() / tot), float((s * x).sum() / tot)


def nucleus_centroid(path, box=12):
    """Centroide do pixel mais brilhante, refinado por centro de massa."""
    d = load_2d(path)
    if not np.isfinite(d).any():
        return np.nan, np.nan
    cy, cx = np.unravel_index(np.nanargmax(d), d.shape)
    y0, y1 = max(0, cy - box), min(d.shape[0], cy + box + 1)
    x0, x1 = max(0, cx - box), min(d.shape[1], cx + box + 1)
    dy, dx = centroid_com(d[y0:y1, x0:x1])
    return y0 + dy, x0 + dx


def find_one(pattern_dir, pattern, label):
    hits = sorted(Path(pattern_dir).glob(pattern))
    if not hits:
        print(f"   [!] nao encontrei {label}: {pattern_dir}/{pattern}")
        return None
    if len(hits) > 1:
        print(f"   [!] varios candidatos para {label}, usando {hits[0].name}")
    return hits[0]


# ----------------------------------------------------------------------------
# Teste 1 — ordenacao das PRFs do IRAC
# ----------------------------------------------------------------------------
def teste_1_prf(input_dir, verbose):
    print(SEP)
    print("TESTE 1 — o master e realmente a PSF mais larga? (P1.5)")
    print(SEP)

    psf_dir = input_dir / "S4G" / "PSF"
    native = {"irac1": 1.221, "irac2": 1.223}
    binned_factor = 5  # PRF do IRAC vem 5x superamostrada

    results = {}
    for filt, nat in native.items():
        f = find_one(psf_dir, f"*{filt.upper()}*col129_row129.fits", f"PRF {filt}")
        if f is None:
            continue
        pixscale = nat / binned_factor
        if verbose:
            print(f"   {filt}: {f.name}  (pixscale {pixscale:.4f} arcsec/px)")
        results[filt] = half_light_radius(load_2d(f), pixscale, verbose=verbose)

    if len(results) < 2:
        print("\n   Nao consegui medir as duas PRFs. Confira os nomes em Input/S4G/PSF.")
        return None

    print("\n   raio de meia-luz (arcsec):")
    for filt in ("irac1", "irac2"):
        print(f"      {filt}: {results[filt]:.4f}")

    mais_largo = max(results, key=results.get)
    diff_pct = 100.0 * abs(results["irac2"] - results["irac1"]) / results["irac1"]

    print(f"\n   PSF mais larga por este criterio: {mais_largo}  "
          f"(diferenca {diff_pct:.1f}%)")
    print(f"   Master escolhido pela pipeline (ajuste gaussiano): irac1")

    if mais_largo == "irac1":
        print("\n   ==> OK. O master esta correto. Nenhuma deconvolucao.")
        print("       Pode passar direto para o P0.2.")
    else:
        print("\n   ==> PROBLEMA CONFIRMADO. O irac2 e mais largo que o irac1.")
        print("       O kernel irac2 -> irac1 esta DECONVOLUINDO o irac2:")
        print("       ringing e ruido amplificado no plano irac2 do cubo,")
        print("       concentrado perto do nucleo. Erro de COR, numa banda so.")
        print("       Acao: trocar o master para irac2 e reconvoluir tudo.")

    if diff_pct < 3.0:
        print("\n   [nota] a diferenca e pequena (<3%). O criterio fica sensivel")
        print("          a ruido da PRF; confira o r50 em px no modo --verbose")
        print("          antes de reprocessar 7 bandas por causa disso.")

    return mais_largo


# ----------------------------------------------------------------------------
# Teste 2 — coeficientes SIP no header do S4G
# ----------------------------------------------------------------------------
def teste_2_sip(input_dir, galaxy):
    print()
    print(SEP)
    print("TESTE 2 — o SIP forcado no S4G corresponde a algo real? (P1.6a)")
    print(SEP)

    gal_dir = input_dir / "S4G" / "galaxies" / galaxy
    f = find_one(gal_dir, "*.phot.1.fits", "mosaico S4G ch1")
    if f is None:
        return None

    h = fits.getheader(f)

    fwd = sorted(k for k in h if k.startswith(("A_", "B_"))
                 and k not in ("A_ORDER", "B_ORDER"))
    inv = sorted(k for k in h if k.startswith(("AP_", "BP_"))
                 and k not in ("AP_ORDER", "BP_ORDER"))

    print(f"   arquivo: {f.name}")
    print(f"   CTYPE1 original: {h.get('CTYPE1')}")
    print(f"   CTYPE2 original: {h.get('CTYPE2')}")
    print(f"   A_ORDER/B_ORDER: {h.get('A_ORDER')} / {h.get('B_ORDER')}")
    print(f"   coef. SIP diretos  (A_i_j, B_i_j):  {len(fwd)}")
    print(f"   coef. SIP inversos (AP_i_j, BP_i_j): {len(inv)}")

    px = np.sqrt(proj_plane_pixel_area(WCS(h)) * 3600 ** 2)
    print(f"   escala do mosaico: {px:.4f} arcsec/px")

    if not fwd:
        print("\n   ==> PROBLEMA CONFIRMADO. Nao ha coeficientes SIP no header.")
        print("       Forcar CTYPE='RA---TAN-SIP' declara uma distorcao que nao")
        print("       existe, e o astropy tenta inverter algo mal-condicionado")
        print("       (e o 'failed to converge' do log).")
        print("       Acao: apply_sip_correction -> False para o S4G em config.py.")
        return False

    if fwd and not inv:
        print("\n   ==> ATENCAO. Ha SIP direto mas nao inverso (sem AP_/BP_).")
        print("       O astropy tem que inverter numericamente, e e ai que ele")
        print("       diverge. O SIP e real, mas a inversao e o gargalo.")
        print("       Veja o teste 3 para saber se o desvio e tolerável.")
        return True

    print("\n   ==> SIP completo (direto + inverso). A distorcao e real e")
    print("       invertivel. A divergencia do log e provavelmente pontual,")
    print("       fora da area util. Confirme com o teste 3.")
    return True


# ----------------------------------------------------------------------------
# Teste 3 — alinhamento entre bandas apos reprojecao
# ----------------------------------------------------------------------------
def teste_3_alinhamento(output_dir, galaxy):
    print()
    print(SEP)
    print("TESTE 3 — as bandas ficaram alinhadas? (P1.6b)")
    print(SEP)

    rep = output_dir / "reprojected_files" / galaxy

    master = find_one(rep, f"{galaxy}_s4g_irac1_master_Jy_per_pixel.fits",
                      "master irac1 em Jy")
    if master is None:
        return

    with fits.open(master) as hdu:
        pix = np.sqrt(hdu[0].header.get("PIXAREA",
                      proj_plane_pixel_area(WCS(hdu[0].header)) * 3600 ** 2))

    ym, xm = nucleus_centroid(master)
    print(f"   referencia (irac1 master): nucleo em (y,x) = ({ym:.2f}, {xm:.2f})")
    print(f"   escala: {pix:.4f} arcsec/px\n")

    print(f"   {'banda':8s} {'dy (px)':>9s} {'dx (px)':>9s} "
          f"{'desvio px':>12s} {'desvio arcsec':>15s}  veredito")
    print("   " + "-" * 68)

    piores = []
    for filt in ("f275w", "f336w", "f438w", "f555w", "f814w", "irac2"):
        f = find_one(rep, f"{galaxy}_*_{filt}_on_s4g_irac1_projection_Jy_per_pixel.fits",
                     f"reprojecao {filt}")
        if f is None:
            continue
        y, x = nucleus_centroid(f)
        if not np.isfinite(y):
            print(f"   {filt:8s} {'-':>9s} {'-':>9s} {'sem sinal':>12s}")
            continue
        dy, dx = y - ym, x - xm
        d = float(np.hypot(dy, dx))
        arc = d * pix
        if d < 0.3:
            v = "OK"
        elif d < 1.0:
            v = "suspeito"
        else:
            v = "GRAVE"
            piores.append(filt)
        print(f"   {filt:8s} {dy:>+9.2f} {dx:>+9.2f} {d:>12.2f} "
              f"{arc:>15.2f}  {v}")

    print()
    if piores:
        print(f"   ==> PROBLEMA CONFIRMADO em: {', '.join(piores)}")
        print("       Desvio > 1 px (0,75 arcsec) entre bandas contamina a cor")
        print("       todas as regioes. Investigue o SIP (teste 2) antes de")
        print("       qualquer outra correcao.")
    else:
        print("   ==> Alinhamento dentro do tolerável (< 1 px em todas as bandas).")
        print("       A divergencia do WCS no log foi pontual, sem efeito util.")

    print("\n   [nota] o nucleo e a referencia mais confiavel, mas em F275W ele")
    print("          pode nao ser o pixel mais brilhante (UV e dominado por")
    print("          regioes HII). Um desvio grande SO em f275w/f336w e mais")
    print("          provavelmente isso do que astrometria — confirme visualmente.")


# ----------------------------------------------------------------------------
# Amostragem (barato, e fecha a discussao da secao I.5)
# ----------------------------------------------------------------------------
def teste_0_amostragem(output_dir, galaxy):
    print()
    print(SEP)
    print("EXTRA — amostragem de Nyquist na grade master (secao I.5)")
    print(SEP)

    cubes = sorted((output_dir / "datacubes" / galaxy).glob("*_sci_*_Jy_per_pixel.fits"))
    if not cubes:
        print("   [!] nenhum cubo encontrado.")
        return
    h = fits.getheader(cubes[0])
    pix = np.sqrt(h["PIXAREA"])
    print(f"   cubo: {cubes[0].name}")
    print(f"   PIXAREA = {h['PIXAREA']} arcsec2  ->  {pix:.4f} arcsec/px")
    print(f"\n   Compare com o FWHM do master na 'Resolutions Table' do log:")
    for fwhm, rotulo in ((1.5620, "gaussiana medida (irac1)"),
                         (1.66, "literatura IRAC ch1"),
                         (1.72, "literatura IRAC ch2")):
        n = fwhm / pix
        flag = "OK" if n >= 2.0 else "SUBAMOSTRADO"
        print(f"      FWHM {fwhm:.4f} arcsec  ({rotulo:24s}) -> "
              f"{n:.2f} px/FWHM  {flag}")
    print("\n   Nyquist pede >= 2,0 px por FWHM.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("AsTrovello — diagnostico etapa 1 (somente leitura)")
    ap.add_argument("--galaxy", required=True, help="nome da galaxia (ex: ngc2903)")
    ap.add_argument("--verbose", action="store_true", help="detalhes das medidas de PRF")
    args = ap.parse_args()

    galaxy = args.galaxy.lower()

    base_dir = Path.cwd().parents[1]
    input_dir = base_dir / "Input"
    output_dir = base_dir / "Output"

    print(SEP)
    print(f"DIAGNOSTICO ETAPA 1 — {galaxy.upper()}   (somente leitura)")
    print(SEP)
    print(f"   base: {base_dir}")
    if not input_dir.is_dir():
        print(f"\n   [!] 'Input' nao encontrado. Rode este script do mesmo")
        print(f"       diretorio de onde voce roda o astrovello_cli_2.0.py.")
        return

    teste_1_prf(input_dir, args.verbose)
    teste_2_sip(input_dir, galaxy)
    teste_3_alinhamento(output_dir, galaxy)
    teste_0_amostragem(output_dir, galaxy)

    print()
    print(SEP)
    print("FIM. Nada foi alterado em disco.")
    print(SEP)


if __name__ == "__main__":
    main()
