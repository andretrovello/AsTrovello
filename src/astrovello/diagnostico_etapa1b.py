"""
diagnostico_etapa1b.py — AsTrovello 2.0

Substitui os testes 1 e 3 do diagnostico_etapa1.py, que estavam mal
construidos. Continua SOMENTE LEITURA.

  TESTE 1b — inspeciona os KERNELS que o PyPHER produziu, em vez de comparar
             larguras de PRF. Um kernel de suavizacao legitima e quase todo
             positivo; um kernel de deconvolucao tem lobulos negativos fortes.
             Medida direta, sem inferencia sobre forma de PSF.

  TESTE 3b — mede o deslocamento entre bandas por CORRELACAO CRUZADA num
             recorte ao redor da posicao do nucleo prevista pelo WCS, em vez
             de usar o argmax global (que encontra estrelas de campo e
             artefatos, nao o nucleo).

COMO RODAR
----------
    conda activate capivara
    python diagnostico_etapa1b.py --galaxy ngc2903

A coordenada default e a de NGC 2903. Para outra galaxia:
    python diagnostico_etapa1b.py --galaxy ngc1087 --ra 41.6046 --dec -0.4986

Dependencias: numpy + astropy (mesmas de sempre).
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SEP = "=" * 78

# NGC 2903: RA 09h32m10.1s, Dec +21d30m03s
DEFAULT_RA = 143.04212
DEFAULT_DEC = 21.50083


# ----------------------------------------------------------------------------
def load_2d(path):
    with fits.open(path, ignore_missing_end=True, ignore_missing_simple=True) as hdu:
        data = next((h.data for h in hdu if h.data is not None), None)
    if data is None:
        raise ValueError(f"Sem dados em {path}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
        data = np.nanmean(data, axis=0)
    return data


def encircled_radii(data, pixscale, fractions=(0.5, 0.8, 0.9)):
    """Raios de energia encerrada, em arcsec. As asas aparecem em r80/r90."""
    d = np.nan_to_num(data, nan=0.0)
    cy, cx = np.unravel_index(np.argmax(d), d.shape)
    y, x = np.indices(d.shape)
    r = np.sqrt((x - cx) ** 2.0 + (y - cy) ** 2.0)
    r_max = min(d.shape) / 2.0

    outer = (r > 0.85 * r_max) & (r <= r_max)
    d = d - (np.median(d[outer]) if outer.sum() > 20 else 0.0)

    inside = r <= r_max
    rf, vf = r[inside].ravel(), d[inside].ravel()
    order = np.argsort(rf)
    cum = np.cumsum(vf[order])
    if cum[-1] <= 0:
        return {f: np.nan for f in fractions}
    cum = cum / cum[-1]
    return {f: float(np.interp(f, cum, rf[order])) * pixscale for f in fractions}


# ----------------------------------------------------------------------------
# TESTE 1b — os kernels do PyPHER
# ----------------------------------------------------------------------------
def teste_1b_kernels(input_dir, output_dir):
    print(SEP)
    print("TESTE 1b — o kernel irac2 -> irac1 e suavizacao ou deconvolucao?")
    print(SEP)

    # --- contexto: energia encerrada das duas PRFs -------------------------
    psf_dir = input_dir / "S4G" / "PSF"
    native = {"irac1": 1.221, "irac2": 1.223}
    print("   contexto — energia encerrada das PRFs (arcsec):")
    print(f"      {'PRF':8s} {'r50':>8s} {'r80':>8s} {'r90':>8s}")
    ee = {}
    for filt, nat in native.items():
        hits = sorted(psf_dir.glob(f"*{filt.upper()}*col129_row129.fits"))
        if not hits:
            print(f"      [!] PRF {filt} nao encontrada")
            continue
        ee[filt] = encircled_radii(load_2d(hits[0]), nat / 5.0)
        print(f"      {filt:8s} {ee[filt][0.5]:>8.4f} {ee[filt][0.8]:>8.4f} "
              f"{ee[filt][0.9]:>8.4f}")

    if len(ee) == 2:
        print("\n      diferenca irac2 - irac1, em %:")
        for f in (0.5, 0.8, 0.9):
            dpct = 100.0 * (ee["irac2"][f] - ee["irac1"][f]) / ee["irac1"][f]
            print(f"         r{int(f*100)}: {dpct:+6.2f}%")
        print("      (as asas dominam o PSF matching; olhe r80/r90, nao r50)")

    # --- o que importa: os kernels -----------------------------------------
    kdir = output_dir / "PSF_Kernels"
    kernels = sorted(kdir.glob("kernel_*_to_*.fits"))
    if not kernels:
        print(f"\n   [!] nenhum kernel em {kdir}")
        return

    print(f"\n   kernels em {kdir.name}:\n")
    print(f"      {'kernel':28s} {'soma':>9s} {'neg %':>8s} "
          f"{'min/max':>9s}  veredito")
    print("      " + "-" * 70)

    rows = []
    for k in kernels:
        d = np.nan_to_num(load_2d(k), nan=0.0)
        total = d.sum()
        abs_sum = np.abs(d).sum()
        neg_frac = 100.0 * np.abs(d[d < 0].sum()) / abs_sum if abs_sum > 0 else np.nan
        ratio = abs(d.min()) / d.max() if d.max() > 0 else np.nan
        rows.append((k.stem.replace("kernel_", ""), total, neg_frac, ratio))

    # os kernels HST->irac sao suavizacao garantida (PSF 20x menor):
    # servem de linha de base para o que e "normal" de ringing no PyPHER
    baseline = [r[2] for r in rows if r[0].startswith("f")]
    base_med = float(np.median(baseline)) if baseline else np.nan

    for name, total, neg_frac, ratio in rows:
        if name.startswith("f"):
            v = "referencia (suavizacao)"
        elif np.isfinite(base_med) and neg_frac > max(3.0 * base_med, base_med + 5.0):
            v = "SUSPEITO"
        else:
            v = "compativel c/ suavizacao"
        print(f"      {name:28s} {total:>9.4f} {neg_frac:>7.2f}% "
              f"{ratio:>9.3f}  {v}")

    print(f"\n   Linha de base (kernels HST -> irac1): potencia negativa mediana "
          f"= {base_med:.2f}%")
    print("   Os kernels HST -> irac1 sao suavizacao pura por construcao (a PSF")
    print("   do HST e ~20x menor que a do IRAC), entao o ringing deles e o piso")
    print("   normal do PyPHER. Se o irac2 -> irac1 estiver muito acima disso,")
    print("   e deconvolucao. Se estiver na mesma faixa, o master esta OK.")


# ----------------------------------------------------------------------------
# TESTE 3b — alinhamento por correlacao cruzada
# ----------------------------------------------------------------------------
def parabolic_subpixel(c, i):
    """Refinamento sub-pixel por parabola em 3 pontos."""
    if i <= 0 or i >= len(c) - 1:
        return 0.0
    ym1, y0, yp1 = c[i - 1], c[i], c[i + 1]
    den = ym1 - 2.0 * y0 + yp1
    return 0.0 if den == 0 else 0.5 * (ym1 - yp1) / den


def measure_shift(ref, img):
    """
    Deslocamento de img em relacao a ref, por correlacao cruzada FFT.
    Retorna (dy, dx) em pixels: quanto img esta deslocada em relacao a ref.
    """
    a = np.nan_to_num(ref - np.nanmedian(ref), nan=0.0)
    b = np.nan_to_num(img - np.nanmedian(img), nan=0.0)
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan

    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    A = np.fft.fft2(a * w)
    B = np.fft.fft2(b * w)
    cc = np.fft.fftshift(np.fft.ifft2(A * np.conj(B)).real)

    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    cy0, cx0 = cc.shape[0] // 2, cc.shape[1] // 2
    dy = (py - cy0) + parabolic_subpixel(cc[:, px], py)
    dx = (px - cx0) + parabolic_subpixel(cc[py, :], px)
    return -dy, -dx


def cutout_at_world(path, ra, dec, half=64):
    """Recorte centrado na coordenada de ceu dada, via WCS do proprio arquivo."""
    with fits.open(path) as hdu:
        data = np.asarray(hdu[0].data, dtype=np.float64)
        w = WCS(hdu[0].header, naxis=2)
    x, y = w.all_world2pix(ra, dec, 0)
    xi, yi = int(round(float(x))), int(round(float(y)))
    y0, y1 = yi - half, yi + half
    x0, x1 = xi - half, xi + half
    if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
        return None, (yi, xi), 0.0
    sub = data[y0:y1, x0:x1]
    finite = float(np.isfinite(sub).mean())
    return sub, (yi, xi), finite


def teste_3b_alinhamento(output_dir, galaxy, ra, dec):
    print()
    print(SEP)
    print("TESTE 3b — alinhamento por correlacao cruzada (P1.6b, corrigido)")
    print(SEP)
    print(f"   nucleo esperado em RA={ra:.5f} Dec={dec:+.5f}")

    rep = output_dir / "reprojected_files" / galaxy
    master = rep / f"{galaxy}_s4g_irac1_master_Jy_per_pixel.fits"
    if not master.exists():
        print(f"   [!] master nao encontrado: {master}")
        return

    ref, ref_px, ref_finite = cutout_at_world(master, ra, dec)
    if ref is None:
        print(f"   [!] a coordenada cai fora do master (pixel previsto {ref_px}).")
        print("       Confira --ra/--dec.")
        return

    print(f"   pixel previsto pelo WCS do master: (y,x) = "
          f"({ref_px[0]}, {ref_px[1]})  [imagem {fits.getdata(master).shape}]")
    print(f"   fracao finita no recorte de referencia: {ref_finite:.1%}\n")

    print(f"      {'banda':8s} {'px previsto':>16s} {'finito':>8s} "
          f"{'dy':>7s} {'dx':>7s} {'desvio px':>10s} {'arcsec':>8s}  veredito")
    print("      " + "-" * 86)

    graves, suspeitos = [], []
    for filt in ("f275w", "f336w", "f438w", "f555w", "f814w", "irac2"):
        hits = sorted(rep.glob(
            f"{galaxy}_*_{filt}_on_s4g_irac1_projection_Jy_per_pixel.fits"))
        if not hits:
            print(f"      {filt:8s} {'(nao encontrado)':>16s}")
            continue

        sub, px, finite = cutout_at_world(hits[0], ra, dec)
        if sub is None or finite < 0.5:
            print(f"      {filt:8s} {str(px):>16s} {finite:>7.1%} "
                  f"  recorte sem dado suficiente")
            continue

        dy, dx = measure_shift(ref, sub)
        if not np.isfinite(dy):
            print(f"      {filt:8s} {str(px):>16s} {finite:>7.1%} "
                  f"  sem estrutura para correlacionar")
            continue

        d = float(np.hypot(dy, dx))
        if d < 0.5:
            v = "OK"
        elif d < 1.5:
            v = "suspeito"
            suspeitos.append(filt)
        else:
            v = "GRAVE"
            graves.append(filt)
        print(f"      {filt:8s} {str(px):>16s} {finite:>7.1%} "
              f"{dy:>+7.2f} {dx:>+7.2f} {d:>10.2f} {d*0.75:>8.2f}  {v}")

    print()
    if graves:
        print(f"   ==> DESALINHAMENTO em: {', '.join(graves)}")
        print("       Investigue o SIP e a reprojeção antes de qualquer")
        print("       outra correcao — isso contamina a cor de todas as regioes.")
    elif suspeitos:
        print(f"   ==> Desvio sub-pixel em: {', '.join(suspeitos)}")
        print("       Entre 0,5 e 1,5 px. Tolerável para SEDs de regiao grande")
        print("       (grupos de ~1000 px), mas anote como sistematica conhecida.")
    else:
        print("   ==> Alinhamento OK (< 0,5 px em todas as bandas).")
        print("       O 'failed to converge' do log foi pontual, sem efeito util.")
        print("       O P1.6 pode ser fechado.")

    print("\n   [nota] a correlacao cruzada usa toda a estrutura do recorte de")
    print("          128x128 px, nao um pixel so — imune a estrela de campo e")
    print("          a artefato, ao contrario do argmax global do teste 3 antigo.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("AsTrovello — diagnostico etapa 1b (leitura)")
    ap.add_argument("--galaxy", required=True)
    ap.add_argument("--ra", type=float, default=DEFAULT_RA,
                    help="RA do nucleo em graus (default: NGC 2903)")
    ap.add_argument("--dec", type=float, default=DEFAULT_DEC,
                    help="Dec do nucleo em graus (default: NGC 2903)")
    args = ap.parse_args()

    galaxy = args.galaxy.lower()
    base_dir = Path.cwd().parents[1]
    input_dir = base_dir / "Input"
    output_dir = base_dir / "Output"

    print(SEP)
    print(f"DIAGNOSTICO ETAPA 1b — {galaxy.upper()}   (somente leitura)")
    print(SEP)
    print(f"   base: {base_dir}")

    teste_1b_kernels(input_dir, output_dir)
    teste_3b_alinhamento(output_dir, galaxy, args.ra, args.dec)

    print()
    print(SEP)
    print("FIM. Nada foi alterado em disco.")
    print(SEP)


if __name__ == "__main__":
    main()
