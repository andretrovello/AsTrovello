"""
diagnostico_etapa1c.py — AsTrovello 2.0

SOMENTE LEITURA.

Motivacao: o diagnostico 1b mostrou que os cinco kernels HST -> irac1 tem
TODOS os pixels negativos e somam ~-0.50, enquanto o kernel irac2 -> irac1
(o unico par com escalas de pixel iguais) soma +0.95. Isso e o padrao de um
problema de GRADE: as PSFs limpas de HST e IRAC tem tamanhos de array
parecidos mas extensoes angulares que diferem por ~31x.

Este script responde a pergunta que decide tudo:

  O kernel, aplicado a PSF de origem, reproduz a PSF de destino?

Se sim, o casamento de PSF esta funcionando e o sinal invertido e cosmetico
(a normalizacao do create_convolvedFITS o cancela). Se nao, as cinco bandas
HST nunca foram levadas a resolucao do IRAC.

COMO RODAR
----------
    conda activate capivara
    python diagnostico_etapa1c.py

Requer que --create_kernel tenha rodado (precisa de Input/<survey>/PSF_CLEAN).
Dependencias: numpy, astropy, scipy (todas ja usadas pela pipeline).
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

SEP = "=" * 78

NATIVE_SCALE = {
    "f275w": 0.0395, "f336w": 0.0395, "f438w": 0.0395,
    "f555w": 0.0395, "f814w": 0.0395,
    "irac1": 1.221, "irac2": 1.223,
}


def load_2d(path):
    with fits.open(path, ignore_missing_end=True, ignore_missing_simple=True) as hdu:
        data = next((h.data for h in hdu if h.data is not None), None)
        hdr = hdu[0].header
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
        data = np.nanmean(data, axis=0)
    return np.nan_to_num(data, nan=0.0), hdr


def header_pixscale(hdr):
    """Escala de pixel em arcsec, tentando as convencoes na ordem usual."""
    if "PIXSCALE" in hdr:
        return float(hdr["PIXSCALE"]), "PIXSCALE"
    for k in ("CDELT2", "CDELT1"):
        if k in hdr and float(hdr[k]) != 0.0 and abs(float(hdr[k])) != 1.0:
            return abs(float(hdr[k])) * 3600.0, k
    for k in ("CD2_2", "CD1_1"):
        if k in hdr and float(hdr[k]) != 0.0:
            return abs(float(hdr[k])) * 3600.0, k
    return np.nan, "ausente"


def encircled(data, pixscale, fractions=(0.5, 0.8)):
    """Raios de energia encerrada em arcsec, sobre |data| (aceita kernel negativo)."""
    d = np.abs(data)
    cy, cx = np.unravel_index(np.argmax(d), d.shape)
    y, x = np.indices(d.shape)
    r = np.sqrt((x - cx) ** 2.0 + (y - cy) ** 2.0)
    r_max = min(d.shape) / 2.0
    inside = r <= r_max
    rf, vf = r[inside].ravel(), d[inside].ravel()
    order = np.argsort(rf)
    cum = np.cumsum(vf[order])
    if cum[-1] <= 0:
        return {f: np.nan for f in fractions}
    cum = cum / cum[-1]
    return {f: float(np.interp(f, cum, rf[order])) * pixscale for f in fractions}


def find_clean_psf(input_dir, filt):
    """Localiza a PSF limpa correspondente ao filtro, nos dois surveys."""
    pats = {
        "irac1": "*IRAC1*", "irac2": "*IRAC2*",
    }
    pat = pats.get(filt, f"*{filt.upper()}*")
    for survey in ("PHANGS", "S4G"):
        d = input_dir / survey / "PSF_CLEAN"
        if not d.is_dir():
            continue
        hits = sorted(d.glob(pat + ".fits"))
        if hits:
            return hits[0]
    return None


# ----------------------------------------------------------------------------
def inventario(input_dir, kernel_dir):
    print(SEP)
    print("PARTE 1 — inventario de grades (PSFs limpas e kernels)")
    print(SEP)
    print("   Se a escala do kernel nao bater com a escala NATIVA da imagem de")
    print("   ciencia a que ele e aplicado, a convolucao esta errada em extensao")
    print("   angular, mesmo que a forma pareca razoavel.\n")

    print(f"   {'objeto':30s} {'shape':>12s} {'escala':>10s} {'origem':>10s} "
          f"{'extensao':>10s}")
    print("   " + "-" * 76)

    for filt in NATIVE_SCALE:
        f = find_clean_psf(input_dir, filt)
        if f is None:
            print(f"   {('PSF_CLEAN ' + filt):30s} {'(ausente)':>12s}")
            continue
        d, h = load_2d(f)
        px, src = header_pixscale(h)
        ext = d.shape[0] * px if np.isfinite(px) else np.nan
        print(f"   {('PSF_CLEAN ' + filt):30s} {str(d.shape):>12s} "
              f"{px:>10.4f} {src:>10s} {ext:>9.2f}\"")

    print()
    for k in sorted(kernel_dir.glob("kernel_*_to_*.fits")):
        d, h = load_2d(k)
        px, src = header_pixscale(h)
        ext = d.shape[0] * px if np.isfinite(px) else np.nan
        name = "kernel " + k.stem.replace("kernel_", "")
        ext_s = f"{ext:>9.2f}\"" if np.isfinite(ext) else f"{'?':>10s}"
        print(f"   {name:30s} {str(d.shape):>12s} "
              f"{px:>10.4f} {src:>10s} {ext_s}")

    print("\n   Escalas nativas das imagens de ciencia, para comparar:")
    print("      PHANGS (f*):  0.0395 arcsec/px  (mosaico drizzled ~0.04)")
    print("      S4G (irac*):  0.7500 arcsec/px  (mosaico .phot, do WCS)")
    print("\n   [!] note que a escala NATIVA do detector IRAC (1.221) que o")
    print("       clean_psf grava na PSF limpa NAO e a escala do mosaico (0.75).")


# ----------------------------------------------------------------------------
def raw_prf(input_dir, filt):
    """PRF/PSF ORIGINAL (superamostrada), com sua escala. Serve de alvo
    confiavel: a PSF limpa do IRAC esta subamostrada e nao serve de referencia."""
    pats = {"irac1": "*IRAC1*col129_row129.fits", "irac2": "*IRAC2*col129_row129.fits"}
    pat = pats.get(filt, f"*{filt.upper()}*.fits")
    for survey in ("PHANGS", "S4G"):
        d = input_dir / survey / "PSF"
        if not d.is_dir():
            continue
        hits = sorted(d.glob(pat))
        if hits:
            binf = 5 if filt.startswith("irac") else 4
            data, _ = load_2d(hits[0])
            return data, NATIVE_SCALE[filt] / binf
    return None, np.nan


def validacao(input_dir, kernel_dir):
    print()
    print(SEP)
    print("PARTE 2 — o kernel reproduz a PSF de destino? (teste decisivo)")
    print(SEP)
    print("   Para cada kernel K, normalizado exatamente como a pipeline faz")
    print("   (K / soma(K)), calculamos  PSF_origem_limpa (*) K  na grade da")
    print("   origem, e comparamos o r50 em ARCSEC com o da PRF ORIGINAL de")
    print("   destino (superamostrada). Nao usamos a PSF limpa de destino como")
    print("   referencia porque ela esta subamostrada — ver Parte 3.\n")

    print(f"   {'kernel':22s} {'soma K':>9s} {'r50 origem':>11s} "
          f"{'r50 obtido':>11s} {'r50 alvo':>10s} {'razao':>7s}  veredito")
    print("   " + "-" * 86)

    resultados = []
    for k in sorted(kernel_dir.glob("kernel_*_to_*.fits")):
        stem = k.stem.replace("kernel_", "")
        try:
            src_filt, tgt_filt = stem.split("_to_")
        except ValueError:
            continue

        f_src = find_clean_psf(input_dir, src_filt)
        if f_src is None:
            print(f"   {stem:22s}  PSF limpa de origem ausente, pulando")
            continue

        kern, _ = load_2d(k)
        psf_src, h_src = load_2d(f_src)
        px_src, _ = header_pixscale(h_src)

        ksum = kern.sum()
        if ksum == 0:
            print(f"   {stem:22s}  soma do kernel = 0, pulando")
            continue
        kern_n = kern / ksum

        tgt_raw, px_tgt_raw = raw_prf(input_dir, tgt_filt)
        if tgt_raw is None:
            print(f"   {stem:22s}  PRF original de destino ausente, pulando")
            continue

        # "full" (nao "same"): "same" truncaria o resultado ao tamanho da
        # PSF de origem, cortando justamente o borramento que queremos medir.
        obtido = fftconvolve(psf_src, kern_n, mode="full")

        r50_src = encircled(psf_src, px_src)[0.5]
        r50_obt = encircled(obtido, px_src)[0.5]
        r50_tgt = encircled(tgt_raw, px_tgt_raw)[0.5]

        razao = r50_obt / r50_tgt if r50_tgt > 0 else np.nan

        # Se o pixel da grade de origem for grande em relacao ao r50 alvo, a
        # medida e limitada por quantizacao e nao decide nada.
        if px_src > 0.4 * r50_tgt:
            print(f"   {stem:22s} {ksum:>+9.4f} {r50_src:>10.3f}\" "
                  f"{r50_obt:>10.3f}\" {r50_tgt:>9.3f}\" {razao:>7.2f}  "
                  f"indeterminado (grade grosseira)")
            resultados.append((stem, razao, "indeterminado"))
            continue

        if 0.85 <= razao <= 1.15:
            v = "OK"
        elif 0.6 <= razao <= 1.5:
            v = "marginal"
        elif razao < 0.6:
            v = "FALHA (nao borrou)"
        else:
            v = "FALHA (borrou demais)"

        print(f"   {stem:22s} {ksum:>+9.4f} {r50_src:>10.3f}\" "
              f"{r50_obt:>10.3f}\" {r50_tgt:>9.3f}\" {razao:>7.2f}  {v}")
        resultados.append((stem, razao, v))

    print("\n   Interpretacao da coluna 'razao' (r50 obtido / r50 alvo):")
    print("      ~1.0        -> casamento de PSF funciona. O sinal invertido dos")
    print("                     kernels HST e cosmetico: a normalizacao o cancela.")
    print("      << 1 (<0.6) -> o kernel quase nao borrou. As bandas HST nunca")
    print("                     foram levadas a resolucao do IRAC. Isso invalida")
    print("                     todas as cores e exige reconvoluir tudo.")
    print("      >> 1        -> borrou demais; perda de resolucao desnecessaria.")

    falhas = [s for s, _, v in resultados if v.startswith("FALHA")]
    if falhas:
        print(f"\n   ==> FALHA em: {', '.join(falhas)}")
        print("       Ver Parte 3: a causa mais provavel e a grade das PSFs limpas.")
    elif resultados:
        print("\n   ==> Todos os kernels reproduzem a resolucao de destino.")
        print("       O casamento de PSF esta funcionando. P1.5 pode ser fechado.")


def parte3_amostragem_psf(input_dir):
    print()
    print(SEP)
    print("PARTE 3 — as PSFs limpas estao bem amostradas para casar? ")
    print(SEP)
    print("   O clean_psf faz block_reduce pelo binned_factor, levando a PRF da")
    print("   grade superamostrada para a grade NATIVA do detector. Para o IRAC")
    print("   isso destroi a superamostragem que torna a PRF utilizavel.\n")

    print(f"   {'PSF':10s} {'grade bruta':>13s} {'grade limpa':>13s} "
          f"{'FWHM aprox':>11s} {'px/FWHM limpa':>14s}  status")
    print("   " + "-" * 78)

    for filt, fwhm_lit in (("f814w", 0.08), ("irac1", 1.66), ("irac2", 1.72)):
        binf = 5 if filt.startswith("irac") else 4
        raw_px = NATIVE_SCALE[filt] / binf
        clean_px = NATIVE_SCALE[filt]
        n = fwhm_lit / clean_px
        status = "OK" if n >= 2.0 else "SUBAMOSTRADA"
        print(f"   {filt:10s} {raw_px:>12.4f}\" {clean_px:>12.4f}\" "
              f"{fwhm_lit:>10.2f}\" {n:>14.2f}  {status}")

    print("\n   Casar PSFs exige as duas na MESMA grade, e essa grade precisa")
    print("   amostrar bem a PSF mais larga (>= 2 px por FWHM). Hoje a PSF do")
    print("   HST fica em 0.0395\"/px e a do IRAC em 1.221\"/px: um descasamento")
    print("   de 31x, com o alvo subamostrado. O PyPHER tem que reconciliar isso")
    print("   sozinho, e o resultado e imprevisivel.")
    print("\n   Pratica padrao (Aniano et al. 2011): reamostrar AMBAS para uma")
    print("   grade comum fina — no seu caso a grade de ciencia do PHANGS,")
    print("   0.0395\"/px — e so entao gerar o kernel. A PRF do IRAC bruta esta")
    print("   em 0.2442\"/px e pode ser interpolada para 0.0395 sem perda, porque")
    print("   ja e superamostrada. Assim o kernel sai na grade em que ele e")
    print("   realmente aplicado.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("AsTrovello — diagnostico etapa 1c (leitura)")
    ap.add_argument("--verbose", action="store_true", help="aceito e ignorado")
    ap.parse_args()

    base_dir = Path.cwd().parents[1]
    input_dir = base_dir / "Input"
    kernel_dir = base_dir / "Output" / "PSF_Kernels"

    print(SEP)
    print("DIAGNOSTICO ETAPA 1c — validacao dos kernels   (somente leitura)")
    print(SEP)
    print(f"   base: {base_dir}")

    if not kernel_dir.is_dir():
        print(f"\n   [!] {kernel_dir} nao existe. Rode com --create_kernel antes.")
        return

    faltando = [s for s in ("PHANGS", "S4G")
                if not (input_dir / s / "PSF_CLEAN").is_dir()]
    if faltando:
        print(f"\n   [!] PSF_CLEAN ausente em: {faltando}")
        print("       Rode o astrovello com --create_kernel para gerar.")
        return

    inventario(input_dir, kernel_dir)
    validacao(input_dir, kernel_dir)
    parte3_amostragem_psf(input_dir)

    print()
    print(SEP)
    print("FIM. Nada foi alterado em disco.")
    print(SEP)


if __name__ == "__main__":
    main()
