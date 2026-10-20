# AsTrovello 2.0 — histórico da construção da pipeline

**Documento de referência para o projeto semestral do mestrado**
André Almeida Trovello (IAG-USP) · compilado em 2026-10-13

Este documento reconstrói, em ordem cronológica, como a pipeline AsTrovello 2.0
chegou ao estado atual: o que foi construído, que defeitos apareceram, como
cada um foi diagnosticado e medido, e por que as decisões finais foram
tomadas, em particular a de **construir os cubos apenas com PHANGS-HST e
PHANGS-JWST, deixando o S4G de fora**.

Fontes: os relatórios do repositório (`astrovello_pipeline_documentation.md`,
`audit_evaluation.md`, `session_report_audit_and_validation.md`,
`fix_verification_2026-09-28/29/30.md`, `per_band_grid_report.md`,
`CLAUDE.md`), o histórico do git, o notebook `code_audit.ipynb` e algumas
medições novas feitas sobre os arquivos em `Output/` para este documento
(marcadas como **[medição nova, 2026-10-13]**).

> **Como usar.** As Seções 1–3 dão o panorama. As Seções 4–9 contam a história
> por fase. A Seção 10 reúne as tabelas de validação. A Seção 11 explica a
> decisão sobre o S4G, a 12 trata dos negativos e a 13 traz a minha avaliação,
> incluindo **um problema novo no cubo final que precisa ser resolvido antes
> de qualquer ajuste de SED** (Seção 12.3).

---

## Sumário

1. [Resumo executivo](#1-resumo-executivo)
2. [Linha do tempo](#2-linha-do-tempo)
3. [A arquitetura final em uma página](#3-a-arquitetura-final-em-uma-página)
4. [Fase 1 — construção dos módulos (jul–set)](#4-fase-1--construção-dos-módulos-julset)
5. [Fase 2 — o bug da grade da v1.x](#5-fase-2--o-bug-da-grade-da-v1x)
6. [Fase 3 — a entrada do JWST](#6-fase-3--a-entrada-do-jwst)
7. [Fase 4 — auditoria e primeira rodada de correções](#7-fase-4--auditoria-e-primeira-rodada-de-correções)
8. [Fase 5 — segunda rodada: o problema do SIP no S4G](#8-fase-5--segunda-rodada-o-problema-do-sip-no-s4g)
9. [Fase 6 — B8: a grade de convolução por banda](#9-fase-6--b8-a-grade-de-convolução-por-banda)
10. [Os testes de validação: tabelas consolidadas](#10-os-testes-de-validação-tabelas-consolidadas)
11. [Por que o S4G foi deixado de fora](#11-por-que-o-s4g-foi-deixado-de-fora)
12. [Negativos: nos kernels e no cubo](#12-negativos-nos-kernels-e-no-cubo)
13. [Avaliação e pendências](#13-avaliação-e-pendências)
14. [Apêndices](#14-apêndices)

---

## 1. Resumo executivo

A AsTrovello recebe imagens de uma galáxia em vários levantamentos e produz um
cubo de dados em Jy/pixel, numa resolução angular comum e numa grade de pixels
comum, para alimentar a segmentação e o ajuste de SED (Capivara). A galáxia de
teste é **NGC 1087**.

**Configuração final:** PHANGS-HST (5 bandas: F275W, F336W, F438W, F555W,
F814W) + PHANGS-JWST (8 bandas: F200W, F300M, F335M, F360M, F770W, F1000W,
F1130W, F2100W). O master de resolução é o **MIRI F2100W**. O cubo final tem
13 bandas × 1202 × 1109 pixels de 0,1109″.

| Item | Critério | Medido | Resultado |
|---|---|---|---|
| Check 1: consistência de WCS HST↔JWST | < 0,5 px | **0,079 px** (0,0087″), sem sistemático | PASSA, margem de 6× |
| Check 2: casamento de PSF, 12 pares em 4 grades | razão r50 entre 0,95 e 1,05 | **0,9999–1,0001** | PASSA |
| Check 3: conservação de fluxo, 12 bandas | < 1% | **0,01%** no pior caso, dispersão máx. 0,10% | PASSA |
| Potência negativa dos kernels HST | W₋ de Aniano ≲ 1,2 (35,3% na nossa escala) | **0,00%** (medida no kernel bruto) | PASSA |
| Potência negativa dos kernels JWST | idem | **2,02% (F200W) a 16,67% (F1130W)** | PASSA |
| WCS do cubo | escala WCS = escala de `PIXAREA` | 0,11090538″ vs 0,11090537″ | OK |

**Três decisões estruturais**, todas tomadas a partir de medidas:

1. **O S4G foi retirado dos cubos.** A grade de mosaico do S4G (0,75″,
   reamostrada a partir dos pixels nativos de 1,22″ do IRAC) e os coeficientes
   SIP presentes nos cabeçalhos tornam a astrometria do levantamento
   não-verificável com os dados disponíveis. Incluí-lo também degradaria a
   resolução do cubo inteiro de ~0,64″ para 1,72″ (Seção 11).
2. **Pares de bandas com resolução parecida não são convoluídos.** Quando o
   alargamento necessário cabe em menos de ~2 pixels da grade de trabalho, o
   kernel vira quase uma delta e a divisão de Fourier produz lóbulos
   negativos. O caso emblemático é IRAC1→IRAC2: alargamento de 0,658″ = 0,88 px,
   ~31% de potência negativa nos dois sentidos (Seção 12.1).
3. **O kernel vive na grade da banda em que é aplicado.** Este princípio foi
   violado duas vezes: na v1.x, por um fator ~12, e de novo no JWST até
   30/09, por fatores de 1,8 a 3,6 (Seções 5 e 9).

**Um problema ainda aberto, encontrado ao preparar este documento:** a
subtração de céu do cubo atual remove luz da galáxia. Isto altera todos os
valores do cubo de forma diferente por banda (Seção 12.3). Pela regra de
triagem do projeto, é um **bloqueador** para o cubo HST+JWST.

---

## 2. Linha do tempo

Datas dos commits do branch `v2-dev/reproject_changes-audit`.

| Período | Commits | O que aconteceu |
|---|---|---|
| 28–30 jul | `first_commit` → `pypher kernel function refactoring` | Testes de convolução, cálculo de FWHM, tabela de resoluções, primeiro protótipo de geração de kernel com PyPHER |
| 08–13 ago | `Class implementation` → `input stage complete` | Arquitetura de *drivers* por levantamento; estágio de entrada |
| 17–30 ago | `convolution algorithm beginning` → `convolution module complete` | `clean_psf`, geração de kernels, convolução |
| 31 ago – 02 set | `alignment module - prototype` → `alignment module - tests` | Módulo de reprojeção (então chamado *alignment*) |
| 03–06 set | `unit conversion` → `data cube creation` | Conversão para Jy/pixel e montagem do cubo |
| **11 set** | `convolution module refactoring for better grid resampling` | **Correção do bug da grade da v1.x.** Remoção dos módulos v1.x (`convolution.py`, `alignment.py`, `cube.py`); `preflight.py`; diagnósticos `diagnostico_etapa1/1b/1c.py` |
| 13 set | `jwst implementation - testing` | Driver do PHANGS-JWST |
| **17 set** | `pipeline audit` | **Auditoria completa** do repositório (`CLAUDE.md`), documentação metodológica e resposta à auditoria |
| 21–26 set | `astrometry diagnostics` → `checks validated` | **Check 1**, consistência astrométrica HST↔JWST |
| **27 set** | `fix audit A1, A2, A3, B1` → `audit fixes report` | Primeira rodada de correções; `validation.py` (Checks 2 e 3 como travas da pipeline); cubo 1277×1135 |
| 28 set | `Claude Code fix verifications` | Verificação independente nº 1 |
| **29 set** | `bug fixes and new reports` | Segunda rodada (A2 em todos os caminhos, B2–B6); teste do SIP; verificação nº 2 → **descoberta do C4 (SIP espúrio do S4G)** |
| **30 set** | `pixscale and per band grid bug fixes` | **B8**: grade de convolução por banda; cubo 1202×1109; verificação nº 3 |
| 02 out | `evaluating negative pixels in final cube` | Análise dos pixels negativos no cubo final; `apply_mask` passa a ser `False` por padrão também em `create_data_cube` |

A trajetória das rodadas de verificação mostra convergência. A primeira
auditoria encontrou erros **nos dados** (WCS do cubo errado por 4800×, erro de
cor de 0,61% no HST, pixels de borda perdidos). A segunda encontrou **buracos
nas travas**. A terceira não encontrou erros nas correções, só endurecimento,
e um problema específico do S4G. A exceção foi o B8, que só apareceu quando
uma trava passou a funcionar de verdade.

---

## 3. A arquitetura final em uma página

### Os quatro estágios

1. **Casamento de PSF + convolução.** Cada banda é degradada à resolução da
   banda de pior resolução (o *master*).
2. **Reprojeção.** Cada banda é reamostrada para a grade de pixels do master.
3. **Conversão de unidades.** Cada banda é convertida para Jy/pixel.
4. **Cubo.** As bandas são empilhadas, o céu é subtraído, aplica-se a máscara e
   o cubo é escrito.

Duas travas (`validation.py`) abortam a execução se falharem:
`check_psf_matching` depois de gerar os kernels e antes de convoluir (razão r50
entre 0,95 e 1,05), e `check_flux_conservation` depois da reprojeção e antes da
conversão de unidades (< 1%). Os limites são constantes do módulo de propósito:
um limite que pode ser relaxado na chamada não é uma trava.

### Vocabulário (mantido separado de propósito)

| Termo | Operação | Status |
|---|---|---|
| **Reprojeção** | Levar as imagens para a mesma grade de pixels, confiando no WCS de entrada | É o que a pipeline faz |
| **Registro astrométrico** | Medir offsets entre fontes comuns e corrigir o WCS | **Não implementado** |
| **Casamento de PSF** (homogeneização) | Levar as imagens à mesma resolução angular | PyPHER + estágio de convolução |

O módulo antes chamado `alignment_2_0.py` virou `reprojection_2_0.py`, e o
modo `--mode alignment_only` virou `reprojection_only`, porque "alinhamento"
sugere registro astrométrico, que a pipeline não faz.

### Os princípios que sustentam o código

| # | Invariante | Por quê |
|---|---|---|
| 1 | O kernel só é válido na grade das PSFs usadas para construí-lo, e essa grade é **por banda** | O PyPHER devolve o kernel na grade das PSFs de entrada (Boucaud et al. 2016, Alg. 1: `N = N_b; p = p_b`) |
| 2 | Convoluir **antes** de reprojetar | O master (FWHM 1,72″ no caso S4G) numa grade de 0,75″ fica com 2,29 px/FWHM, acima de Nyquist. Uma imagem HST não convoluída (FWHM 0,08″) na mesma grade ficaria com 0,11 px/FWHM, subamostrada ~17× |
| 3 | O master é escolhido pelo **r80**, nunca pelo FWHM gaussiano | O casamento é dominado pelas asas. O ajuste gaussiano responde ao núcleo e inverteu a ordem dos dois canais IRAC |
| 4 | O *binning* antes da convolução usa a **média**, não a soma | Cada pixel binado continua significando "e⁻/s por pixel **nativo**", que é o que a conversão de unidades espera |
| 5 | Escalas de pixel das imagens de ciência vêm do **WCS**, nunca do `config.py` | A constante do config é a escala de referência da **PSF**, que para o S4G (1,221″) não é a do mosaico (0,75″) |
| 6 | O fator de binning é propriedade do **alvo**, derivado a cada execução | Depende de qual filtro é master |
| 7 | A informação viaja com os dados | Proveniência no cabeçalho (`PSFTARGT`, `PSFMATCH`, `PSFRESID`, `NATPXAR`, `BINFACT`, `PIXSCALE`) |
| 8 | Chave de proveniência ausente ou trava não verificável é **erro**, não *fallback* | Todos os defeitos A2/B2/B3 foram fallbacks para valores errados ou checagens puladas |

---

## 4. Fase 1 — construção dos módulos (jul–set)

A versão 2.0 nasceu de uma reescrita da 1.x com uma arquitetura de **drivers**:
cada levantamento guarda os dados de um jeito (convenção de nomes, unidade,
HDU de ciência, marcação de pixels inválidos), e o driver isola essas
diferenças atrás de uma interface comum.

| Driver | Unidade | Pixels inválidos | Conversão para Jy |
|---|---|---|---|
| `PHANGS_Driver` (HST) | e⁻/s | exatamente 0 | `PHOTFNU` (Jy por e⁻/s **por pixel nativo**), passando por brilho superficial |
| `PHANGS_JWST_Driver` | MJy/sr | exatamente 0 (confirmado: 0 NaN, ~1,5 M zeros na borda dos mosaicos MIRI) | × ângulo sólido do pixel (do WCS) |
| `S4G_Driver` | MJy/sr | NaN | × ângulo sólido do pixel (do WCS) |

A conversão é feita em dois passos por necessidade:

```
brilho_superficial = valor_bruto / área_do_pixel_nativo     # independe da grade
fluxo_por_pixel    = brilho_superficial × área_do_pixel_atual
```

O `PHOTFNU` do HST supõe um pixel nativo, mas nessa etapa a imagem já foi
reamostrada para outra grade. Passar pelo brilho superficial torna a conversão
correta em qualquer grade, que é a forma natural dos levantamentos em MJy/sr.

O primeiro "bug de unidade" do projeto vem dessa fase: a escala de pixel do S4G
foi lida do `config.py` (1,221″, o pixel nativo do IRAC) quando o mosaico está
em 0,75″. Daí nasceu o invariante 5. Ele reaparece na Seção 11.

---

## 5. Fase 2 — o bug da grade da v1.x

### O sintoma

Na v1.x as bandas HST não chegavam à resolução do IRAC. Os scripts de
diagnóstico (`diagnostico_etapa1*.py`, somente leitura, commit de 11/09)
foram escritos para responder, nesta ordem:

1. O master (então **IRAC1**) é mesmo a PSF mais larga? Se não for, o kernel
   IRAC2→IRAC1 é uma **deconvolução**.
2. Os coeficientes SIP do S4G são reais?
3. As bandas PHANGS e IRAC estão alinhadas depois da reprojeção?

### Os dois indicadores independentes

**Indicador 1, kernels com o sinal invertido** (`diagnostico_etapa1b.py`, lendo
os kernels brutos do disco):

```
5 kernels HST -> IRAC1 : TODOS os pixels negativos, soma ~ -0,50
kernel IRAC2 -> IRAC1  : soma +0,95 (o único par com escalas de pixel iguais)
```

Esse padrão aponta para um problema de grade: as PSFs limpas de HST e IRAC
tinham tamanhos de array parecidos, mas extensões angulares diferentes por ~31×.

**Indicador 2, o kernel reproduz a PSF alvo?** (`diagnostico_etapa1c.py`):

$$\frac{r_{50}(\text{PSF fonte} \ast \text{kernel})}{r_{50}(\text{PSF alvo})}$$

| | Razão r50 |
|---|---|
| Antes da correção | **0,08**: o alargamento aplicado era ~12× menor que o necessário |
| Depois da correção | **~1,06** |

### O diagnóstico

Os kernels eram construídos na grade do IRAC e aplicados nas imagens HST, na
grade nativa do HST. O PyPHER devolve o kernel na escala de pixel da PSF alvo
(Boucaud et al. 2016, Alg. 1: *"output: k_{a,b} … pixel scale p. N = N_b ;
p = p_b"*). Aniano et al. (2011, Sect. 7) dão a mesma instrução: *"Before
performing an image convolution, the kernel K{A ⇒ B} should be resampled onto
a grid with the same pixel size as the original image I_A."*

**Correção:** `clean_psf(convolution_grid_arcsec=...)` reamostra as duas PSFs
para a **grade de convolução** antes do PyPHER, e uma checagem em
`pypher_kernel_creation` levanta erro se alguma PSF estiver em outra grade.

### O master errado

O diagnóstico 1 mostrou também que o master estava trocado. O ajuste gaussiano
ordenava IRAC1 como mais larga; o raio de energia contida r80, medido nos
arquivos de PRF, dá o contrário:

| | r80 |
|---|---|
| IRAC1 | 2,0864″ |
| IRAC2 | **2,1878″** (master) |

O FWHM gaussiano responde ao núcleo e subestima em ~25% a largura de perfis
com asas extensas, como a PRF do IRAC. Escolher o master errado transforma
suavização em deconvolução: **31% de potência negativa** no kernel com o master
errado, contra 0% com o correto. O r80 foi escolhido por ficar onde o sinal
ainda domina o ruído mas as asas já entram, na faixa de 70–90% usada por
Aniano et al. (2011).

### O resíduo de 1,06

A razão ~1,06 depois da correção foi investigada mais tarde
(`pypher_regularisation_test.py`, 27/09). Hipóteses testadas e descartadas:

| Hipótese | Teste | Resultado |
|---|---|---|
| Regularização de Wiener do PyPHER | Varredura de `r` de 10⁻⁶ a 10⁻² | Razão fixa em 1,0000 (sintético e F1130W→F2100W) |
| Erro na medida | Controle sintético com gaussianas (kernel analítico) | 1,0000, D ~ 10⁻⁶ |
| Subamostragem da PSF fonte | Kernel analítico de 1,99 a 0,20 px/FWHM | Razão 1,0000 em todos; só D piora (7×10⁻⁸ → 1,6×10⁻³) |

Efeito real de `r` num par real (F275W→F2100W): 0,9999 (r = 10⁻⁶), **1,0001
(r = 10⁻⁴, padrão)**, 1,0167 (r = 10⁻²). O padrão do PyPHER está numa região
segura.

**Mecanismo mais provável, ainda não testado:** `diagnostico_etapa1c.py`
(linhas 207–209) mede o numerador na grade de convolução e o denominador na
grade bruta da PRF. A razão mistura duas quantizações diferentes. Os checks
atuais medem os dois termos na mesma grade e dão 1,0000–1,0001, o que é
consistente com essa explicação.

---

## 6. Fase 3 — a entrada do JWST

O driver do PHANGS-JWST (13/09) trouxe três novidades:

- **Unidades em MJy/sr** (brilho superficial), não e⁻/s. Os cabeçalhos JWST
  nem têm `PHOTFNU`.
- **Convenção de pixels inválidos:** zero exato, como no HST. A correção dessa
  convenção acelerou a convolução, porque a FFT deixou de interpolar sobre uma
  região inválida enorme.
- **Três escalas de pixel no mesmo levantamento:**

| Instrumento | Bandas | Escala nativa |
|---|---|---|
| NIRCam SW | F200W | 0,0307″ |
| NIRCam LW | F300M, F335M, F360M | 0,0630″ |
| MIRI | F770W, F1000W, F1130W, F2100W | 0,1109″ |

Esse terceiro ponto só foi tratado corretamente em 30/09 (Seção 9). Até lá a
pipeline supunha uma grade por levantamento.

Com o JWST, o F2100W passou a ser o candidato natural a master para uma
configuração sem S4G, numa resolução ~2,7× melhor que a do IRAC2.

---

## 7. Fase 4 — auditoria e primeira rodada de correções

### A auditoria (17/09)

Uma leitura completa do repositório, sem alterar nada, com cada afirmação
quantitativa verificada nos dados reais. Os defeitos foram agrupados em
A (corrompe a saída científica), B (convolução), C (reprojeção),
D (documentação × código) e E (estado do repositório).

| ID | Defeito | Efeito medido | Situação |
|---|---|---|---|
| **A1** | WCS do cubo copiado campo a campo; com matriz CD, `cdelt` sai como o padrão `[1,0; 1,0]` | Cubo dizia **1 grau/pixel** em vez de 0,75″: **erro de 4800×**. `PIXAREA`, calculado à parte, estava certo, por isso passou despercebido | Corrigido (27/09) |
| **A2** | `NATPXAR` escrito mas nunca lido; área nativa do HST vinha da constante 0,0395″ em vez do WCS (0,039620″) | Todas as bandas HST **0,61% mais brilhantes**: um termo de **cor** puro entre UV/óptico e IR | Corrigido (27/09 e 29/09) |
| **A3** | Off-by-one em `crop_to_mask_bbox` (índice inclusivo usado como limite exclusivo) | Perda de 2 linhas e 2 colunas de sinal | Corrigido (27/09) |
| **B1** | `binary_dilation` com elemento estruturante quadrado do tamanho do kernel | **~1400× mais lento** (29 s vs 0,02 s por banda) e `MemoryError` para kernels ≥ ~157–201 px | Corrigido (27/09) |
| **B2** | `inspect_kernel` mede depois de normalizar | Um kernel 100% negativo com soma −0,50 aparecia como 0,00% negativo, a mesma leitura de um kernel saudável | Corrigido (29/09) |
| **B3** | Sem `PIXSCALE` no kernel, a checagem de grade era pulada em silêncio | Kernels antigos da v1.x passariam sem checagem | Corrigido (29/09) e revelou o B8 |
| **B4** | Glob de kernels sem filtro pelo master atual | Kernel de outro master poderia ser aplicado | Corrigido (29/09) |
| **B5** | HST/JWST marcavam a borda com 0, o S4G com NaN | A interpolação bilinear misturava 0 com dados na borda | Corrigido (29/09) |
| **B6** | `min_blur` global, e não por levantamento | Grade HST limitada por um par IRAC que nem era casado: bin 5 em vez de 14, **8× mais pixels** | Corrigido (29/09) |
| B7 | `np.interp` sobre curva cumulativa não monotônica, entre outros | Estável em r80; falharia em 0,9/0,95 | Aberto (endurecimento) |
| C1, C2 | Chaves SIP não removidas / sufixo `-SIP` reanexado sem coeficientes | Latentes | Abertos; perdem a relevância sem S4G |
| C3 | `reproject_interp` é bilinear | Item de **documentação**: a 2,29 px/FWHM o efeito é pequeno, e o `reproject_adaptive` foi testado e não ganha S/N | Documentar |

#### Verificação de A1, A2 e A3 nos dados

```
A1  escala pelo WCS    : 0,11090538″/px
    escala por PIXAREA : 0,11090537″/px        (concordam em 8 casas)

A2  F814W novo/velho   : 0,993960  (previsto 0,99394, a razão de áreas)
    F2100W novo/velho  : 1,000005  (previsto 1,000000; brilho superficial não muda)

A3  antes : (2017, 1640) -> (1276, 1134)   947.925 px válidos
    depois: (2017, 1640) -> (1277, 1135)   947.959 px válidos
```

A2 é o exemplo mais claro da regra que guiou o redesenho: **"um erro que afeta
algumas bandas e não outras é um erro de cor, e um erro de cor vira um erro de
idade."** Um erro uniforme de 0,61% seria inofensivo; um de 0,61% entre UV e IR
não é.

#### B1 em detalhe

| k (px) | Idêntico? | Antigo | Novo | Ganho |
|---:|:---:|---:|---:|---:|
| 15 | sim | 0,046 s | 0,004 s | 12× |
| 41 | sim | 0,397 s | 0,005 s | 82× |
| 101 | sim | 5,684 s | 0,007 s | 771× |
| 157 | — | **MemoryError** | 0,007 s | — |

A dilatação por um quadrado plano k×k é **idêntica** a k//2 iterações de um
quadrado 3×3 (verificado com `np.array_equal`). Na prática, a convolução passou
de minutos para segundos por filtro. Sem essa correção, o kernel de 647 px do
F200W (Seção 9) provavelmente não teria rodado.

Fica registrado um mecanismo que foi **refutado**: a aceleração observada ao
corrigir a convenção de pixels inválidos do JWST não veio dessa dilatação. Com
uma máscara menor ela fica mais **lenta** (0,4% True → 2,32 s; 43,6% True →
1,33 s). Aquela aceleração foi da FFT.

### As travas de validação (27/09)

Os Checks 2 e 3 passaram a rodar **dentro** da pipeline e a abortá-la em caso de
falha (`ValidationError`). Comportamento verificado com casos que precisam
falhar:

```
kernel correto       (razão 1,0000)  -> PASSA
kernel 30% mais largo (razão 1,5811) -> ValidationError, aborta
reprojeção correta   (1,0010)        -> PASSA
fluxo vazando 5%     (0,9509)        -> ValidationError, aborta
```

---

## 8. Fase 5 — segunda rodada: o problema do SIP no S4G

### O ponto de partida: IRAC1 com 0,84% no Check 3

Numa execução HST+JWST+S4G (master IRAC2), todas as bandas passaram no Check 3,
menos uma anomalia: o **IRAC1** mostrava desvio de 0,84% (dispersão 1,43%).
Supostamente ele nem era reamostrado, já que estaria na mesma grade do IRAC2.

### Primeira tentativa: o raciocínio que estava errado

A sessão de 27–29/09 testou se os coeficientes SIP explicavam a anomalia,
removendo-os do arquivo **depois** da reprojeção (`test_sip_hypothesis.py`):

```
deslocamento ao remover SIP: mediana 6,6″, máx 49,7″
desvio IRAC1:  com SIP 0,840%   sem SIP 11,273%
```

A conclusão na época foi que "o SIP carrega parte da solução astrométrica do
S4G" e a mudança foi revertida. **Essa conclusão estava errada**, e o motivo
virou uma das lições do projeto: o teste mudou **um só lado** da comparação. Os
pixels do arquivo "depois" foram posicionados **com** SIP, e o astropy continuava
aplicando SIP ao arquivo "antes". O teste mediu se o cabeçalho ainda descrevia
como os pixels foram colocados, e não se a astrometria estava certa. Só podia
piorar.

### A verificação de 29/09: o SIP do S4G parece espúrio

**Duas premissas que não valiam:**

1. **O astropy aplica SIP mesmo sem o sufixo `-SIP` no CTYPE.** A própria
   mensagem diz: *"astropy.wcs is using the SIP distortion coefficients"*.
2. **O IRAC1 é reamostrado.** Os dois mosaicos do S4G não estão na mesma grade
   de pixels:

| | NAXIS | CRPIX | CRVAL | CD | A_2_0 | B_0_2 |
|---|---|---|---|---|---|---|
| `NGC1087.phot.1.fits` | 737×1052 | 270, 271 | 41,6049, −0,49868 | ±0,75″/px | −2,39×10⁻⁵ | 2,31×10⁻⁵ |
| `NGC1087.phot.2.fits` | 733×1050 | 468, 778 | igual | igual | +1,91×10⁻⁵ | 3,44×10⁻⁵ |

Mesmo CRVAL, mesmo CD, CRPIX deslocado por **um número inteiro** de pixels
(198, 507) e SIP **diferente** em cada canal. Sem SIP, o mapeamento
IRAC1→IRAC2 é um deslocamento inteiro exato, ou seja, uma cópia sem perdas. Com
SIP vira uma deformação que varia no campo, e o IRAC1 (que não é convoluído)
passa por interpolação bilinear.

**Medidas:**

Check 3 para IRAC1, mesma função, WCS consistente dos dois lados:

| Configuração | Razão | Dispersão |
|---|---|---|
| Pipeline (SIP aplicado dos dois lados) | 0,9916 | 1,43% |
| SIP removido antes e depois, reprojeção refeita | **1,0000** | **0,00%** |

Posições de estrelas IRAC1 × IRAC2 (picos > 15σ, centroide, casamento no céu):

| | n (< 1,5″) | Separação mediana |
|---|---|---|
| Com SIP | 131 | 0,321″ |
| Sem SIP | 137 | **0,126″** |

Por distância ao centro da galáxia:

| Raio | Com SIP | Sem SIP |
|---|---|---|
| 0–1,5′ (n = 98) | 0,193″ | 0,090″ |
| 1,5–3′ (n ≈ 44) | 0,544″ | 0,153″ |
| 3–5′ (n ≈ 4) | 1,458″ | 0,121″ |

**Com SIP, os dois canais divergem à medida que o raio cresce**, que é a
assinatura de uma distorção espúria: uma solução de distorção real, comum ao
céu, faria os canais concordarem **melhor**. Sem SIP, eles concordam a ~0,1″ em
todo o campo, no nível do ruído de centroide.

Tamanho do deslocamento que o SIP introduz (pixel→céu, com vs sem SIP):

| | 0–0,5′ | 0,5–1′ | 1–1,5′ | 1,5–3′ | 3–5′ |
|---|---|---|---|---|---|
| IRAC1 | 0,01″ | 0,05″ | 0,15″ | 0,67″ | 3,51″ |
| IRAC2 | 0,01″ | 0,07″ | 0,17″ | 0,79″ | 5,45″ |

**Interpretação:** os coeficientes foram muito provavelmente herdados dos
cabeçalhos dos quadros individuais (BCD), e a ausência do sufixo `-SIP` nos
mosaicos é deliberada: depois da mosaicagem a distorção já foi corrigida, e os
coeficientes ficaram como resíduo. Os cabeçalhos dos mosaicos do S4G
**carregam o histórico do pipeline de BCD** (`TRANHEAD`, `CALTRANS`,
`INSBPOSDOM`, `cvti2r4`, `hdrupd8` no `HISTORY`) **[medição nova,
2026-10-13]**, o que é compatível com essa herança. Isto é uma **inferência** a
partir da estrutura dos cabeçalhos e das medidas, não algo encontrado na
documentação do S4G.

### Consequência para cubos com S4G (defeito C4)

`reproject_to_reference` reanexa `-SIP` ao cabeçalho de **referência**. Assim,
toda banda HST/JWST colocada na grade do IRAC2 passa pelo SIP do IRAC2:
0,07–0,17″ a 0,5–1,5′ e ~0,8″ a 1,5–3′. É um **desalinhamento radial que
cresce para fora**, ou seja, um erro de cor dependente da posição. Cubos
HST+JWST (master F2100W) não são afetados.

---

## 9. Fase 6 — B8: a grade de convolução por banda

### A trava que nunca tinha disparado

A correção do B3 tornou fatal a ausência de `PIXSCALE` no kernel. A execução
seguinte abortou imediatamente:

```
ValueError: Kernel kernel_f336w_to_f2100w.fits declares no pixel scale
```

Regenerar os kernels não resolveu. O `clean_psf` escreve `PIXSCALE` nas PSFs,
mas quem escreve o kernel é o **PyPHER**, que descarta palavras-chave não
padronizadas:

```
PSF    PSFSTD_WFC3UV_F275W.fits      PIXSCALE = 0.1981
KERNEL kernel_f1000w_to_f2100w.fits  PIXSCALE = None
```

A trava de grade em `create_convolvedFITS`, a proteção contra a volta do bug da
v1.x, **nunca teve uma escala para comparar**. Ela caía no ramo
`k_px is None` em todas as bandas de todas as execuções desde que foi escrita.
Tinha sido revisada e documentada, e nunca funcionou.

### O que a trava encontrou quando passou a funcionar

Com `PIXSCALE` carimbado nos kernels:

```
ValueError: Kernel for 'f300m' is on 0.1109 arcsec/px but the convolution
image is on 0.0630 arcsec/px. The blur would be wrong by a factor of ~0.6.
```

`derive_bin_factors` usava `science_pixel_scale(imgs[0])`, o primeiro arquivo
em ordem alfabética, que era MIRI. Os kernels das quatro bandas NIRCam foram
construídos na grade MIRI (0,1109″) e aplicados em imagens de 0,0307″ e
0,0630″: **3,6× grosseiros demais para o F200W e 1,8× para LW**. É o bug da
v1.x de novo, num fator menor, presente em **todos os cubos HST+JWST anteriores
a 30/09**.

A orientação da Paula no e-mail de setembro já dizia isso: *"no pypher, para
cada banda, leve a PSF alvo para a grade nativa daquela imagem"*. A
implementação por levantamento não a seguia.

### A correção

- Bandas agrupadas por escala nativa × fator de bin.
- PSF master limpa **uma vez por grade distinta**.
- PSFs fonte limpas na grade da própria imagem de ciência.
- PyPHER roda uma vez por grupo de grade.
- `PIXSCALE` carimbado no kernel depois do PyPHER.
- PSFs limpas prefixadas com a grade (`0.0630_PSF_NIRCam_F300M.fits`).

### Verificação: todos os kernels cobrem a mesma extensão angular

| Kernel | PIXSCALE | Tamanho | Extensão |
|---|---|---|---|
| HST (×5) | 0,1981″ | 99 px | 19,6″ |
| F200W | 0,0307″ | 647 px | 19,9″ |
| F300M, F335M, F360M | 0,0630″ | 315 px | 19,8″ |
| F770W, F1000W, F1130W | 0,1109″ | 179 px | 19,9″ |

A extensão vem do array da PSF master: o WebbPSF F2100W `OVERSAMP` tem
724 px × 0,0277″ = 20,05″.

Verificação em cinco configurações:

| Configuração | Master | Bins | Kernels / grades | Checagem de grade | Check 2 |
|---|---|---|---|---|---|
| Só HST | F814W | 1 | 0: F275W/336W/438W/555W não casadas (0,067–0,079″ ≈ 2 px) | — | — |
| Só S4G | IRAC2 | 1 | 0: IRAC1 não casada (0,658″) | — | — |
| Só JWST | F2100W | 1 | 7 em 3 grades | 7/7 | 7/7 PASSA |
| **HST+JWST** | **F2100W** | 5 / 1 | **12 em 4 grades** | **12/12** | **12/12 PASSA** |
| HST+JWST+S4G | IRAC2 | 12 / 4 / 1 | 13 em 5 grades | 13/13 | 13/13 PASSA |

### O cubo encolheu, e corretamente

```
antes : 1277 × 1135      (27/09, NÃO USAR)
depois: 1202 × 1109      (−5,9% em y, −2,3% em x)
```

| Borda | Antes | Depois | Variação | Banda limitante |
|---|---|---|---|---|
| y inferior | 597 | 635 | +38 | NIRCam LW antes e depois |
| y superior | 1873 | 1836 | −37 | MIRI antes, LW agora |
| x direita | 1498 | 1475 | −23 | MIRI antes, LW agora |
| x esquerda | 364 | 367 | +3 | MIRI F1000W; anel de NaN do B5 |

A banda limitante é a **NIRCam LW**, não o F200W: o kernel LW passou de 179 px
para 315 px, e a borda cresceu (157 − 89) × 0,0630″ = 4,28″ = 38,6 px do master.
Agora **todas as bandas perdem os mesmos ~9,9″ por lado**. Antes a NIRCam
perdia menos só porque os kernels eram fisicamente pequenos demais.

Um fator de bin por banda não recuperaria essa área. A borda mascarada é metade
da extensão **angular** do kernel, qualquer que seja o bin. Serviria só para
reduzir o custo computacional (F200W poderia usar bin 6, ~36× menos pixels).

### Magnitude do erro antigo: não reportada

Uma comparação cubo novo/cubo velho em aberturas no céu deu F2100W (controle)
0,9963, NIRCam 0,946–0,963, HST 0,963–0,988 e MIRI 0,990–0,991, mas com
**dispersão de 7–30%**. A direção (a NIRCam foi a que mais mudou) é sustentada
pelos dados; a magnitude não. **Não citar um número.**

---

## 10. Os testes de validação: tabelas consolidadas

### Check 1 — consistência de WCS (HST × JWST)

**Pergunta:** o `reproject` confia no WCS de entrada. As soluções de WCS do
PHANGS-HST e do PHANGS-JWST concordam?

**Método:** cinco fontes pontuais escolhidas no DS9 com os quadros travados por
WCS. As coordenadas foram lidas **uma única vez**, no HST, para que o erro da
mão (~1 pixel de tela) não entrasse na medida. Cada coordenada foi convertida
para pixel nas duas imagens, o centroide foi refinado em cada uma
independentemente (`photutils.centroid_2dg`, caixa de 15 px no HST e 7 px no
JWST, cobrindo área de céu comparável), convertido de volta para o céu, e as
posições comparadas. As fontes cobrem ~60″ em AR e ~120″ em Dec, o que
detectaria rotação ou escala, não só translação.

**Teste de sanidade:** o mesmo procedimento com HST contra ele mesmo dá
**exatamente zero** nas cinco fontes.

| Fonte | AR (°) | Dec (°) | Sep (″) | Sep (px) | ΔAR (″) | ΔDec (″) |
|---:|---|---|---:|---:|---:|---:|
| 1 | 41,6036123 | −0,5152057 | 0,0072 | 0,065 | +0,0031 | +0,0065 |
| 2 | 41,6007478 | −0,5130389 | 0,0091 | 0,082 | −0,0021 | +0,0089 |
| 3 | 41,6135966 | −0,5130459 | 0,0018 | 0,016 | −0,0013 | −0,0012 |
| 4 | 41,6014181 | −0,4820472 | 0,0110 | 0,099 | +0,0080 | −0,0076 |
| 5 | 41,6173553 | −0,4850417 | 0,0087 | 0,079 | −0,0038 | +0,0079 |

```
separação mediana : 0,0087″ = 0,079 px da grade final (0,1109″)
dispersão         : 0,0031″
offset médio      : ΔAR +0,0008″, ΔDec +0,0029″
sistemático/mediana: 0,34
```

**PASSA, com margem de 6×.** O vetor de offset médio é três vezes menor que a
dispersão, e os sinais de ΔAR alternam. **Não há deslocamento sistemático**: o
que se mediu foi a precisão do método de centroide.

> ⚠️ O script `check1_astrometry.py` **não está no repositório**
> (`diagnostico_astrometria.py` é um esboço de duas linhas). Faça o commit
> antes de citar o Check 1.

### Check 2 — casamento de PSF

**Método:** convolui a PSF fonte com o kernel e compara o r50 do resultado com o
do master (Aniano et al. 2011, Sect. 5: *"For a perfect kernel, both quantities
should coincide at all radii"*). Métricas secundárias: razão r80 (mais
sensível às asas), potência negativa `W₋/(1+2W₋)` (o limite W₋ ≲ 1,2 de Aniano
equivale a 35,3%) e D de Aniano (Eq. 20, `∫|Ψ_B − K⋆Ψ_A|`).

**PHANGS-HST**, grade 0,1981″/px, master F2100W:

| Banda | px/FWHM | Razão r50 | Razão r80 | Pot. neg. | D |
|---|---:|---:|---:|---:|---:|
| F275W | 0,39 | 1,0001 | 1,0000 | 0,00% | 1,63×10⁻⁴ |
| F336W | 0,41 | 1,0001 | 1,0000 | 0,00% | 1,66×10⁻⁴ |
| F438W | 0,42 | 1,0001 | 1,0000 | 0,00% | 1,69×10⁻⁴ |
| F555W | 0,42 | 1,0001 | 1,0000 | 0,00% | 1,73×10⁻⁴ |
| F814W | 0,40 | 1,0000 | 1,0000 | 0,00% | 1,93×10⁻⁴ |

**PHANGS-JWST**, master F2100W:

| Banda | Grade | Razão r50 | Pot. neg. | D |
|---|---|---:|---:|---:|
| F200W | 0,0307″ | 1,0000 | **2,02%** | † |
| F300M | 0,0630″ | 0,9999–1,0000 | † | † |
| F335M | 0,0630″ | 0,9999–1,0000 | † | † |
| F360M | 0,0630″ | 0,9999–1,0000 | † | † |
| F770W | 0,1109″ | 1,0000 | 9,71% | 1,25×10⁻³ |
| F1000W | 0,1109″ | 1,0000 | 13,86% | 1,27×10⁻³ |
| F1130W | 0,1109″ | 1,0000 | 16,67% | 1,25×10⁻³ |

† Os valores publicados no relatório de sessão para as bandas NIRCam (F200W
2,54%, F300M 3,17%, F335M 3,65%, F360M 3,96%) foram medidos **antes do B8**, com
os kernels na grade errada. Depois do B8 só o F200W foi re-reportado (2,02%).
Os kernels MIRI não mudaram com o B8, então os valores MIRI continuam válidos.
**Rode o Check 2 standalone de novo para a NIRCam** (e antes ajuste
`check2_psf_matching.py`, que hoje pula o JWST por esperar um único
`master_*.fits`).

**Limite deste check, a declarar na dissertação:** ele convolui as **mesmas**
PSFs limpas que o PyPHER usou para gerar o kernel. Uma razão ≈ 1,0000 é
praticamente garantida por construção. O check pega kernel velho, grade errada,
kernel com sinal invertido e kernel alargado (o teste de 1,58), mas **não é
evidência independente** de que as imagens de ciência chegaram à resolução do
master. Essa evidência exigiria medir a PSF de estrelas de campo nas imagens
convoluídas, o que ainda não foi feito.

### Check 3 — conservação de fluxo

**Método:** aberturas fixas em **coordenadas de céu**, com a soma em cada
abertura **ponderada pela área do pixel** da sua grade (`|det(pixel_scale_matrix)|`,
correto sob CD, PC e rotação). Exige-se que cada abertura tenha ≥ 5% do fluxo da
mais brilhante.

**Duas correções no caminho, ambas instrutivas:**

1. A primeira versão não ponderava pela área e "falhou" de forma espetacular:
   cada razão era exatamente a razão das áreas de pixel. O `reproject_interp`
   preserva o **valor** do pixel (comportamento de brilho superficial), não a
   soma.

   ```
   HST       0,0697   (0,1981/0,75)² = 0,0698
   MIRI      0,0218   (0,1109/0,75)² = 0,0219
   NIRCam LW 0,0071   (0,0630/0,75)² = 0,0071
   NIRCam SW 0,0017   (0,0310/0,75)² = 0,0017
   ```

2. Aberturas em céu vazio davam ruído sobre ruído (dispersão de 23.000%). O
   corte de 5% derrubou a dispersão para ~0,3%.

**Validação do método** em reprojeções sintéticas de resposta conhecida, até
24× em escala (600× em área): razões 0,9991–1,0005, dispersão 0,29–0,41%. Um
caso corrompido em 5% falha, como deve.

**Resultado** (HST+JWST, master F2100W, grade 0,1109″):

| Banda | n ab. | Grade de entrada | Razão | Dispersão | Desvio |
|---|---:|---:|---:|---:|---:|
| F275W | 8 | 0,1981 | 0,9999 | 0,04% | +0,01% |
| F336W | 8 | 0,1981 | 0,9999 | 0,03% | +0,01% |
| F438W | 8 | 0,1981 | 0,9999 | 0,02% | +0,01% |
| F555W | 8 | 0,1981 | 0,9999 | 0,01% | +0,01% |
| F814W | 8 | 0,1981 | 0,9999 | 0,05% | +0,01% |
| F200W | 7 | 0,0307 | 0,9999 | 0,10% | +0,01% |
| F300M | 6 | 0,0630 | 0,9999 | 0,07% | +0,01% |
| F335M | 8 | 0,0630 | 1,0000 | 0,01% | +0,00% |
| F360M | 8 | 0,0630 | 0,9999 | 0,03% | +0,01% |
| F770W | 8 | 0,1109 | 0,9999 | 0,00% | +0,01% |
| F1000W | 8 | 0,1109 | 0,9999 | 0,01% | +0,01% |
| F1130W | 8 | 0,1109 | 0,9999 | 0,01% | +0,01% |

**PASSA**, pior desvio 0,01%, margem de 100× sobre o critério. São 12 bandas
porque o F2100W é o master e não é reprojetado. As três bandas MIRI já estão na
grade de referência (reamostragem nula) e servem de controle.

> Esta tabela é de uma execução anterior ao B8. O Check 3 mede a reprojeção, que
> o B8 não alterou (as imagens convoluídas sempre estiveram na grade nativa; só
> os kernels estavam errados). O relatório do B8 registra que o Check 3 voltou a
> passar, mas não traz a tabela. Para a dissertação, vale salvar a tabela da
> execução final.

---

## 11. Por que o S4G foi deixado de fora

Este é o argumento central da decisão, reunido a partir das Seções 5, 8 e 9. São
cinco razões independentes, e qualquer uma delas já pesaria.

### 11.1 Grade nativa × grade de ciência: o mosaico reamostrado

O detector IRAC tem pixels nativos de **1,221″** (canal 1) e **1,223″**
(canal 2), mas os mosaicos `.phot` do S4G estão em **0,75″/px**: são o resultado
da combinação e reamostragem (*drizzle*/mosaicagem) de muitos quadros
individuais pelo pipeline do S4G. Daí três consequências:

| Aspecto | Problema | Onde apareceu |
|---|---|---|
| **Unidades** | A escala do `config.py` (1,221″) é a escala de referência da PRF, não a do mosaico. Usá-la na conversão de unidades dava um erro de (1,221/0,75)² ≈ 2,65× | O primeiro bug de unidade da pipeline; deu origem ao invariante 5 |
| **PSF** | A PRF do IRAC é definida para o pixel nativo. O mosaico em 0,75″ tem PSF efetiva = PRF ⊛ reamostragem, e o ruído é **correlacionado** entre pixels vizinhos (Fruchter & Hook 2002) | O master IRAC2 (FWHM 1,72″) fica com 2,29 px/FWHM em 0,75″, mal acima de Nyquist |
| **Astrometria** | A mosaicagem herda dos quadros individuais coeficientes SIP que **não descrevem** o mosaico final | Seção 8; defeito C4 |

A diferença entre as duas grades é o ponto que mais causou problemas. Nos
dados PHANGS, a grade em que o arquivo está e a grade a que os metadados se
referem coincidem. No S4G elas não coincidem, e cada metadado (escala, PRF, SIP)
precisa ser verificado para saber a qual das duas pertence.

### 11.2 A astrometria do S4G não pôde ser validada

O Check 1 (HST×JWST) só foi possível porque as duas imagens têm resolução
parecida e muitas fontes pontuais nítidas em comum. Com o S4G:

- **Poucas fontes em comum.** A comparação S4G × JWST F200W (calibrado contra o
  Gaia) achou só **7–8 estrelas**, todas a menos de ~1′ do centro, onde o SIP
  desloca no máximo 0,15″. Com o SIP aplicado ou não, a mediana ficou em
  0,39–0,45″. O teste **não discrimina**.
- **Um indício de offset absoluto.** A mesma comparação sugere um offset médio
  S4G − JWST de **~−0,36″ em AR**. Com n = 8 e fontes estendidas, é só um
  indício, mas é 3× a precisão do Check 1 e nunca foi investigado.
- **Fontes estendidas e confusão.** Com PSF de 1,7″, o que é pontual no JWST é
  um aglomerado misturado no IRAC; o centroide mede outra coisa.
- **O teste decisivo não foi feito.** Seria comparar com estrelas do **Gaia DR3
  a 3–5′ do centro**, onde o SIP desloca as posições em 3,5–5,5″. O campo
  PHANGS-JWST não chega lá.

Ou seja: a consistência **interna** do S4G (IRAC1×IRAC2) foi medida, ~0,1″ sem
SIP, mas a concordância **absoluta** do S4G com HST/JWST nunca foi. A pipeline
não faz registro astrométrico (Seção 3), então incluir o S4G seria **supor**
um alinhamento que não se conseguiu medir.

### 11.3 O SIP introduz um erro de cor radial

Se o S4G entra, o master é o IRAC2, e toda banda HST/JWST é posicionada pelo SIP
do IRAC2 (Seção 8): 0,07–0,17″ a 0,5–1,5′ e ~0,8″ a 1,5–3′. A correção existe
(remover as chaves SIP na ingestão, ~5 linhas), mas afirmar na dissertação que
ela está **certa** exige o teste do Gaia.

### 11.4 IRAC1 e IRAC2 não podem ser casados

O alargamento necessário entre os dois canais, √(2,1878² − 2,0864²) = 0,658″, é
**0,88 px** na grade de 0,75″, abaixo do limite de 2 px. O kernel teria ~31% de
potência negativa nos dois sentidos (Seção 12.1). O IRAC1 entraria no cubo
**sem** casamento de PSF, só com o resíduo registrado em `PSFRESID`. É um
compromisso aceitável, mas é mais um item que a configuração sem S4G evita.

### 11.5 O custo em resolução

| | HST+JWST (master F2100W) | HST+JWST+S4G (master IRAC2) |
|---|---|---|
| Resolução do cubo | FWHM ~0,64″ | FWHM ~1,72″ (**~2,7× pior**) |
| Grade final | 0,1109″/px | 0,75″/px (**~46× menos pixels por área**) |
| Bins de convolução (HST/JWST/S4G) | 5 / 1 | 12 / 4 / 1 |
| Astrometria verificada | HST↔JWST, 0,079 px | Só HST↔JWST; S4G suposto |
| Problemas abertos | Nenhum bloqueador na convolução/reprojeção | C4 + teste do Gaia |

O objetivo científico é a SED **resolvida espacialmente**. Degradar o cubo
inteiro para 1,72″ para ganhar 3,6 e 4,5 µm, que o JWST já cobre em parte com
F335M/F360M e o MIRI estende até 21 µm, é uma troca ruim. As bandas
F300M–F360M amostram a mesma região espectral (a emissão estelar além do pico
de 1,6 µm e a banda de PAH de 3,3 µm).

### 11.6 Síntese

O S4G não foi retirado por um único defeito, mas porque **não foi possível
verificar** que ele se combina corretamente com os outros levantamentos. O SIP
espúrio tem correção, mas a correção não pode ser validada em termos absolutos
com os dados disponíveis. E o ganho espectral não compensa a perda de 2,7× em
resolução. Para a dissertação, a formulação honesta é: *o S4G foi avaliado,
sua consistência interna foi medida (IRAC1×IRAC2 ~0,1″ sem SIP), mas a
concordância absoluta com o HST/JWST não pôde ser verificada com as fontes
disponíveis no campo, e por isso ele não foi incluído.*

---

## 12. Negativos: nos kernels e no cubo

São dois fenômenos diferentes, que o projeto às vezes chama pelo mesmo nome. É
importante separá-los no texto.

### 12.1 Negativos nos kernels: resoluções parecidas

**Origem.** O kernel é obtido por divisão no espaço de Fourier:
K = FT⁻¹[FT(Ψ_B) / FT(Ψ_A)]. Aniano et al. (2011, Sect. 2): *"a condition for
the existence of a kernel is that … the PSF of camera A must be narrower than
the PSF of camera B."* Quando as duas PSFs têm larguras parecidas, as
frequências de corte quase coincidem, o filtro de regularização domina, e o
kernel precisa ser quase uma delta. Uma delta não é representável com poucos
pixels: a divisão devolve lóbulos laterais de sinc, ou seja, **ringing**. O
kernel passa a ter regiões negativas que tiram fluxo de um lugar para pôr em
outro.

**Métrica.** Aniano et al. definem W₋ como a integral da parte negativa do
kernel. *"Well-behaved kernels have small W₋ values"* (W₋ = 0,07 no exemplo
canônico), e *"we do not recommend using any kernel with W₋ ≳ 1,2"*. Na nossa
escala, potência negativa = W₋/(1 + 2W₋), então 0,07 ↔ 6,1% e 1,2 ↔ 35,3%.

**O padrão nos nossos kernels** (casamento para F2100W, FWHM 0,643″):

| Fonte | FWHM fonte | Potência negativa | W₋ equivalente |
|---|---:|---:|---:|
| HST (5 bandas) | ~0,08″ | 0,00% | 0,00 |
| F200W | 0,062″ | 2,02% | 0,02 |
| F770W | — | 9,71% | 0,12 |
| F1000W | — | 13,86% | 0,19 |
| F1130W | 0,357″ | 16,67% | 0,25 |
| *IRAC1 → IRAC2* | *r80 2,086″ vs 2,188″* | *~31%* | *~0,8* |

Dentro de uma mesma grade, a potência negativa cresce **monotonicamente** à
medida que a PSF fonte se aproxima da largura do master (MIRI: 9,71% → 13,86% →
16,67%). É o comportamento esperado, e a progressão F1130W → IRAC1 mostra o
regime de falha se aproximando. O F1130W é a banda a acompanhar, mas está bem
abaixo do limite. (O HST tem 0,00% mesmo com FWHM nativo maior que o do F200W
porque seus kernels são construídos na grade de 0,1981″, onde a PSF do HST
ocupa menos de meio pixel: em pixels, a fonte é muito mais estreita que o
alvo, e o kernel fica bem condicionado.)

**O critério de resolubilidade.** A pipeline expressa a condição de Aniano no
espaço real, em pixels da grade de trabalho:

$$\text{alargamento} = \sqrt{r_{80,\text{alvo}}^2 - r_{80,\text{fonte}}^2} \geq \sim 2 \text{ px}$$

| Par | Alargamento | Grade | Em pixels | Decisão |
|---|---:|---:|---:|---|
| IRAC1 → IRAC2 | 0,658″ | 0,75″ | **0,88 px** | Não casar |
| HST azul → F814W (config. só HST) | 0,067–0,079″ | 0,0396″ | **≈ 2 px** | Não casar |
| HST → F2100W | ~0,6″ | 0,1981″ | ≫ 2 px | Casar |
| JWST → F2100W | — | 0,03–0,11″ | ≥ 8 px de margem | Casar |

**Por que não convoluir esses pares.** Convoluir com um kernel de W₋ ~ 0,8 não
deixa a imagem "um pouco mais parecida" com o alvo. Ele cria artefatos, como
fluxo negativo perto de fontes pontuais e amplificação de ruído (o fator de
amplificação é a integral de |K| = 1 + 2W₋ ≈ 2,6), enquanto o ganho em
resolução seria de poucos por cento (as PSFs diferem ~4% em r80). A pipeline
então escreve essas bandas **sem casamento**, com `PSFMATCH = False` e o resíduo
em `PSFRESID`, para que a decisão acompanhe os dados. É o mesmo espírito de
Aniano et al., que só publicam kernels com FWHM_B ≥ FWHM_A/1,35.

### 12.2 Negativos no cubo: o que se esperava

O documento metodológico descreve a "assinatura de saúde" da subtração de céu:
depois dela, todas as bandas convergem para ~45–50% de pixels negativos, e mais
de 50% indicaria sobressubtração. No cubo atual **[medição nova, 2026-10-13]**:

| Banda | % neg. | Banda | % neg. | Banda | % neg. |
|---|---:|---|---:|---|---:|
| F275W | 45,4 | F200W | 46,4 | F770W | 44,2 |
| F336W | 46,4 | F300M | 46,3 | F1000W | 44,3 |
| F438W | 47,6 | F335M | 45,9 | F1130W | 44,4 |
| F555W | 47,6 | F360M | 46,0 | F2100W | 44,2 |
| F814W | 47,0 | | | | |

Pela regra antiga, isso "passa". **Mas a regra só vale quando a área tem muito
céu**, como nos campos amplos do S4G para os quais foi formulada. A análise do
notebook (02/10) já mostrava o sintoma: 45% dos pixels válidos têm mediana
espectral ≤ 0, e **269.746 pixels são negativos nas 13 bandas ao mesmo tempo**.

### 12.3 O problema: a subtração de céu está removendo a galáxia

> **Classificação: BLOQUEADOR para o cubo HST+JWST** (`ngc1087_datacube_sci_1109x1202`).
> Muda todos os valores do cubo, de forma diferente por banda.

Se os negativos fossem ruído de céu independente em cada banda, a fração de
pixels negativos nas 13 bandas ao mesmo tempo seria ~0,46¹³ ≈ **4×10⁻⁵**. A
observada é **31%** (269.746 de 860.668). Os negativos são coerentes entre
bandas, então são **estrutura**, não ruído.

**Medidas [medição nova, 2026-10-13]** nos arquivos em Jy/pixel **antes** da
subtração, dentro da área do cubo:

| Banda | % neg. **antes** da subtração | "Céu" subtraído / mediana dos pixels |
|---|---:|---:|
| F275W | 24,6 | 0,79 |
| F336W | 16,2 | 0,88 |
| F438W | 11,1 | 0,92 |
| F555W | 8,0 | 0,92 |
| F814W | 7,5 | 0,90 |
| F200W | 3,3 | 0,89 |
| F300M–F360M | 0,3–2,8 | 0,89 |
| F770W, F1000W, F1130W | 0,0–0,1 | 0,83–0,84 |
| F2100W | 4,4 | 0,83 |

Distribuição radial dos pixels negativos nas 13 bandas:

| Raio ao centro | Fração negativa nas 13 bandas |
|---|---:|
| 0–0,5′ | 0,000 |
| 0,5–1′ | 0,390 |
| 1–1,5′ | 0,869 |

**O mecanismo.** `sky_level` (`mask_2_0.py:43`) toma a **mediana com
sigma-clipping de toda a área do cubo** como nível de céu. Com o master F2100W, a
área do cubo é a interseção dos campos PHANGS, e para NGC 1087 ela é
**praticamente só disco da galáxia**: antes da subtração, o MIRI tem 0% de
pixels negativos. O "céu" estimado é ~80–90% do pixel mediano da galáxia. A
subtração leva a zero o brilho do pixel mediano do disco, e o disco externo
(> 0,5′) fica negativo em todas as bandas, seguindo o perfil de luz, que é
comum a todas elas.

**Por que é um erro de cor.** O nível subtraído, relativo ao sinal, varia por
banda (sky/p90 vai de 0,15 no F275W a 0,31 no F555W). Subtrair frações
diferentes da luz da galáxia em cada banda distorce a SED, sobretudo nas
regiões fracas, onde vira negativa. É o tipo de erro que o projeto foi
desenhado para eliminar.

**O que fazer (decisão sua; o código não foi alterado):**

- Os produtos PHANGS já chegam com o fundo subtraído (o próprio documento
  metodológico diz isso). A opção mais simples é `sky_subtraction=False` para
  configurações só PHANGS.
- Se for preciso estimar um resíduo de fundo, que seja numa região **fora da
  galáxia**, no mosaico original (muito maior que a área do cubo), e não na
  interseção recortada.
- A regra de "45–50% de negativos" deve sair da documentação como critério de
  saúde, ou ser condicionada à fração de céu na área.
- Depois da mudança: uma execução limpa, as duas travas, e a nova fração de
  negativos por banda registrada.

Isto também tem relação com a pendência da segmentação Sagui
(`session_report`, Sect. 12): pixels com SED negativa em todas as bandas
entrando na segmentação agravariam qualquer agrupamento guiado por brilho
superficial.

---

## 13. Avaliação e pendências

### 13.1 O que eu acho

**A pipeline está metodologicamente sólida no que importa para a dissertação.**
As decisões de casamento de PSF estão ancoradas na literatura (Aniano et al.
2011; Boucaud et al. 2016) e, mais importante, foram tomadas a partir de
medidas: o r80 contra o FWHM, a recusa de casar IRAC1↔IRAC2, a ordem
convoluir→reprojetar, o binning pela média. A convolução e a reprojeção do
cubo HST+JWST passaram em todos os testes, com margens grandes.

**O histórico é, ele mesmo, material de dissertação.** Três padrões se repetiram
e merecem um parágrafo no capítulo de métodos:

1. **Intenção registrada, implementação desviada.** `NATPXAR` escrito e nunca
   lido; travas documentadas como "levanta erro" que só imprimiam; uma métrica
   medida depois da transformação que escondia o que ela procurava. Em todos os
   casos a ideia estava certa e o código não.
2. **Um teste que muda um só lado da comparação mede consistência, não
   correção** (o teste do SIP, Seção 8). Para saber se um componente do WCS está
   certo, é preciso uma referência independente.
3. **Uma trava só está verificada quando foi vista disparando** (B8). A trava de
   grade foi revisada e documentada e nunca podia disparar. Ao adicionar uma
   trava, alimente-a com um caso que precisa falhar.

**Uma observação crítica:** a validação até aqui olhou para **as
transformações** (kernel, reamostragem, unidades), e não para **o produto
final** como dado científico. Por isso a subtração de céu (Seção 12.3) passou: o
número de negativos estava "na faixa esperada". Antes do Capivara, vale um
Check 4 sobre o cubo: fração de pixels negativos em todas as bandas (esperado
~0 numa área só de disco), SED de algumas regiões conhecidas comparada com a
fotometria publicada do PHANGS, e a PSF medida em estrelas de campo nas imagens
convoluídas, que é a evidência independente que falta ao Check 2.

### 13.2 Pendências, pela regra de triagem do projeto

| # | Item | Classe | Afeta o cubo HST+JWST? |
|---|---|---|---|
| 1 | **Subtração de céu remove luz da galáxia** (12.3) | **Bloqueador** | **Sim**: todos os valores, de forma diferente por banda |
| 2 | Execução limpa (esvaziar `convolved_fits`, `reprojected_files`, `datacubes`; `--create_kernel`), as duas travas, congelar e criar tag | Procedimento | Sim (é o que gera o cubo final) |
| 3 | Apagar `ngc1087_datacube_sci_1135x1277` (pré-B8) e os cubos antigos de ngc1433/ngc2903 com `CDELT = 1,0` | Limpeza | Evita usar o cubo errado |
| 4 | Commit do `check1_astrometry.py` | Dissertação | Não, mas o Check 1 não é reproduzível sem ele |
| 5 | Refazer o Check 2 standalone para NIRCam (pós-B8) e salvar a tabela do Check 3 da execução final | Dissertação | Não |
| 6 | Testar o 1,06 com as duas medidas de r50 na mesma grade | Dissertação | Não |
| 7 | `NATPXAR` ausente → erro; trava `abs(soma − 1) > 0,01` | Endurecimento | Não |
| 8 | Restos do B8 (`imgs[0]` → maior escala nativa; preflight e `rediscover_unmatched` por banda; Check 2 "skipped" → erro) | Endurecimento | Não |
| 9 | Atualizar o documento metodológico: laço por banda, trava que nunca disparou antes de 30/09, "sem aliasing" suavizado, tolerância de r50, ordem de interpolação do `reproject_interp`, regra dos 45–50% | Dissertação | Não |
| 10 | C4 (SIP do S4G) + teste com o Gaia | Só se o S4G voltar | Não |
| 11 | Empacotamento (E1–E8) | Backlog | Não |

**Decisões que são suas (e da Paula), não do código:**

- A borda de **~9,9″ por lado** que a dilatação descarta. O "15,5″" dos
  documentos antigos é da configuração HST→IRAC; com o master F2100W são 9,9″.
- Se o `--apply_mask` (máscara de sinal) deve ser ligado. Hoje ele está
  desligado nos dois lugares, depois do commit de 02/10.
- Como tratar o fundo (item 1).

### 13.3 Cuidados ao citar números na dissertação

| Número | Pode citar? | Observação |
|---|---|---|
| 0,079 px (Check 1) | Sim, depois do commit do script | Vale só para HST↔JWST em NGC 1087 |
| Razões r50 do Check 2 | Sim, **com o limite** de que são quase garantidas por construção | |
| 0,00% de potência negativa HST | Sim | Verificado no kernel bruto |
| Potência negativa NIRCam | **Refazer** | Os valores publicados são pré-B8 |
| Check 3, 0,01% | Sim | Salvar a tabela da execução final |
| 0,08 → 1,06 (v1.x) | Sim, como histórico | O 1,06 tem explicação provável não testada (grades misturadas) |
| 0,61% (A2) | Sim, como defeito corrigido | |
| Magnitude do erro NIRCam pré-B8 | **Não** | Só a direção |
| SIP do S4G espúrio | Só como **inferência** | Falta o teste com o Gaia |
| Referências além de Aniano 2011 e Boucaud 2016 | **Verificar no ADS** | Compiladas de memória |

---

## 14. Apêndices

### A. Mapa de arquivos

| Arquivo | Papel |
|---|---|
| `astrovello_cli_2.0.py` | Orquestrador; seleção do master, fatores de bin, estágios |
| `config.py` | Constantes por levantamento (só o que não se mede) |
| `drivers.py` | Comportamento por levantamento: HDU, máscara de inválidos, convolução, Jy |
| `convolution_2_0.py` | Larguras de PSF, `clean_psf`, kernels, binning, convolução |
| `reprojection_2_0.py` | Descoberta por cabeçalho, `reproject_to_reference` |
| `validation.py` | Travas `check_psf_matching`, `check_flux_conservation` |
| `check2_psf_matching.py`, `check3_flux_conservation.py` | Versões standalone para relatório |
| `cube_2_0.py`, `mask_2_0.py` | Montagem do cubo, máscaras, subtração de céu |
| `preflight.py` | Valida a configuração antes de qualquer processamento |
| `diagnostico_etapa1*.py` | Forense da v1.x (manter executável) |
| `pypher_regularisation_test.py` | Investigação do 1,06 |
| `test_sip_hypothesis.py` | O teste de SIP de um lado só (Seção 8) |

### B. Cubos em disco (NGC 1087)

| Arquivo | Data | Status |
|---|---|---|
| `ngc1087_datacube_sci_1109x1202_Jy_per_pixel.fits` | 30/09 | Convolução/reprojeção corretas; **subtração de céu a refazer** (12.3) |
| `ngc1087_datacube_sci_1135x1277_Jy_per_pixel.fits` | 27/09 | **Não usar**: NIRCam na grade MIRI (pré-B8) |

### C. Cabeçalho do cubo atual

```
NAXIS3 × NAXIS2 × NAXIS1 : 13 × 1202 × 1109
CDELT1 = CDELT2          : 3,0807051217028e-05 deg (= 0,11090538″)
PC1_1                    : -0,99999999991961
PIXAREA                  : 0,0123 arcsec²
FILT001..FILT013         : F275W … F2100W, ordenadas por comprimento de onda
```

### D. Referências centrais

- **Aniano, Draine, Gordon & Sandstrom (2011), PASP 123, 1218**, verificado no
  PDF. Seções 2 (existência do kernel), 4.2–4.14 (reamostragem, centragem,
  corte em 99,9% da energia), 5 (métricas D e W±), 7 (uso: kernel na grade da
  imagem, centrado, normalizado, imagem em brilho superficial; W₋ ≲ 1,2).
- **Boucaud et al. (2016), A&A 596, A63** (PyPHER), verificado no PDF.
  Sect. 2.1 (regularização de Wiener), Alg. 1 (`N = N_b; p = p_b`).
- Fruchter & Hook (2002, PASP 114, 144): drizzle, ruído correlacionado.
- Sheth et al. (2010); Muñoz-Mateos et al. (2015): S4G.
- Lee et al. (2022): PHANGS-HST. Williams et al. (2024): PHANGS-JWST.
- Shupe et al. (2005): convenção SIP.

*(As referências além de Aniano e Boucaud devem ser conferidas no ADS antes de
citar.)*
