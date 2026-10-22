# Site de documentação do AsTrovello: registro e próximos passos

Sessão de 2026-10-22. Documento interno para estudo: registra o que foi
feito, **por que** cada coisa foi feita e o que falta. Fica em `reports/`
e não vai para a `main`.

---

## 1. Onde estamos

| Item | Estado |
|---|---|
| Release 2.0.0 (PyPI, GitHub, Zenodo) | **concluído**. Ver Sect. 2 |
| Ferramenta do site | **Zensical 0.0.71** (fixada) |
| Ambiente conda | `astrovello-docs` (Python 3.11) |
| Branch do site | `v2-dev/docs`, criada a partir da `v2-dev/package_adaptation` |
| Fase 1: ambiente e esqueleto | **concluída** (commit `8ccf268`) |
| Fase 2: configuração | **feita, falta commitar** (Sect. 6.1) |
| Fases 3 a 5: páginas, publicação, ir ao ar | a fazer (Sect. 6) |
| Endereço futuro do site | https://andretrovello.github.io/AsTrovello/ |

> **Atenção antes do próximo push.** A `v2-dev/docs` foi enviada ao GitHub
> **antes** do `git commit --amend` que corrigiu o e-mail. O GitHub tem o
> commit antigo (`133508b`, e-mail `@example.com`); o computador tem o
> corrigido (`0562d09`) e mais o esqueleto (`8ccf268`). As duas cópias
> divergiram, e um `git push` normal será recusado. Ver Sect. 6.1, passo 3.

---

## 2. O que foi feito hoje antes do site (resumo)

O detalhe está em `reports/instruction_package_creation.md` (Partes 2 e 5).

1. **Release notes da 2.0.0** escritas em inglês e publicadas como GitHub
   Release a partir da tag `v2.0.0`. Release notes descrevem a versão para
   quem usa o pacote: o que ele faz, o que mudou, o que quebrou em relação
   à 0.1.x e as limitações conhecidas (incluindo o C4 do S4G).
2. **Zenodo** ligado ao repositório **antes** do Release. O Zenodo só
   arquiva Releases publicados depois de ligado. Ele gerou dois DOIs:

   | DOI | O que é | Onde usar |
   |---|---|---|
   | `10.5281/zenodo.23569095` | **conceito**: todas as versões, abre sempre a mais recente | README, badge, `CITATION.cff`, site |
   | `10.5281/zenodo.23569096` | **versão 2.0.0**: aponta para a 2.0.0 para sempre | dissertação (com o commit `29ac4c1`) |

   Os números são consecutivos: confira o final antes de colar.
3. **Metadados do Zenodo corrigidos à mão.** O Zenodo lê o `CITATION.cff`
   **de dentro da tag**, no momento do Release, e a tag `v2.0.0` não o
   tinha. Por isso o registro saiu com dados do perfil do GitHub (título
   `andretrovello/AsTrovello: ...`, autor sem "Almeida"). Corrigidos em
   **Edit → Publish**: título, autor com ORCID e afiliação, 12
   palavras-chave, data 2026-10-20. Editar metadados não muda o DOI nem o
   arquivo. Nas próximas versões, a tag já terá o `CITATION.cff`.
4. **`CITATION.cff`** na `main` (PR #2). Ativa o botão *Cite this
   repository* do GitHub. O campo `doi` leva o DOI do **conceito**: o DOI de
   uma versão só existe depois do Release, então não poderia estar no
   arquivo da própria versão. A cada release, atualize `version` e
   `date-released` junto com a versão do `pyproject.toml`.
5. **README da `main` igualado ao da v2** (PR #3). Se os dois lados mudam o
   mesmo arquivo de formas diferentes, o merge do próximo release pode
   juntar as duas versões **sem acusar conflito**. A simulação mostrou duas
   seções *Citing*. Com o mesmo texto nos dois lados, o merge fica limpo.
6. **A `main` deixou de ser protegida** (decisão sua). O fluxo por PR
   continua, por escolha. Nunca dê push para a `main` a partir de uma
   branch `v2-dev/*`, que levaria os documentos internos junto.
7. **Relatórios movidos para `reports/`** (commit `0562d09`). O `CLAUDE.md`
   ficou na raiz, porque é lá que o Claude Code o lê ao abrir uma sessão.

---

## 3. Decisões do site, e por quê

### 3.1 Por que um site

O README cobre instalação e uso básico. O site cobre o que não cabe num
README: o método (por que convolução antes de reprojeção, master por r80,
grid por banda), a referência completa da CLI, o formato do cubo, o que as
validações garantem e o que não garantem, e as limitações. É também
pré-requisito para o JOSS (`instruction_package_creation.md`, 5.4).

### 3.2 Zensical, e não MkDocs

O plano original (5.3) era MkDocs + Material for MkDocs. Mudou porque:

- **MkDocs** está parado: a última versão é a 1.6.1, de agosto de 2024. Um
  MkDocs 2.0 foi anunciado em 2026, com mudanças que quebram compatibilidade
  (inclusive no sistema de plugins).
- **Material for MkDocs** está em modo de manutenção desde novembro de 2025:
  não recebe recursos novos, e o changelog oficial promete correções
  críticas só **até 5 de maio de 2027**.
- **Zensical** é o sucessor, feito pela mesma equipe, com licença MIT. Lê
  `mkdocs.yml` e a configuração do Material, publica no GitHub Pages por
  GitHub Actions e já suporta referência de API (`mkdocstrings`, via
  `api-autonav`).

**Risco:** o Zensical está em **alpha** (0.0.71, com versão nova quase toda
semana). Para um site que é basicamente Markdown, o risco é pequeno,
desde que a versão seja fixada (3.3).

### 3.3 Versão fixada (`zensical==0.0.71`)

`pip install zensical` instala a versão mais recente, que muda toda semana.
Com `==0.0.71`, o site é sempre gerado pela mesma versão, no seu computador
e no GitHub Actions. Atualizar passa a ser uma decisão: trocar o número,
testar com `zensical serve` e commitar.

### 3.4 Ambiente separado (`astrovello-docs`)

O Zensical exige Python ≥ 3.11 e traz dependências próprias. Um ambiente
só para a documentação não interfere no `capivara`, onde o pipeline roda.
Para trabalhar no site:

```bash
conda activate astrovello-docs
cd ~/AsTrovello          # é um atalho para /mnt/c/Users/dedet/Desktop/AsTrovello
git checkout v2-dev/docs
```

### 3.5 Fontes do site na `main` (Opção A)

"Fontes" são os arquivos que **geram** o site: `docs/`, `zensical.toml` e
`.github/workflows/docs.yml`. Eles vão para a `main` porque:

- o GitHub Pages publica, por padrão, a partir da branch principal;
- a `main` só recebe releases, então o site sempre descreve a versão que
  alguém instala com `pip install astrovello`;
- é o padrão dos pacotes de astronomia com site (astropy, photutils).

Documentação pública não é documento interno, então isso não contraria a
regra de manter a `main` limpa. **No `git rm` de cada release, não apague
`docs/`, `zensical.toml` nem `.github/`.**

### 3.6 Uma branch só para o site (`v2-dev/docs`)

Uma branch é uma linha de trabalho dentro do **mesmo** repositório. Nada é
copiado no disco: a pasta é a mesma, e o git só anota os commits na branch
nova. O site fica isolado até ser levado adiante.

---

## 4. Conceitos para estudar

### 4.1 `git status`: as duas colunas

```
A  reports/x.md     ← 1ª coluna: registrado (vai para o commit)
 D x.md             ← 2ª coluna: mudou, mas NÃO registrado
?? docs/            ← arquivo novo que o git ainda não acompanha
R  x.md -> reports/x.md   ← renomeado (o git entendeu que é o mesmo arquivo)
```

- `git add <pasta>` registra o que está **na** pasta. Ao mover arquivos, a
  **saída** do lugar antigo também precisa ser registrada.
- `git add -u -- '*.md'`: `-u` só mexe em arquivos que o git **já conhece**
  (registra modificações e remoções, ignora arquivos novos). Foi o que
  transformou os pares `D` + `A` em `R`.
- `R100` no `git show` = renomeação com conteúdo 100% idêntico.
- **Um assunto por commit.** A mudança para `reports/` e o site são assuntos
  diferentes; em commits separados, o histórico explica o que aconteceu.
- **Prefira listar os caminhos no `git add`** (`git add docs zensical.toml`)
  a usar `git add .`. Assim você sabe o que entra no commit.

### 4.2 Identidade do git e `--amend`

O git do WSL estava com um e-mail de exemplo. Commits com e-mail que não é
da sua conta aparecem no GitHub sem o seu avatar e não contam no seu
perfil. Corrigido de vez com:

```bash
git config --global user.name "André Almeida Trovello"
git config --global user.email "andre.a.trovello@gmail.com"
```

`git commit --amend --reset-author --no-edit` refaz o **último** commit com
o autor novo e a mesma mensagem. Isso cria um commit **novo**, com outro
hash. Se o commit antigo já estava no GitHub, as duas cópias divergem (é o
que aconteceu; ver o aviso na Sect. 1 e a Sect. 6.1). Regra prática:
**amend só antes do push.** Depois do push, ou se faz um commit novo, ou se
reescreve com `--force-with-lease`, e só numa branch que só você usa.

### 4.3 Sincronizar com o GitHub sem conflito

Quando você edita pelo GitHub e algo é commitado no computador:
- antes de mexer localmente, `git pull`;
- depois de commitar localmente, `git push` logo.

Se os dois lados commitaram em arquivos diferentes, um `git pull --rebase`
(ou `git rebase origin/<branch>`) coloca os commits locais depois dos
remotos, sem conflito. Foi o que aconteceu com o seu ajuste do BibTeX no
README.

### 4.4 TOML

O `zensical.toml` (e o `pyproject.toml`) são TOML. Regras que apareceram:

- `#` começa um **comentário**: a linha é ignorada. Valor preenchido com `#`
  na frente não vale.
- `[project]` é um **cabeçalho de seção**. Cada seção só pode ser declarada
  **uma vez**. Quando a documentação mostra `[project]` acima de uma linha,
  ela indica **onde** a linha vai, não que você deva repetir o cabeçalho.
- `[[project.extra.social]]` (colchete duplo) cria **um item de uma lista**
  de seções. Cada bloco é um ícone.
- `{ ... }` é uma **tabela inline** (um dicionário). **Precisa ficar numa
  linha só**; a única quebra permitida é dentro de uma lista `[ ... ]`. Por
  isso o `nav` tem `= [` na linha do título e `] },` fechando junto.
- **Lista vs dicionário:** o menu é uma sequência, e só a lista garante a
  ordem (a especificação do TOML diz que a ordem das chaves de uma tabela
  não tem significado). Por isso o `nav` é uma **lista de pares**, e não um
  dicionário único. A lista também permite misturar itens sem título
  (`"index.md"`), com título e seções.
- Para testar o TOML antes do `serve`:

  ```bash
  python -c "import tomllib; print(tomllib.load(open('zensical.toml','rb'))['project']['nav'])"
  ```

  O erro aponta linha e coluna (ex.: *Invalid value (at line 15, column
  25)*).

**YAML** é outro formato para os mesmos fins, com a estrutura marcada pela
**indentação** (`chave: valor`, `- item`). O `mkdocs.yml`, o `CITATION.cff`
e o workflow `.github/workflows/docs.yml` são YAML. O mesmo `nav` em YAML:

```yaml
nav:
  - Home: index.md
  - Getting Started:
      - Installation: installation.md
```

### 4.5 Caminhos relativos a `docs/`

O conteúdo de `docs/` vira a **raiz** do site; a pasta `docs/` em si não
existe no site publicado:

```
docs/assets/logo.png  →  site/assets/logo.png  →  .../AsTrovello/assets/logo.png
```

Por isso `logo = "assets/logo.png"`, `nav` com `"index.md"` e
`extra_javascript` com `"javascripts/mathjax.js"`, tudo **sem** `docs/`.
(`docs` é o valor padrão de `docs_dir`.) Endereços completos (`https://...`)
não seguem essa regra.

### 4.6 `site/` é gerado e não vai para o git

`zensical build` (e o `serve`) gera o HTML em `site/`. Ela é recriada a cada
build, e na publicação quem a gera é o GitHub Actions. Por isso `site/` está
no `.gitignore`.

### 4.7 `site_url` vs `repo_url`

- `site_url` = onde o **site** fica: `https://andretrovello.github.io/AsTrovello/`.
  O Zensical o usa para links internos, mapa do site e navegação
  instantânea. Errado, o site funciona no `localhost` e quebra publicado.
- `repo_url` = onde está o **código**: `https://github.com/andretrovello/AsTrovello`.
  Faz aparecer o link do GitHub no topo.

### 4.8 Fórmulas: `arithmatex` + MathJax

A extensão `pymdownx.arithmatex` só **marca** as fórmulas no HTML. Quem as
**desenha** no navegador é o MathJax, carregado por `extra_javascript`:
`docs/javascripts/mathjax.js` (configuração) e o MathJax do CDN. Sintaxe nas
páginas: `$$ ... $$` em bloco, `\( ... \)` em linha.

### 4.9 Particularidades do WSL com `/mnt/c`

- **`d?????????` no `ls`**: o WSL às vezes demora a enxergar uma pasta
  renomeada, sobretudo com o VS Code aberto nela. O arquivo existe (o
  Windows mostra). Espere um pouco ou feche e reabra o VS Code.
- **`core.fileMode false`**: o Windows marca todo arquivo em `/mnt/c` como
  executável, e o git contava isso como alteração (25 arquivos "modificados"
  sem mudança de conteúdo). Desligado neste repositório.
- **CRLF**: arquivos editados no Windows têm fim de linha CRLF. Com
  `core.autocrlf = input`, o git guarda LF no repositório; os avisos
  *"CRLF will be replaced by LF"* são normais.

### 4.10 Comandos do Zensical

| Comando | Para quê |
|---|---|
| `zensical new .` | cria o esqueleto (feito uma vez) |
| `zensical serve` | servidor local em http://localhost:8000; recarrega ao salvar; **Ctrl+C** para parar |
| `zensical build --clean` | gera o site em `site/` do zero (é o que o GitHub Actions roda) |

---

## 5. Arquivos do site hoje

```
.github/workflows/docs.yml   publicação (Fase 4: fixar a versão)
docs/
  index.md ... citing.md     10 páginas, só com título (Fase 3)
  markdown.md                exemplo do esqueleto; remover (6.1)
  assets/logo.png            cópia do logo da main
  javascripts/mathjax.js     configuração do MathJax
zensical.toml                configuração do site
```

Configurado no `zensical.toml`: nome, descrição, autor, copyright, os dois
endereços (4.7), `edit_uri`, `nav` com 4 seções e 10 páginas, logo e
favicon, GitHub e PyPI no rodapé, MathJax. Último build de teste: *"No
issues found"*.

---

## 6. Próximos passos

### 6.1 Terminar a Fase 2

1. `conda activate astrovello-docs`, `zensical serve` e o checklist:
   - [ ] menu com 4 seções e 10 páginas;
   - [ ] logo no topo e na aba do navegador;
   - [ ] GitHub e PyPI no rodapé, abrindo os links certos;
   - [ ] a fórmula de teste em *How it Works* formatada;
   - [ ] nenhum aviso no terminal do `serve`.
2. Commit:
   ```bash
   git rm docs/markdown.md        # quando não precisar mais da referência
   git add docs zensical.toml
   git status
   git commit -m "Docs site: navigation, logo, social links, MathJax"
   ```
3. **Push com reescrita** (por causa do aviso da Sect. 1):
   ```bash
   git push --force-with-lease -u origin v2-dev/docs
   ```
   `--force-with-lease` substitui a cópia do GitHub pela sua, **mas só se
   ninguém tiver mexido nela desde o seu último fetch**; senão, recusa. É a
   versão segura do `--force`. `-u` liga a branch local à remota, para os
   próximos `git push`/`git pull` saírem sem argumentos. Pode fazer isso
   porque só você usa a `v2-dev/docs`. Depois, `git status -sb` deve
   mostrar `## v2-dev/docs...origin/v2-dev/docs`, sem ahead/behind.

### 6.2 Fase 3: escrever as páginas

Uma página por vez, em inglês, com revisão. **Regra de ouro:** conferir
cada afirmação contra o **código**, não contra o
`astrovello_pipeline_documentation.md`, que mistura método com histórico de
auditoria (`CLAUDE.md`, Sect. 7: o doc é especificação a verificar).

| Página | Conteúdo | Fontes |
|---|---|---|
| Home (`index.md`) | o que o pacote faz, para quem, as 4 etapas em uma frase cada, links para Installation/Quickstart | README |
| Installation | `pip install astrovello`, Python ≥ 3.10, ambiente conda, PyPHER no `PATH` | README, `pyproject.toml` |
| Quick Start | layout de `Input/<SURVEY>/galaxies/<galaxy>/` e `PSF/`, nomes exatos das pastas, um comando completo, o que sai em `Output/` | README, `CLAUDE.md` Sect. 1 |
| How it Works (`method.md`) | as 4 etapas; convolução antes da reprojeção; master por r80; grid de convolução por banda; binning pela média; conversão de unidades (com fórmulas) | doc metodológico + código |
| CLI Reference | todos os flags e modos | `astrovello --help`, `astrovello_cli.py` |
| Output Cube | Jy/pixel, WCS, `PIXAREA`, chaves `PSFTARGT`, `PSFMATCH`, `PSFRESID`, `NATPXAR`, `BINFACT` | `cube.py`, headers de um cubo real |
| Validation | os dois gates, limiares, o que garantem **e o que não garantem** (o Check 2 usa os mesmos PSFs que geraram o kernel) | `validation.py`, `CLAUDE.md` Sect. 4a |
| Limitations | reprojeção ≠ registro astrométrico; só HST↔JWST verificado; SIP do S4G (C4); borda de ~10″ | `CLAUDE.md` topo e C4, release notes |
| Changelog | 2.0.0 vs 0.1.x | release notes da 2.0.0 |
| Citing | os dois DOIs e quando usar cada um; BibTeX | README (seção Citing), Sect. 2 |

Recursos do Markdown que vão ajudar (ver `docs/markdown.md` antes de
apagá-lo, ou https://zensical.org/docs/): caixas de aviso
(`!!! warning`, `!!! note`) para limitações e cuidados; abas para mostrar o
mesmo comando de jeitos diferentes; blocos de código com botão de copiar
(já ligado).

### 6.3 Fase 4: publicação

1. **Fixar a versão no workflow** (`.github/workflows/docs.yml`):
   - `pip install zensical` → `pip install zensical==0.0.71`;
   - `python-version: 3.x` → `python-version: "3.11"` (igual ao local);
   - em `branches:`, pode remover `master` (só existe `main`).
2. No GitHub: **Settings → Pages → Build and deployment → Source: GitHub
   Actions**.
3. O workflow só roda quando algo entra na **`main`**. Enquanto o site
   estiver só na `v2-dev/docs`, nada é publicado (é o esperado).

### 6.4 Fase 5: ir ao ar

1. Levar a `v2-dev/docs` para a `v2-dev/package_adaptation` (merge), para
   a branch v2 principal ter o site.
2. **Release para a `main`**, pelo roteiro 2.4 de
   `instruction_package_creation.md`: branch `release/...` a partir da
   `main`, `git merge --no-commit` da v2, `git rm` dos documentos internos
   e PR com **Merge pull request**. Com a mudança para `reports/`, o `git rm`
   fica mais simples:
   ```bash
   git rm -rf --ignore-unmatch reports CLAUDE.md scripts notebooks tests .claude .Rhistory
   ```
   (Hoje a v2 versiona, na raiz: `.Rhistory`, `.claude`, `.github`,
   `.gitignore`, `CLAUDE.md`, `LICENSE`, `README.md`, `docs`, `notebooks`,
   `pyproject.toml`, `reports`, `scripts`, `src`, `tests`, `zensical.toml`.
   Confira de novo na hora: algo novo na raiz precisa de uma decisão.)
   e **não** apagar `docs/`, `zensical.toml` nem `.github/`. Antes do commit,
   `git ls-files | cut -d/ -f1 | sort -u` deve mostrar só: `.github`,
   `.gitignore`, `AsTrovello_logo.png`, `CITATION.cff`, `LICENSE`,
   `README.md`, `docs`, `pyproject.toml`, `src`, `zensical.toml`.
3. Depois do merge, aba **Actions** → *Documentation*: o workflow deve
   ficar verde, e o site aparece em https://andretrovello.github.io/AsTrovello/.
4. No `pyproject.toml`, em `[project.urls]`, acrescentar
   `Documentation = "https://andretrovello.github.io/AsTrovello/"`. O link
   só aparece no PyPI na **próxima versão** publicada.
5. Opcional: link do site no README e na descrição do repositório no GitHub
   (**About → Website**).

### 6.5 Pendências fora do site

- **Caminhos antigos depois da mudança para `reports/`:**
  - `CLAUDE.md` cita os relatórios pelo caminho da raiz;
  - `.claude/workflows/verify-astrovello-audit-fixes.js` procura
    `${ROOT}/session_report_audit_and_validation.md` e
    `${ROOT}/astrovello_pipeline_documentation.md` (→ `${ROOT}/reports/...`);
  - `reports/instruction_package_creation.md`: a lista de `git rm` da
    Parte 2 (→ `git rm -r reports`, como em 6.4) e a seção 5.3, que ainda
    descreve MkDocs (→ Zensical, com o motivo da Sect. 3.2).
- **`CITATION.cff` no próximo release:** acrescentar as 5 palavras-chave que
  estão só no Zenodo (`HST`, `S4G`, `galaxies`, `multi-wavelength imaging`,
  `astronomy software`) e atualizar `version`/`date-released`.
- **Contribuidora no Zenodo (opcional):** a orientadora pode ser incluída a
  qualquer momento em **Edit → Contributors**, role *Supervisor*. Cada versão
  tem metadados próprios; para não refazer à mão a cada versão, existe o
  arquivo `.zenodo.json`.

---

## 7. Referências

- Zensical: documentação https://zensical.org/docs/ · compatibilidade com
  MkDocs https://zensical.org/docs/compatibility/mkdocs/ · fórmulas
  https://zensical.org/docs/authoring/math/ · publicação
  https://zensical.org/docs/publish-your-site/ · PyPI
  https://pypi.org/project/zensical/
- Material for MkDocs, changelog (manutenção até 2027-05-05):
  https://squidfunk.github.io/mkdocs-material/changelog/
- TOML (especificação): https://toml.io/en/v1.0.0
- Registro Zenodo da 2.0.0: https://zenodo.org/records/23569096
- Roteiro de release e publicação: `reports/instruction_package_creation.md`
