# AsTrovello 2.0: levar a v2 para a `main`, publicar no PyPI e documentar

Roteiro passo a passo, escrito em 2026-10-20 para ser seguido depois, em
outro computador (onde estão as chaves do PyPI e do TestPyPI). Documento
interno: não vai para a `main` (Parte 2).

Ordem geral:

1. commit e push do branch `v2-dev/package_adaptation` (feito ou em andamento);
2. ajustes finais no `pyproject.toml` e no `README.md` (Parte 1);
3. v2 na `main`, só com o README de documentação (Parte 2);
4. tag `v2.0.0` e publicação no PyPI a partir da `main` (Partes 3 e 4);
5. site de documentação, DOI e demais passos depois da publicação (Parte 5).

**Estado em 2026-10-20:** os passos 1 a 4 estão feitos. A 2.0.0 está em
https://pypi.org/project/astrovello/2.0.0/. O passo 5 está em aberto.

> **DOIs do Zenodo: não confundir** (registrados em 2026-10-22, detalhes na 5.1)
>
> | DOI | O que é | Onde usar |
> |---|---|---|
> | `10.5281/zenodo.23569095` | **conceito**: todas as versões; abre sempre a mais recente | README, badge, `CITATION.cff` |
> | `10.5281/zenodo.23569096` | **versão 2.0.0**: aponta para a 2.0.0 para sempre | dissertação (com o commit `29ac4c1`) |
>
> - Os números são **consecutivos** (`…095` e `…096`): é fácil trocar um
>   pelo outro ao copiar. Confira o final antes de colar.
> - Hoje os dois abrem a mesma página, porque a 2.0.0 é a única versão.
>   Quando sair a próxima (ex.: 2.1.0), o `…095` passa a abrir a nova, e o
>   `…096` continua na 2.0.0.
> - Cada versão nova ganha um DOI de versão próprio, gerado pelo Zenodo ao
>   publicar o Release. O DOI do conceito nunca muda.
> - Badge do README (aponta para o conceito):
>   `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23569095.svg)](https://doi.org/10.5281/zenodo.23569095)`

---

## 0. O que já foi verificado (2026-10-20)

- **O nome `astrovello` no PyPI é seu.** Lá estão as versões 0.1.0, 0.1.1 e
  0.1.2, que são a v1. A v2 sai como **2.0.0 no mesmo projeto**: não há
  projeto novo a criar. No TestPyPI o nome também existe (provavelmente seu,
  da v1).
- **O build funciona.** `python -m build` gera `astrovello-2.0.0.tar.gz` e
  `astrovello-2.0.0-py3-none-any.whl`, e `twine check` passa nos dois.
- **O pacote leva só** `src/astrovello/`, `README.md`, `LICENSE` e
  `pyproject.toml`. Documentos internos, `scripts/`, `notebooks/` e `tests/`
  nunca entram no PyPI, de qualquer branch.
- **Instalação do zero:** num ambiente vazio, sem conda, o pip baixou todas as
  dependências do PyPI. Ficaram instalados os comandos `astrovello`,
  `astrovello-check-psf`, `astrovello-check-flux` e o executável `pypher`, e
  `astrovello --preflight_only` rodou até "All checks passed".
- **Único aviso do build:** `project.license as a TOML table is deprecated`,
  aceito até 2027-02-18. Corrigido no ajuste 1.2.

---

## Parte 1: ajustes antes de publicar

> **Feito em 2026-10-20**, no branch `v2-dev/package_adaptation`, depois do
> commit `d63682e`. O build passou a sair sem nenhum aviso, o `twine check`
> passa, e os metadados mostram `License-Expression: MIT` e os dois
> `Project-URL`. As três URLs (logo, repositório, issues) respondem HTTP 200.
> Além do previsto, saíram do README as linhas de `scripts/forensics/` e
> `notebooks/` na seção "Project structure", porque esses diretórios não
> estarão na `main`. O texto abaixo fica como registro.

O PyPI **não aceita reenviar a mesma versão**. Uma versão publicada pode ser
retirada da listagem ("yank"), mas não substituída. Por isso estes ajustes
vêm antes.

### 1.1 Logo do README

`README.md` usa `<img src="AsTrovello_logo.png">`. Há dois problemas:

- caminho relativo não funciona na página do PyPI;
- o arquivo só existe na `main`, não no branch v2.

Troque pelo link absoluto do arquivo na `main`:

```html
<img src="https://raw.githubusercontent.com/andretrovello/AsTrovello/main/AsTrovello_logo.png" alt="AsTrovello Logo" width="300">
```

### 1.2 Licença no formato novo (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=77.0"]          # o formato novo exige >= 77
build-backend = "setuptools.build_meta"

[project]
...
license = "MIT"                          # era: license = {text = "MIT"}
license-files = ["LICENSE"]
```

### 1.3 Links na página do PyPI (`pyproject.toml`)

```toml
[project.urls]
Repository = "https://github.com/andretrovello/AsTrovello"
Issues = "https://github.com/andretrovello/AsTrovello/issues"
# Documentation = "https://andretrovello.github.io/AsTrovello/"   # quando o site existir
```

### 1.4 README

- A instalação principal passa a ser `pip install astrovello`. Mantenha a
  instalação pelo GitHub como alternativa para a versão em desenvolvimento.
- A última linha cita `astrovello_pipeline_documentation.md`, que não estará
  na `main`. Remova a linha, ou troque pelo link do site quando ele existir.

Depois dos ajustes, confira o build (Parte 3.3) **sem** enviar nada. Faça o
commit e o push no branch v2.

---

## Parte 2: levar a v2 para a `main`

> **Feito em 2026-10-20.**
> - A tag `v1.0` está em `6a600b0`, publicada.
> - O merge foi preparado na `main` local, no commit `d95d4b7 AsTrovello 2.0`.
> - **O push direto para a `main` foi recusado**:
>   `GH006: Protected branch update failed ... Changes must be made through a
>   pull request`. O commit foi então enviado como `release/v2.0.0`, e o
>   PR #1 foi integrado com "Merge pull request" (merge commit). A `main` está
>   em `45e9cb9`.
> - Conferido: a página inicial do GitHub mostra só `.gitignore`,
>   `AsTrovello_logo.png`, `LICENSE`, `README.md`, `pyproject.toml` e `src`.
>   `git merge-base main v2-dev/package_adaptation` = `424adba` (a ponta do
>   branch v2), então os próximos releases são merges normais (2.4).

### 2.0 Decisões (tomadas em 2026-10-20)

| Decisão | Escolha |
|---|---|
| Qual branch vai para a `main` | **`v2-dev/package_adaptation`, direto** |
| Nome da tag da v1 | **`v1.0`** |
| `scripts/forensics/`, `notebooks/`, `tests/` na `main`? | **fora** |
| Reunir os relatórios numa pasta `dev_notes/` no branch v2? | não decidido; opcional, facilita os próximos releases (2.4) |

Abaixo, `<V2>` = `v2-dev/package_adaptation`.

### 2.1 Por que esta parte exige cuidado

**A v2 e a `main` não têm nenhum commit em comum.** O branch v2 nasce de
`68e2c06 first_commit`, e a `main` de `9b90a5c first commit`.
`git merge-base main <V2>` não retorna nada. Consequências:

- um `git merge` comum é recusado ("refusing to merge unrelated histories");
- com `--allow-unrelated-histories`, todo arquivo presente nos dois lados vira
  conflito: `README.md`, `pyproject.toml`, `.gitignore` e
  `src/astrovello/{__init__, astrovello_cli, convolution, cube, mask, units, utils}.py`;
- a `main` carrega arquivos da v1 que precisam sair: `dist/`,
  `src/astrovello.egg-info/`, `src/astrovello/alignment.py`,
  `src/astrovello/batch_runner.py`, `.Rhistory` e `.pyproject.toml`.

O procedimento abaixo resolve tudo a favor da v2, preserva o histórico das
duas versões e **não** usa `push --force`.

**Ensaiado em 2026-10-20** numa cópia local do repositório, com o estado do
branch v2 a ser commitado. O merge deixou zero conflitos pendentes. A `main`
ficou só com `.gitignore`, `AsTrovello_logo.png`, `LICENSE`,
`pyproject.toml`, `README.md` e `src/`, com `src/`, `pyproject.toml` e
`README.md` idênticos aos da v2. O cenário da 2.4 também foi ensaiado e
funciona como descrito.

### 2.2 Preservar a v1 com uma tag

```bash
git checkout main
git pull origin main
git tag v1.0 main               # ou o nome escolhido
git push origin v1.0
```

Assim a v1.0 fica acessível para sempre no GitHub (aba Tags/Releases).

### 2.3 Merge da v2 na `main`

```bash
git checkout main
git merge --allow-unrelated-histories --no-commit -X theirs <V2>
```

`-X theirs` resolve os conflitos de conteúdo a favor da v2 (no ensaio, não
sobrou nenhum). Se ainda sobrar algum arquivo em conflito (`git status` mostra `both added` ou
`both modified`):

```bash
git checkout --theirs -- <arquivo>
git add <arquivo>
```

Remova os arquivos da v1 que a v2 não usa:

```bash
git rm -rf --ignore-unmatch dist src/astrovello.egg-info \
    src/astrovello/alignment.py src/astrovello/batch_runner.py \
    .Rhistory .pyproject.toml
```

Remova os documentos e o material interno. Esta é a lista do branch v2 em
2026-10-20; confira com `ls -a` se surgiram outros:

```bash
git rm -rf --ignore-unmatch \
    CLAUDE.md .claude \
    astrovello_pipeline_documentation.md \
    audit_evaluation.md \
    fix_verification_2026-09-28.md fix_verification_2026-09-29.md fix_verification_2026-09-30.md \
    session_report_audit_and_validation.md \
    per_band_grid_report.md \
    package_adaptation_report.md \
    historico_pipeline_astrovello.md \
    instruction_package_creation.md
# decidido em 2.0: também fora da main
git rm -rf --ignore-unmatch scripts notebooks tests
```

O `-f` é necessário. Esses arquivos acabaram de entrar pelo merge e ainda
não estão em nenhum commit da `main`, e sem o `-f` o `git rm` recusa
("changes staged in the index"). Como o comando é tudo-ou-nada, sem o `-f`
ele não remove **nenhum** arquivo.

**Mantenha:** `README.md`, `LICENSE`, `pyproject.toml`, `.gitignore`,
`AsTrovello_logo.png` (vem da `main` e é usado pelo README) e
`src/astrovello/`.

**Confira antes do commit:**

```bash
git status
ls -a                                   # só o que deve aparecer na main
git diff --cached --stat <V2> -- src pyproject.toml README.md
# não deve imprimir nada: o código, o pyproject e o README da main ficam
# idênticos aos da v2. Se aparecer algum arquivo de src/, é um arquivo da v1
# cujo nome não existe na v2: remova-o com git rm -f.
```

Commit:

```bash
git commit -m "AsTrovello 2.0"
```

**A `main` do GitHub era protegida e só aceitava pull request** (até
2026-10-22, quando a proteção foi retirada). Um `git push origin main` era
recusado com `GH006`. O fluxo por PR continua sendo o adotado. Leve o commit
por PR:

```bash
git push origin main:release/v2.0.0     # o conteúdo da main local vai para um branch novo no GitHub
```

1. No GitHub: **Compare & pull request**, com base `main` ← compare
   `release/v2.0.0`. Depois, **Create pull request**.
2. No fim da aba *Conversation* do PR, use o botão verde **Merge pull
   request** (é o "Create a merge commit") e **Confirm merge**. **Não use**
   "Squash and merge" nem "Rebase and merge". Só o merge commit mantém a v2
   na história da `main`, e é isso que faz os próximos releases serem merges
   normais. Se o botão mostrar outro método, a seta ▾ ao lado troca.
3. **Delete branch** apaga só o `release/v2.0.0` no GitHub. Os commits já
   estão na `main`, e o PR permite "Restore branch".
4. No terminal:

   ```bash
   git checkout main
   git pull origin main              # fast-forward para o merge do PR
   git status -sb                    # "## main...origin/main", sem ahead/behind
   git fetch --prune                 # limpa a referência ao branch apagado
   git checkout v2-dev/package_adaptation
   ```

Abra o repositório no GitHub e confira se a página inicial mostra só o
README, a licença, o `pyproject.toml` e o `src/`.

### 2.4 Próximos releases (v2 → `main`)

Depois do 2.3, os dois lados passam a ter um ancestral comum, e o merge vira
o normal. Prepare o release num branch criado a partir dela e leve-o por PR,
como no 2.3. A `main` não é mais protegida (desde 2026-10-22), então nada
impede um push direto ou um force push: o PR é uma escolha, não uma
exigência do GitHub. Nunca dê push para a `main` a partir de um branch
`v2-dev/*`, que levaria os documentos internos junto.

```bash
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z
git merge --no-commit <V2>
```

- Documento interno que **não** mudou no branch v2: continua apagado na
  `main`, sem nenhuma ação.
- Documento interno que **mudou** (o `CLAUDE.md`, quase sempre): conflito do
  tipo "modified in <V2>, deleted in HEAD". Resolva apagando de novo:

  ```bash
  git rm -rf --ignore-unmatch <arquivo>     # ex.: CLAUDE.md, ou CLAUDE.md dev_notes
  git commit
  ```

- Confira (`git ls-files | cut -d/ -f1 | sort -u`), depois
  `git push origin release/vX.Y.Z`, PR para a `main` com **Merge pull
  request**, e os passos 3 e 4 do 2.3.

Os documentos internos continuam públicos nos branches `v2-dev/*` e no
histórico: a `main` só deixa de mostrá-los na página inicial.

---

## Parte 3: publicar no PyPI

> **Feito em 2026-10-20**, no WSL do Dell, a partir da `main` em `29ac4c1`.
> Os dois commits depois do `45e9cb9` só trocam o `AsTrovello_logo.png`,
> e `src/`, `pyproject.toml` e `README.md` ficaram idênticos.
> - Tag `v2.0.0` em `29ac4c1`, enviada ao GitHub.
> - Build sem avisos e `twine check` PASSED nos dois arquivos. O wheel
>   instalou e rodou (`astrovello --help`) num ambiente conda novo.
> - **TestPyPI pulado** (3.6): o `~/.pypirc` do WSL só tinha a seção
>   `[pypi]`. O `twine check` e a instalação local cobriram o que o ensaio
>   verificaria. Use o TestPyPI quando mudar a forma do pacote (dependências,
>   build, README com HTML ou imagens).
> - Upload com o token de 2026-05 do `~/.pypirc` do WSL, que ainda valia.
>   A permissão do arquivo foi corrigida de `644` para `600`.
> - Verificação final: `pip install astrovello` num ambiente novo instalou a
>   2.0.0, e o `astrovello --help` funcionou. A API do PyPI mostra
>   `License-Expression: MIT` e os links Repository e Issues.

Faça a partir da `main`, já com a v2 e os ajustes da Parte 1. A página do
PyPI mostra o README do momento do build.

### 3.1 Contas e chaves

- **PyPI** (https://pypi.org) e **TestPyPI** (https://test.pypi.org) são
  sites separados, com contas e tokens separados. Use as contas da v1, que
  são as donas do projeto `astrovello`.
- A autenticação em dois fatores é obrigatória nos dois.
- Crie um **API token** em cada site: Account settings → API tokens → Add
  API token. Escopo: só o projeto `astrovello`, e não a conta inteira. O token
  começa com `pypi-` e só é mostrado uma vez.

### 3.2 Guardar as chaves no computador da publicação

Arquivo `~/.pypirc` (na pasta pessoal, **nunca** dentro do repositório):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AAAA...           # token do PyPI

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-BBBB...           # token do TestPyPI
```

```bash
chmod 600 ~/.pypirc
```

Sem esse arquivo, o `twine` pede as credenciais a cada envio. O usuário é
literalmente `__token__`, e a senha é o token.

**O `~` muda com o terminal.** No WSL, `~/.pypirc` é
`/home/andretrovello/.pypirc`. No Git Bash do Windows, é
`C:\Users\dedet\.pypirc`, que não existe. Rode o `twine` no terminal onde o
arquivo está, ou deixe o twine pedir o token e cole na hora. Na hora de colar,
nada aparece na tela.

**Token ≠ recovery codes.** Os arquivos `*-Recovery-Codes*.txt` são os códigos
de recuperação do 2FA, para entrar na conta sem o app autenticador. Eles não
servem para o `twine`. Guarde os dois num gerenciador de senhas (Bitwarden,
1Password ou KeePassXC), nunca no repositório nem em `.txt` solto numa nuvem.
Prefira um token por computador, com escopo só do projeto `astrovello`: se uma
máquina vazar, você revoga só aquele token.

### 3.3 Preparar o outro computador

Os dados (`Input/`, `Output/`) não são necessários para publicar, só o
código.

```bash
git clone git@github.com:andretrovello/AsTrovello.git     # ou git pull, se já existir
cd AsTrovello
git checkout main
git pull origin main

conda activate <ambiente>          # qualquer Python >= 3.10
pip install --upgrade build twine
```

### 3.4 Tag da versão

```bash
git checkout main
git pull origin main
grep '^version' pyproject.toml      # deve dizer 2.0.0
git tag v2.0.0                      # feito em 29ac4c1 (ponta da main)
git push origin v2.0.0
```

A proteção da `main` não impede o push de tags (o `v1.0` foi enviado assim).

### 3.5 Build

```bash
rm -rf dist build src/*.egg-info
python -m build
twine check dist/*                  # os dois devem dar PASSED
ls dist/                            # astrovello-2.0.0.tar.gz e astrovello-2.0.0-py3-none-any.whl
```

`dist/`, `build/` e `src/*.egg-info` estão no `.gitignore`: não são
commitados.

### 3.6 Ensaio no TestPyPI (recomendado)

```bash
twine upload --repository testpypi dist/*
```

Confira a página: https://test.pypi.org/project/astrovello/ (README, logo,
links, licença).

Teste a instalação num ambiente limpo. O TestPyPI não tem as dependências,
por isso o segundo índice:

```bash
python -m venv /tmp/teste_astrovello
/tmp/teste_astrovello/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    astrovello==2.0.0
/tmp/teste_astrovello/bin/astrovello --help
```

### 3.7 Publicação

**Passo sem volta.** Confira antes que `dist/` tem só os dois arquivos da
versão certa: `twine upload dist/*` envia tudo o que estiver lá.

```bash
ls dist/
twine upload --non-interactive dist/*
```

Com `--non-interactive`, o twine falha com `403` se o token não servir, em vez
de ficar parado pedindo senha. Se for colar o token na hora, tire a opção.

Página: https://pypi.org/project/astrovello/

### 3.8 Verificação final

```bash
python -m venv /tmp/teste_final
/tmp/teste_final/bin/pip install astrovello
/tmp/teste_final/bin/pip show astrovello          # Version: 2.0.0
/tmp/teste_final/bin/astrovello --help
```

Com conda, a mesma coisa. Crie o ambiente **com** Python, senão o `pip` que
roda é o de outro ambiente e o teste não vale:

```bash
conda create -n teste_pypi python=3.11 -y
conda activate teste_pypi
which python pip                    # os dois devem estar em envs/teste_pypi/
pip install astrovello
astrovello --help
conda deactivate && conda env remove -n teste_pypi
```

A v1 continua disponível com `pip install astrovello==0.1.2`.

---

## Parte 4: versões futuras

1. Suba a versão em `pyproject.toml` (`2.0.1` para correções, `2.1.0` para
   funcionalidades novas, `3.0.0` para mudanças incompatíveis).
2. Leve as mudanças para a `main` (2.4).
3. Tag `vX.Y.Z`, build, check e upload (3.4 a 3.7).

### Automação opcional (trusted publishing)

Uma GitHub Action pode publicar sozinha a cada tag `v*` enviada ao GitHub,
sem token guardado em lugar nenhum:

1. No PyPI: projeto `astrovello` → Settings → Publishing → Add a new
   publisher (GitHub). Preencha owner `andretrovello`, repository
   `AsTrovello`, workflow `publish.yml` e environment `pypi`.
2. No repositório, crie `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install build && python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Depois disso, `git push origin vX.Y.Z` publica a versão. Confira o
procedimento atual na documentação do PyPI ("Trusted Publishers") antes de
configurar.

---

## Parte 5: depois da publicação

Planejado em 2026-10-20. Estado em 2026-10-22: 5.1 feito (incluindo os
metadados no Zenodo); 5.2 feito (na `main` pelo PR #2). Depois vêm o site
(5.3) e o resto (5.4).

### 5.1 DOI pelo Zenodo + GitHub Release

> **Feito em 2026-10-22.**
> - Release `AsTrovello 2.0.0` publicado no GitHub a partir da tag `v2.0.0`
>   (`29ac4c1`), com as release notes em inglês. Era o primeiro Release do
>   repositório, por isso a opção "Set as the latest release" não aparece.
> - Registro: https://zenodo.org/records/23569096
> - **DOI da versão 2.0.0: `10.5281/zenodo.23569096`** (dissertação).
> - **DOI do conceito: `10.5281/zenodo.23569095`** (README e `CITATION.cff`).
> - A descrição foi importada do Release e renderizada corretamente.
> - Data de lançamento: **2026-10-20** em todos os lugares (notes,
>   `CITATION.cff`). O upload no PyPI foi 2026-10-21 02:41 UTC, ou seja,
>   2026-10-20 23:41 em Brasília.
> - **Metadados revisados em 2026-10-22.** O Zenodo os montou a partir da
>   conta do GitHub (título `andretrovello/AsTrovello: AsTrovello 2.0.0`,
>   autor `André Trovello` sem afiliação, sem palavras-chave, data
>   2026-10-22). Corrigidos em **Edit** → **Publish** (DOI e arquivo não
>   mudam). Estado conferido pela API:
>   - título `AsTrovello 2.0.0`; data de publicação `2026-10-20`;
>   - autor `Trovello, André Almeida`, ORCID `0009-0007-2118-3749`,
>     afiliação "Instituto de Astronomia, Geofísica e Ciências Atmosféricas,
>     Universidade de São Paulo" (texto livre, igual ao `CITATION.cff`);
>   - palavras-chave: PSF matching, datacube, SED fitting, PHANGS, HST,
>     JWST, Spitzer, S4G, image convolution, galaxies, multi-wavelength
>     imaging, astronomy software;
>   - sem contribuidores. A orientadora pode ser incluída depois, em
>     **Contributors**, role *Supervisor*. Cada versão tem metadados
>     próprios: nas próximas, isso teria de ser refeito à mão, ou ir num
>     `.zenodo.json`.
> - Nos próximos Releases o Zenodo deve preencher os metadados a partir do
>   `CITATION.cff`. Confira o registro depois de cada um.
> - O badge (passo 5) entrou na `main` pelo PR #2, junto do 5.2.

O Zenodo arquiva cada Release do GitHub e dá um **DOI** para ele. Na
dissertação, você cita o DOI da versão exata que gerou o cubo.

1. Entre em https://zenodo.org com a conta do GitHub.
2. Menu da conta → **GitHub** → ligue a chave do repositório `AsTrovello`.
   **Faça isto antes do passo 3**: o Zenodo só arquiva Releases criados
   depois que a chave está ligada.
3. No GitHub: aba **Releases** → **Draft a new release** → escolha a tag
   `v2.0.0` (já existe) → título `AsTrovello 2.0.0` → notas (o que mudou em
   relação à v1: JWST, PSF matching por banda, gates de validação, pacote pip)
   → **Publish release**.
4. Em alguns minutos o DOI aparece no Zenodo. Ele dá dois DOIs:
   - **DOI da versão** (só a 2.0.0): use na dissertação.
   - **DOI do conceito** (sempre a versão mais nova): use no README.
5. Coloque o badge do DOI do conceito no README (o Zenodo dá o Markdown).
   Ele entra na `main` no próximo release (2.4).

Releases futuros: crie a tag (3.4) e depois o Release no GitHub. O Zenodo
gera o DOI da versão nova sozinho.

### 5.2 `CITATION.cff`

Arquivo na raiz do repositório. Com ele, o GitHub mostra o botão **Cite this
repository**, com APA e BibTeX prontos. O GitHub lê o arquivo do branch
padrão, então ele **precisa estar na `main`**: entra pelo PR do próximo
release e não deve ir para a lista de remoção da 2.3/2.4.

> **Feito em 2026-10-22** no branch `release/citation-doi` (commit
> `e01026d`, criado a partir da `main` em `29ac4c1`, já no GitHub). Leva o
> `CITATION.cff` e, no README, o badge do DOI do conceito e uma seção
> *Citing*. Validado com `cffconvert --validate` (schema 1.2.0). Integrado
> na `main` pelo PR #2 com **Merge pull request** (merge commit `7aa7a7a`);
> o branch foi apagado. Conferido: a árvore da `main` é idêntica à do
> `e01026d`, o badge e o link do DOI do conceito respondem.

O arquivo como está na `main`:

```yaml
cff-version: 1.2.0
title: AsTrovello
message: 'If you use AsTrovello, please cite it as below.'
type: software
authors:
  - family-names: Trovello
    given-names: André Almeida
    affiliation: >-
      Instituto de Astronomia, Geofísica e Ciências
      Atmosféricas, Universidade de São Paulo
    orcid: 'https://orcid.org/0009-0007-2118-3749'
abstract: >-
  Multi-survey (PHANGS-HST, PHANGS-JWST, S4G) PSF-matched
  datacubes in Jy/pixel for spatially-resolved SED fitting.
keywords:
  - PSF matching
  - datacube
  - SED fitting
  - PHANGS
  - JWST
  - Spitzer
  - image convolution
doi: 10.5281/zenodo.23569095
identifiers:
  - type: doi
    value: 10.5281/zenodo.23569095
    description: Concept DOI (all versions, resolves to the latest)
  - type: doi
    value: 10.5281/zenodo.23569096
    description: Version 2.0.0
repository-code: 'https://github.com/andretrovello/AsTrovello'
license: MIT
version: 2.0.0
date-released: '2026-10-20'
```

**Por que o `doi` principal é o do conceito, e não o da versão** (o plano
original era o da versão): o Zenodo só gera o DOI de uma versão quando o
Release é publicado, depois do commit que entra na `main`. Com o DOI da
versão no campo principal, o arquivo estaria sempre um release atrasado. O
DOI do conceito nunca muda. Os DOIs de versão vão em `identifiers`. Para a
dissertação nada muda: cite o DOI da versão.

**No próximo release:** acrescente às `keywords` as cinco que foram postas
no Zenodo em 2026-10-22 e não estão no arquivo: `HST`, `S4G`, `galaxies`,
`multi-wavelength imaging`, `astronomy software`. Assim o arquivo e o
registro ficam com a mesma lista.

**A cada release:** atualize `version` e `date-released` no PR do release.
Depois de publicado, acrescente o DOI da versão nova em `identifiers` no
release seguinte (opcional; a página do Zenodo lista todas as versões).

Para validar antes do commit: `pip install cffconvert`, depois
`cffconvert --validate` na raiz do repositório (ou o validador em
https://citation-file-format.github.io).

### 5.3 Site de documentação (MkDocs Material + GitHub Pages)

**Por que MkDocs Material e não Sphinx:** é Markdown, que você já escreve;
basta um `mkdocs.yml` e uma pasta `docs/`; e você vê o site enquanto edita
(`mkdocs serve`). O Sphinx (com Read the Docs) é o padrão histórico em
astronomia (astropy, photutils) e tem o `autodoc` mais maduro, mas vale a pena
se a referência da API virar o centro da documentação. Hoje o centro é o
método. A referência da API pode vir depois, com `mkdocstrings`.

O site fica em `https://andretrovello.github.io/AsTrovello/`.

#### Decisão pendente: onde ficam as fontes do site

| Opção | Prós | Contras |
|---|---|---|
| **`docs/` + `mkdocs.yml` na `main`** (recomendado) | simples; o site sempre descreve o código publicado | toda correção do site passa por PR na `main` |
| fontes no branch v2, publicar só o site gerado | a `main` continua só com o pacote | mais trabalho; o site pode descrever código ainda não lançado |

Documentação pública não é documento interno, então pôr `docs/` na `main`
não contraria a regra da Parte 2. Se escolher a `main`: **não** inclua `docs/`
nem `mkdocs.yml` no `git rm` da 2.3/2.4, e acrescente `site/` ao `.gitignore`
(é a pasta que o `mkdocs build` gera).

#### Montar localmente

Num branch novo a partir do branch v2, por exemplo `v2-dev/docs`:

```bash
pip install mkdocs-material
mkdocs new .                        # cria mkdocs.yml e docs/index.md
mkdocs serve                        # abre em http://127.0.0.1:8000, recarrega ao salvar
```

`mkdocs.yml` inicial:

```yaml
site_name: AsTrovello
site_url: https://andretrovello.github.io/AsTrovello/
repo_url: https://github.com/andretrovello/AsTrovello
theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy
nav:
  - Home: index.md
  - Installation: installation.md
  - Quickstart: quickstart.md
  - How it works: method.md
  - CLI reference: cli.md
  - Output cube: output.md
  - Validation: validation.md
  - Limitations: limitations.md
  - Changelog: changelog.md
  - Citing: citing.md
markdown_extensions:
  - admonition
  - pymdownx.superfences
  - pymdownx.arithmatex:
      generic: true
```

#### O que vai em cada página

| Página | Conteúdo | Fonte |
|---|---|---|
| Installation | `pip install astrovello`; PyPHER no `PATH`; Python >= 3.10 | README |
| Quickstart | layout de `Input/<SURVEY>/galaxies/<galaxy>/` e `PSF/`; um comando completo; o que sai em `Output/` | CLAUDE.md Sect. 1 |
| How it works | as quatro etapas; convolução antes da reprojeção; master por r80; grid de convolução por banda; binning pela média | doc metodológico |
| CLI reference | todos os flags (`--surveys`, `--create_kernel`, `--preflight_only`, `--skip_checks`, `--mode`...) | `astrovello --help` |
| Output cube | Jy/pixel, WCS, `PIXAREA`; chaves `PSFTARGT`, `PSFMATCH`, `PSFRESID`, `NATPXAR`, `BINFACT` | `cube.py`, headers |
| Validation | os dois gates, limiares, o que garantem **e o que não garantem** (limite do Check 2) | `validation.py`, CLAUDE.md Sect. 4a |
| Limitations | reprojeção, não registro astrométrico; só HST↔JWST verificado; C4 para cubos com S4G | CLAUDE.md topo e C4 |
| Changelog | 2.0.0 contra a v1 (0.1.x) | |
| Citing | DOI (5.1) e BibTeX (5.2) | |

**Cuidado:** o `astrovello_pipeline_documentation.md` mistura o método com o
histórico da auditoria. Use-o como **fonte** e reescreva o que o código faz
hoje, sem copiar o arquivo inteiro. Confira cada afirmação contra o código
(CLAUDE.md Sect. 7: o doc é especificação a verificar, não descrição).

#### Publicar com GitHub Actions

1. No GitHub: **Settings → Pages → Source: GitHub Actions**.
2. Crie `.github/workflows/docs.yml`:

```yaml
name: Docs
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install mkdocs-material
      - run: mkdocs build --strict        # falha com link quebrado
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

3. O site sai a cada push na `main` (ou seja, a cada PR integrado), ou à mão
   pela aba **Actions → Docs → Run workflow**.

Confira as versões atuais das actions na documentação do GitHub Pages antes
de configurar.

#### Depois que o site estiver no ar

Descomente no `pyproject.toml`:

```toml
Documentation = "https://andretrovello.github.io/AsTrovello/"
```

O link só aparece no PyPI **na próxima versão** (2.0.1 ou 2.1.0): os
metadados de uma versão publicada não mudam.

### 5.4 Depois

1. **Testes + CI** (defeito E6 do CLAUDE.md). Alguns casos `pytest` sobre
   FITS sintéticos, rodando a cada push por GitHub Actions. O A1 (WCS 4800×
   errado) teria sido pego por uma linha de `assert`. Dá segurança para o C4.
2. **Trusted publishing** (Parte 4). Publicar vira `git push origin vX.Y.Z`,
   sem token guardado.
3. **ASCL** (https://ascl.net): registro de software astronômico, indexado
   pelo ADS. Submissão curta, feita uma vez.
4. **JOSS** (https://joss.theoj.org), depois da dissertação: artigo curto
   revisado sobre o software. Exige documentação, testes e instruções de
   contribuição, ou seja, 5.3 e 5.4.1.

---

## Problemas comuns

| Mensagem | Causa | O que fazer |
|---|---|---|
| `400 File already exists` | essa versão já foi enviada (o PyPI não aceita reenviar) | subir a versão no `pyproject.toml`, refazer o build |
| `403 Invalid or non-existent authentication` | token errado, de outro site (PyPI vs TestPyPI) ou com escopo de outro projeto | conferir o `~/.pypirc` e o token |
| `403 ... isn't allowed to upload to project` | a conta não é dona do projeto | usar a conta da v1 |
| `twine check` falha no README | Markdown/HTML que o PyPI não renderiza | corrigir o README e refazer o build |
| `refusing to merge unrelated histories` | o merge da Parte 2.3 sem a opção | usar `--allow-unrelated-histories` (só na primeira vez) |
| `GH006: Protected branch update failed ... through a pull request` | push direto para a `main` enquanto ela era protegida (até 2026-10-22) | enviar para `release/...` e integrar por PR (2.3) |
| `pypher: command not found` ao rodar a pipeline | ambiente onde o astrovello foi instalado não está ativado | ativar o ambiente; o `pypher` é instalado junto |
| `twine upload` fica parado, sem mensagem | está esperando a senha, num terminal onde não dá para digitar | usar `--non-interactive`, ou rodar num terminal normal e colar o token |
| `twine` pede o token embora o `.pypirc` exista | rodando no Git Bash, cujo `~` é `C:\Users\dedet`, e não no WSL | rodar no WSL, ou colar o token |
| `twine: command not found` | o ambiente ativo não é o que tem o twine | `ls ~/miniconda3/envs/*/bin/twine` mostra onde está |
| o teste instala o pacote no ambiente errado | ambiente conda criado sem `python=...`, então o `pip` é o de outro ambiente | `conda create -n <nome> python=3.11`; conferir com `which python pip` |
