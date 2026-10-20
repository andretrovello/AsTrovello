# AsTrovello 2.0: levar a v2 para a `main` e publicar no PyPI

Roteiro passo a passo, escrito em 2026-10-20 para ser seguido depois, em
outro computador (onde estão as chaves do PyPI e do TestPyPI). Documento
interno: não vai para a `main` (Parte 2).

Ordem geral:

1. commit e push do branch `v2-dev/package_adaptation` (feito ou em andamento);
2. ajustes finais no `pyproject.toml` e no `README.md` (Parte 1);
3. v2 na `main`, só com o README de documentação (Parte 2);
4. tag `v2.0.0` e publicação no PyPI a partir da `main` (Partes 3 e 4).

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

### 2.0 Decisões a tomar antes

| Decisão | Opções |
|---|---|
| Qual branch vai para a `main` | `v2-dev/package_adaptation` direto, ou primeiro para `v2-dev/main` e dele para a `main` (mais organizado) |
| Nome da tag da v1 | `v1.0`? (o `pyproject` da v1 dizia 0.1.2) |
| `scripts/forensics/`, `notebooks/`, `tests/` na `main`? | sugestão: fora, como os documentos internos |
| Reunir os relatórios numa pasta `dev_notes/` no branch v2? | facilita os próximos releases (2.4) |

Abaixo, `<V2>` é o branch escolhido.

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
# se decidido em 2.0:
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

Commit e push:

```bash
git commit -m "AsTrovello 2.0"
git push origin main
```

Abra o repositório no GitHub e confira se a página inicial mostra só o
README, a licença, o `pyproject.toml` e o `src/`.

### 2.4 Próximos releases (v2 → `main`)

Depois do 2.3, os dois lados passam a ter um ancestral comum, e o merge vira
o normal:

```bash
git checkout main
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

Os documentos internos continuam públicos nos branches `v2-dev/*` e no
histórico: a `main` só deixa de mostrá-los na página inicial.

---

## Parte 3: publicar no PyPI

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
grep '^version' pyproject.toml      # deve dizer 2.0.0
git tag v2.0.0
git push origin v2.0.0
```

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

```bash
twine upload dist/*
```

Página: https://pypi.org/project/astrovello/

### 3.8 Verificação final

```bash
python -m venv /tmp/teste_final
/tmp/teste_final/bin/pip install astrovello
/tmp/teste_final/bin/pip show astrovello          # Version: 2.0.0
/tmp/teste_final/bin/astrovello --help
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

## Problemas comuns

| Mensagem | Causa | O que fazer |
|---|---|---|
| `400 File already exists` | essa versão já foi enviada (o PyPI não aceita reenviar) | subir a versão no `pyproject.toml`, refazer o build |
| `403 Invalid or non-existent authentication` | token errado, de outro site (PyPI vs TestPyPI) ou com escopo de outro projeto | conferir o `~/.pypirc` e o token |
| `403 ... isn't allowed to upload to project` | a conta não é dona do projeto | usar a conta da v1 |
| `twine check` falha no README | Markdown/HTML que o PyPI não renderiza | corrigir o README e refazer o build |
| `refusing to merge unrelated histories` | o merge da Parte 2.3 sem a opção | usar `--allow-unrelated-histories` (só na primeira vez) |
| `pypher: command not found` ao rodar a pipeline | ambiente onde o astrovello foi instalado não está ativado | ativar o ambiente; o `pypher` é instalado junto |
