# 環境構築手順書

Pythonのプロジェクトで `venv` を作成し、`pyproject.toml` に記載された依存関係（モジュール）をインストールして開発環境を構築する手順です。

---

## 1. ディレクトリの作成と移動

プロジェクト用のフォルダを作成し、その中に移動します。

```bash
mkdir my_project
cd my_project

```

---

## 2. 仮想環境（venv）の作成と有効化

プロジェクト専用の独立したPython実行環境（`venv`）を作成します。

### 仮想環境の作成

```bash
python -m venv .venv

```

> **Point:** 仮想環境名には `.venv` や `venv` がよく使われます。`.venv` にしておくと、多くのエディタやツールが自動的に仮想環境として認識します。

### 仮想環境の有効化（Activate）

OSに合わせて実行してください。

* **Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1

```

*(スクリプト実行エラーが出た場合は `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` を先に実行してください)*

* **Windows (Command Prompt):**

```cmd
.\.venv\Scripts\activate.bat

```

* **macOS / Linux:**

```bash
source .venv/bin/activate

```

有効化されると、ターミナルの先頭に `(.venv)` と表示されます。

---

## 3. pyproject.toml の作成

プロジェクト直下に `pyproject.toml` ファイルを作成します。

pip（`pip install -e .` または `pip install .`）で依存関係を認識させるため、最小限のパッケージ定義を記述します。

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my_project"
version = "0.1.0"
dependencies = [
    "requests>=2.28.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "mypy",
]

```

---

## 4. モジュールのインストール

`pyproject.toml` に書かれたモジュールを仮想環境へインストールします。

### 最新の pip を準備

先にパッケージ管理ツール（pip）自体を更新しておきます。

```bash
python -m pip install --upgrade pip

```

### 通常の依存関係のみインストール

プロジェクト本体と `dependencies` に書かれたパッケージをインストールします。

```bash
pip install .

```

### 開発用パッケージもまとめてインストール（おすすめ）

開発時（テストやリンター等）に必要な `optional-dependencies.dev` も同時に入れる場合は、編集可能モード（`-e`）でインストールします。

```bash
pip install -e ".[dev]"

```

> **`-e`（Editableモード）とは:** コードを変更した際に再インストールの必要がなく、すぐに修正が反映される開発用のインストール方法です。

---

## 5. インストールの確認

環境が正しく構築されたか確認します。

### パッケージ一覧の確認

```bash
pip list

```

`requests` や `pandas`（およびそれらの依存ライブラリ）がインストールされていることを確認します。

### 動作確認

Pythonインタープリタを起動してインポートを試します。

```bash
python -c "import requests, pandas; print('環境構築完了!')"

```

---

## 6. 仮想環境の無効化（Deactivate）

作業を終了するときは、以下のコマンドで仮想環境から抜けます。

```bash
deactivate

```

---

## (.gitignore の設定)

Gitでバージョン管理を行う場合は、プロジェクト直下の `.gitignore` に `.venv` を追加して、仮想環境フォルダがリポジトリに入らないようにしておきましょう。

```text
.venv/
__pycache__/
*.egg-info/

```
