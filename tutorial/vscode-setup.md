# VSCode 拡張機能 利用マニュアル

Ruff / basedpyright / GitHub Actions を使うプロジェクト向けに、VSCode の推奨拡張機能のインストールから使い方までをまとめます。

## 目次

1. Linter / Formatter(Ruff・Basedpyright・Python)
2. GitHub(Pull Requests and Issues・GitHub Actions・GitLens)
3. そのほか(Git Graph・gitHD・推奨拡張機能の共有)

---

## 0. 事前準備

- VSCode をインストールしておく
- リポジトリを clone しておく
- 拡張機能のインストール方法(共通)
  1. 左サイドバーの拡張機能アイコン(`Ctrl+Shift+X` / Mac は `Cmd+Shift+X`)を開く
  2. 拡張機能 ID(例: `charliermarsh.ruff`)で検索する
  3. 「インストール」をクリックする

> 後述の `.vscode/extensions.json` がリポジトリにある場合は、フォルダを開いた際に右下に表示される「推奨拡張機能をインストール」から一括インストールできます。

---

## 1. Linter / Formatter

### 1-1. Python (`ms-python.python`)

VSCode 公式の Python 基本拡張機能です。インタープリターの選択に必要です。

**手順**

1. 拡張機能 `ms-python.python` をインストールする
2. `Ctrl+Shift+P` でコマンドパレットを開き、`Python: Select Interpreter` を実行する
3. 使用する Python を選択する
4. 右下のステータスバーに選択した Python のバージョンが表示されれば完了

### 1-2. Ruff (`charliermarsh.ruff`)

超高速な Python リンター兼フォーマッタです。コードの自動修正やインポート順序の整頓(`isort` 相当)を担います。

**手順**

1. 拡張機能 `charliermarsh.ruff` をインストールする
2. `.vscode/settings.json` に以下を追記する(プロジェクトルートに `.vscode` フォルダがなければ作成)

```json
{
  // PythonファイルのデフォルトフォーマッタにRuffを指定
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      // 保存時に自動修正可能なリンターエラーを修正
      "source.fixAll.ruff": "explicit",
      // 保存時にインポート順序を自動整頓
      "source.organizeImports.ruff": "explicit"
    }
  }
}
```

**使い方**

- 保存(`Ctrl+S`)するだけで、フォーマット・自動修正・インポート整頓が実行される
- 問題箇所には波線が表示され、マウスを重ねると Ruff のルールコードと内容を確認できる
- 電球アイコン(`Ctrl+.`)から個別に修正を適用できる
- 手動で実行する場合は、コマンドパレットから以下を実行する
  - `Ruff: Fix all auto-fixable problems`
  - `Ruff: Format Document`
  - `Ruff: Organize Imports`

**ルールの変更**

無視するルールや対象ディレクトリなどは、VSCode ではなく `pyproject.toml` に集約されています。ルールを変えたい場合は `pyproject.toml` を編集してください。ただし、`toml`ファイルよりもVScodeなどのユーザー設定は優先されるため、一時的にルールを変更する場合はユーザー設定を行っても構いません。

### 1-3. Basedpyright (`detachhead.basedpyright`)

`pyproject.toml` に設定されている `basedpyright` に対応する、静的型チェックツールの拡張機能です。

**手順**

1. 拡張機能 `detachhead.basedpyright` をインストールする
2. `.vscode/settings.json` に以下を追記する

```json
{
  "basedpyright.analysis.diagnosticMode": "workspace",
  "basedpyright.analysis.autoSearchPaths": true,
  "basedpyright.analysis.useLibraryCodeForTypes": true
}
```

1. 「問題」パネル(`Ctrl+Shift+M`)で型エラーを確認する

**使い方**

- `diagnosticMode` を `workspace` にしているため、開いていないファイルも含めてプロジェクト全体の型エラーが「問題」パネルに表示される
- 型チェックの厳しさや対象は `pyproject.toml` の `[tool.basedpyright]` で制御される

### 1-4. Pylance との衝突防止

Basedpyright を使う場合、標準の Pylance による型チェックと重複してエラーが二重に表示されることがあります。その場合は、`.vscode/settings.json` に以下を追加して Pylance 側の型チェックを無効にします。

```json
{
  "python.analysis.typeCheckingMode": "off"
}
```

---

## 2. GitHub

### 2-1. GitHub Pull Requests and Issues (`GitHub.vscode-pull-request-github`)

VSCode 上で PR の作成、レビュー、コメントのやり取り、スレッドの解決を直接行えます。

**手順**

1. 拡張機能 `GitHub.vscode-pull-request-github` をインストールする
2. 左サイドバーに追加された GitHub アイコンを開く
3. 「Sign in」から GitHub アカウントでサインインする(ブラウザで認証)
4. `.vscode/settings.json` に以下を追記する

```json
{
  "githubPullRequests.remotes": ["origin"],
  "githubPullRequests.focusedMode": true
}
```

**使い方**

- 自分の PR / レビュー依頼された PR / Issue が一覧表示される
- PR を開くと、差分の確認、行コメントの追加、スレッドの解決ができる
- ブランチをプッシュした後、「Create Pull Request」から PR を作成できる
- PR のチェックアウトも一覧から 1 クリックで行える

### 2-2. GitHub Actions (`github.vscode-github-actions`)

`.github/workflows/ci.yml` などの設定ファイルの編集をサポートし、CI ジョブ(`Lint & Type Check` など)の実行結果やログをエディタ内で確認できます。

**手順**

1. 拡張機能 `github.vscode-github-actions` をインストールする
2. GitHub にサインインする(Pull Requests 拡張機能と共通のアカウントで認証される)
3. `.vscode/settings.json` に以下を追記する

```json
{
  "github-actions.workflows.pinned.workflows": [
    ".github/workflows/ci.yml"
  ]
}
```

**使い方**

- 左サイドバーの GitHub Actions アイコンから、ワークフローの実行履歴とジョブの状態を確認できる
- 失敗したジョブを開くと、ステップごとのログをエディタ内で読める
- ワークフローの YAML を編集すると、自動補完と構文検証が効く
- ピン留めしたワークフローは、一覧の上部に常に表示される

### 2-3. GitLens (`eamodio.gitlens`)

行ごとの変更履歴(`git blame`)やコミットグラフの確認など、Git の追跡機能を拡張します。

**手順**

1. 拡張機能 `eamodio.gitlens` をインストールする
2. `.vscode/settings.json` に以下を追記する

```json
{
  "gitlens.currentLine.enabled": true,
  "git.autofetch": true
}
```

**使い方**

- カーソルのある行の末尾に、直近の変更者・日時・コミットメッセージが薄く表示される
- ファイル履歴や行履歴から、過去の変更内容を差分付きで追える
- 左サイドバーの GitLens ビューで、ブランチ・コミット・スタッシュなどを管理できる

---

## 3. そのほか

### 3-1. Git Graph

ブランチやコミットの履歴をグラフで視覚的に確認できます。

**手順**

1. 拡張機能 `Git Graph`(作者: mhutchie)をインストールする
2. ステータスバーの「Git Graph」ボタン、またはコマンドパレットの `Git Graph: View Git Graph` を実行する

**使い方**

- コミットをクリックすると、変更ファイルと差分が表示される
- コミットを右クリックして、チェックアウト・ブランチ作成・マージ・cherry-pick などを実行できる

### 3-2. gitHD (Git History Diff)

Git の履歴と差分を確認するための拡張機能です。コミットごと、ファイルごとの変更内容を差分付きで追えます。

**手順**

1. 拡張機能タブで `Git History Diff` を検索してインストールする
2. インストール後、左サイドバーや SCM ビューに追加される履歴ビューを開く

**使い方**

- コミット履歴の一覧から、コミットを選んで変更ファイルを確認できる
- ファイルを選ぶと、そのコミットでの差分(diff)がエディタで開く
- Git Graph(ブランチ全体の流れ)と併用すると、全体像と個別の差分の両方を確認しやすい

### 3-3. 推奨拡張機能のチーム共有 (`.vscode/extensions.json`)

プロジェクトルートに `.vscode/extensions.json` を作成して以下を配置すると、リポジトリを開いたメンバーに推奨拡張機能が案内されます。

```json
{
  "recommendations": [
    "charliermarsh.ruff",
    "detachhead.basedpyright",
    "ms-python.python",
    "GitHub.vscode-pull-request-github",
    "github.vscode-github-actions",
    "eamodio.gitlens"
  ]
}
```

Git Graph や gitHD も共有したい場合は、各拡張機能の ID(拡張機能の詳細ページで確認できます)を `recommendations` に追加してください。

**メンバー側の手順**

1. リポジトリのフォルダを VSCode で開く
2. 右下に表示される「推奨拡張機能をインストールしますか?」で「インストール」を選ぶ
3. 表示されない場合は、拡張機能タブで `@recommended` と検索すると一覧から導入できる

---

## 4. 動作確認チェックリスト

- [ ] 右下に使用する Python のバージョンが表示されている
- [ ] Python ファイルを保存すると、フォーマットとインポート整頓が自動で実行される
- [ ] 「問題」パネルに Ruff / Basedpyright の診断結果が表示される
- [ ] 型エラーが Pylance と二重に表示されていない
- [ ] GitHub にサインインでき、PR・Issue が一覧に表示される
- [ ] GitHub Actions のワークフロー実行状況が確認できる
- [ ] 行末に GitLens の blame 情報が表示される
