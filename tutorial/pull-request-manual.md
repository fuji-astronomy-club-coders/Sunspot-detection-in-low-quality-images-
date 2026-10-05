# Pull Request 利用マニュアル

このテンプレートから作成したリポジトリで、変更を `main` ブランチに取り込むまでの手順をまとめます。

ルールセット(`Default PR Rules`)を適用すると `main` へ直接 push できなくなるため、**すべての変更は Pull Request(PR)経由**になります。

## 目次

1. PR の全体像とルール
2. 作業の流れ
3. ローカルでのチェック(CI の再現)
4. PR の作成
5. レビューする側の手順
6. マージ
7. CI が落ちたとき
8. よくある質問

---

## 1. PR の全体像とルール

### 1-1. 全体の流れ

```
ブランチ作成 → 編集・commit → ローカルチェック → push → PR作成
   → CI通過 + レビュー承認 + スレッド解決 → Squash and merge → ブランチ削除
```

### 1-2. ルールセットで強制されている内容

`Default PR Rules.json` の設定により、デフォルトブランチ(`main`)には次の制限がかかっています。

| ルール | 内容 | 意味すること |
| --- | --- | --- |
| Restrict deletions | ブランチ削除を禁止 | `main` は削除できない |
| Block force pushes | 強制 push を禁止 | 履歴の書き換えはできない |
| Require a pull request | PR 必須 | 直接 push 不可 |
| 承認数 | 1 人以上の Approve が必要 | 最低 1 人のレビューが必要 |
| Code owner review | コードオーナーのレビューが必要 | `CODEOWNERS` に指定された人の承認が必要 |
| Dismiss stale reviews | push すると過去の Approve が無効化 | 修正を push したら**再レビュー**が必要 |
| Thread resolution | 全レビューコメントのスレッド解決が必須 | 未解決のコメントがあるとマージ不可 |
| マージ方法 | **Squash のみ** | PR の全 commit が 1 つにまとまる |
| Status checks | `Lint & Type Check` の成功が必須 | CI が通らないとマージ不可 |

> [!NOTE]
> `Require code owner review` を有効にしているため、リポジトリに `.github/CODEOWNERS` ファイルがないと、承認条件を満たせずマージできなくなる場合があります。新しいリポジトリを作ったら、`CODEOWNERS` に担当者(例: `* @username`)が設定されているか確認してください。

> [!NOTE]
> リポジトリの Admin 権限を持つ人と Deploy Key はルールを bypass できる設定になっています。ただし、緊急時以外は bypass せず、通常どおり PR を使いましょう。

---

## 2. 作業の流れ

### 2-1. 最新の `main` を取得する

作業を始める前に、ローカルの `main` を最新にします。

```bash
git switch main
git pull
```

GitHub Desktop の場合は、`Current Branch` を `main` にして `Fetch origin` → `Pull origin` をクリックします。

### 2-2. 作業用ブランチを作成する

`main` から新しいブランチを切ります。**1 つの PR では 1 つの目的だけ**を扱い、大きな変更は分割しましょう。

```bash
git switch -c feature/add-image-loader
```

ブランチ名の例です。

| 種類 | 接頭辞 | 例 |
| --- | --- | --- |
| 機能追加 | `feature/` | `feature/add-image-loader` |
| バグ修正 | `fix/` | `fix/crash-on-empty-input` |
| ドキュメント | `docs/` | `docs/update-readme` |
| 設定・CI の変更 | `chore/` | `chore/update-ruff-version` |
| リファクタリング | `refactor/` | `refactor/split-processor` |

ブランチ名は ascii 文字のみ、単語はハイフンでつなぎます。

### 2-3. 編集して commit する

変更を小さな単位で commit します。commit message は「何をしたか」が分かる短い文にしてください。

```bash
git add .
git commit -m "Add image loader"
```

> [!TIP]
> このリポジトリは **Squash merge** のため、PR 内の commit はマージ時に 1 つにまとめられます。途中の commit message が多少粗くても問題ありませんが、**PR のタイトルが最終的な commit message になる**ので、タイトルは丁寧に書いてください。

---

## 3. ローカルでのチェック(CI の再現)

push する前に、CI と同じチェックをローカルで実行しておくと、差し戻しを減らせます。

### 3-1. 開発用パッケージのインストール(初回のみ)

```bash
python -m pip install -e ".[dev]"
```

### 3-2. CI と同じ 4 つのチェック

`.github/workflows/ci.yml` の `Lint & Type Check` ジョブは、次の順で実行されます。

| # | チェック | 失敗時にマージを止めるか |
| --- | --- | --- |
| 1 | `pyproject.toml` の dependencies チェック | 止める |
| 2 | Ruff(警告のみのルール) | **止めない**(警告表示のみ) |
| 3 | Ruff(CI ブロッキングルール) | 止める |
| 4 | basedpyright(型チェック) | 止める |

それぞれのローカル実行コマンドは次のとおりです。

```bash
# 1. dependencies が pyproject.toml と一致しているか
python generate_dependencies.py --check

# 2. 警告のみのルール(失敗しても exit code は 0)
ruff check . --select I001,D,PTH,UP015,UP017,B905 --output-format=concise --exit-zero

# 3. CI でブロックされるルール
ruff check . --config .github/ci/ruff.toml --ignore I001,D,PTH,UP015,UP017,B905

# 4. 型チェック
basedpyright
```

> [!NOTE]
> `generate_dependencies.py --check` が失敗した場合は、コード内の import と `pyproject.toml` の `dependencies` が食い違っています。`--check` を付けずにスクリプトを実行して更新する運用になっている場合は、その結果を commit してください。(スクリプトの仕様に合わせて確認してください)

### 3-3. 自動修正

フォーマットと自動修正可能な指摘は、次のコマンドでまとめて直せます。

```bash
ruff format .
ruff check . --fix
```

VSCode の設定(`vscode-setup.md` 参照)をしていれば、保存時に自動で実行されます。

### 3-4. 「警告のみ」ルールと「ブロッキング」ルールの違い

CI の Ruff チェックは 2 段階に分かれています。

- **警告のみ**: `I001`(import 順序)、`D`(docstring)、`PTH`(pathlib 推奨)、`UP015`、`UP017`、`B905`(zip の strict 指定)。CI のログに表示されますが、マージはブロックされません。
- **ブロッキング**: 上記以外のルール。`.github/ci/ruff.toml` が `pyproject.toml` を継承しつつ、低優先度のルールを除外しています。

警告は無視してよいわけではありません。**新しく書くコードでは、できるだけ警告も解消**してください。

---

## 4. PR の作成

### 4-1. ブランチを push する

```bash
git push -u origin feature/add-image-loader
```

GitHub Desktop の場合は `Publish branch`(2 回目以降は `Push origin`)をクリックします。

### 4-2. PR を作成する

次のいずれかの方法で作成します。

**A. GitHub のブラウザ画面から**

1. push 後にリポジトリページに表示される `Compare & pull request` をクリック
2. 表示されない場合は `Pull requests` タブ > `New pull request` から、base を `main`、compare を作業ブランチにして作成

**B. VSCode から**(`GitHub Pull Requests` 拡張機能)

1. 左サイドバーの GitHub アイコンを開く
2. `Create Pull Request` をクリック

**C. GitHub Desktop から**

1. メニューの `Branch` > `Create Pull Request` を選ぶ(ブラウザが開く)

### 4-3. タイトルと説明を書く

**タイトル**は Squash merge 後の commit message になります。「何を変えたか」が一目で分かる 1 行にしてください。

| 良い例 | 悪い例 |
| --- | --- |
| `Add image loader with resize option` | `update` |
| `Fix crash when input directory is empty` | `修正しました` |

**説明(Description)** には次の内容を書きます。

```markdown
## 概要
何を変更したか、1〜3 行で。

## 背景・目的
なぜこの変更が必要か。関連する Issue があれば `Closes #12` のように書く。

## 変更内容
- 主な変更点 1
- 主な変更点 2

## 確認方法
レビュアーがどう動作確認すればよいか。

## 補足
レビューで特に見てほしい点、未対応の事項など。
```

> [!TIP]
> `Closes #12` と書いておくと、マージ時に Issue #12 が自動でクローズされます。

### 4-4. 作成前のセルフチェック

- [ ] 1 つの PR で 1 つの目的になっている
- [ ] ローカルでチェック(3 章)を実行し、エラーがない
- [ ] 不要なファイル(デバッグ用ファイル、API キー、個人情報など)が含まれていない
- [ ] 差分(`Files changed`)を自分で一度読み返した
- [ ] PR のタイトルと説明を書いた
- [ ] レビュアー(Reviewers)を指定した

### 4-5. Draft PR

作業途中で早めに共有したい場合は、`Create draft pull request` を選びます。Draft の間はマージできません。準備ができたら `Ready for review` をクリックしてください。

---

## 5. レビューする側の手順

### 5-1. レビューの観点

`Files changed` タブで差分を確認します。

- 目的に対して変更内容が適切か
- 不要な変更や、関係のない整形差分が混ざっていないか
- 型ヒント・docstring が付いているか(`ANN`・`D` ルール)
- 命名が分かりやすいか
- 機密情報が含まれていないか

### 5-2. コメントの付け方

1. 差分の行番号の左にある `+` をクリックしてコメントを書く
2. 複数のコメントをまとめて送る場合は `Start a review` を使う
3. レビューを完了するときに、次のいずれかを選んで `Submit review` をクリックする

| 選択肢 | 使いどころ |
| --- | --- |
| Comment | 承認も拒否もしない、感想や質問のみ |
| Approve | 問題なし。マージしてよい |
| Request changes | 修正が必要。解消されるまでマージさせたくない |

### 5-3. 指摘のしかた

- 必須の修正か、任意の提案かを明記する(例: `[must]` `[nits]` `[question]`)
- 理由も一緒に書く
- 良い点も積極的に伝える

---

## 6. 指摘への対応とマージ

### 6-1. 指摘への対応

1. 同じブランチで修正して commit し、push する(PR に自動で反映される)
2. 各コメントのスレッドに、対応内容を返信する
3. 対応が完了したスレッドは `Resolve conversation` をクリックする

> [!IMPORTANT]
> このリポジトリでは **`dismiss_stale_reviews_on_push`** が有効です。修正を push すると、それまでの Approve は無効になります。push 後はレビュアーに **再レビューを依頼**(`Re-request review`)してください。

### 6-2. マージの条件

PR ページ下部のチェック欄が、すべて緑になっていることを確認します。

- [ ] `Lint & Type Check` が成功している
- [ ] 必要な承認数(1 人以上)を満たしている
- [ ] コードオーナーの承認がある
- [ ] すべてのスレッドが解決されている

### 6-3. Squash and merge

マージ方法は **`Squash and merge` のみ**です(他の方法はルールセットで無効)。

1. `Squash and merge` をクリック
2. commit message のタイトル(PR タイトル)と本文を確認し、必要なら編集する
3. `Confirm squash and merge` をクリック

### 6-4. マージ後の片付け

1. PR ページの `Delete branch` をクリックして、リモートの作業ブランチを削除する
2. ローカルを更新し、不要になったブランチを削除する

```bash
git switch main
git pull
git branch -d feature/add-image-loader
```

---

## 7. CI が落ちたとき

PR ページの `Checks` タブ、または失敗したチェックの `Details` から、ログを確認できます。VSCode の `GitHub Actions` 拡張機能でも読めます。

| 失敗したステップ | 主な原因 | 対処 |
| --- | --- | --- |
| `Check pyproject dependencies` | 新しい import を追加したが `pyproject.toml` の dependencies が未更新 | `generate_dependencies.py` で依存関係を更新して commit する |
| `Ruff check - errors` | ブロッキングルール違反(未使用 import、型ヒントの欠落、`SIM`・`B` 系の指摘など) | ログの行番号を確認し、`ruff check . --fix` で自動修正、または手動で修正する |
| `basedpyright` | 型エラー(型の不一致、import できないモジュールなど) | 該当箇所の型ヒントを見直す。`pyproject.toml` の `include` にパッケージ名が設定されているかも確認する |
| `Install project and development dependencies` | `pyproject.toml` の構文エラーや、`packages` に指定したディレクトリが存在しない | パッケージ名のプレースホルダー(`pyproject`)を置換し忘れていないか確認する |

> [!NOTE]
> Ruff の「警告のみ」ステップ(`Ruff check - warnings`)は失敗扱いになりません。ログに表示された警告は、余裕があるときに解消してください。

修正したら、同じブランチに push するだけで CI が自動的に再実行されます。

---

## 8. よくある質問

**Q. `main` に直接 push しようとしたらエラーになった**

ルールセットにより禁止されています。作業用ブランチを作成して PR を出してください。すでに `main` 上で commit してしまった場合は、`git switch -c 新ブランチ名` で現在の状態から新しいブランチを作り、`main` は `git reset --hard origin/main` で戻します(未 push の commit のみ)。

**Q. 承認されているのにマージボタンが押せない**

次のいずれかが原因として考えられます。

- 承認後に push したため、Approve が無効になった(再レビューを依頼する)
- 未解決のスレッドが残っている
- コードオーナーの承認がない(`CODEOWNERS` の設定を確認する)
- CI がまだ実行中、または失敗している

**Q. 他の人の変更を取り込んで、`main` との競合(conflict)が起きた**

作業ブランチに `main` を取り込んで競合を解消します。

```bash
git switch feature/add-image-loader
git fetch origin
git merge origin/main
# 競合ファイルを編集して解消 → git add → git commit
git push
```

> [!NOTE]
> 履歴を書き換える `rebase` + force push は、レビューが始まった後のブランチでは避けてください。レビュー済みの差分が分からなくなります。

**Q. PR が大きくなりすぎた**

目的ごとに PR を分割しましょう。レビューの負担が減り、問題の切り分けもしやすくなります。「リファクタリング」と「機能追加」は別の PR にするのがおすすめです。

**Q. レビュアーがなかなか反応してくれない**

PR にコメントでメンションする(`@username お手すきの際にレビューをお願いします`)か、チームのチャットで連絡してください。

---

## 9. チェックリスト(まとめ)

**PR を出す前**

- [ ] `main` から作業ブランチを切っている
- [ ] ローカルで Ruff / basedpyright / dependencies チェックを通した
- [ ] 差分を自分で読み返した
- [ ] PR のタイトルと説明を書き、レビュアーを指定した

**マージ前**

- [ ] CI(`Lint & Type Check`)が成功している
- [ ] 1 人以上の Approve(コードオーナー含む)がある
- [ ] すべてのスレッドが解決されている
- [ ] Squash 後の commit message(PR タイトル)を確認した

**マージ後**

- [ ] リモート・ローカルの作業ブランチを削除した
- [ ] ローカルの `main` を最新に更新した
