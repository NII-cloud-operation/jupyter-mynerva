---
description: jupyter_mynerva/models.json の allow/deny パターンの妥当性を見直し、新モデル追加・廃止モデル除去の diff を提示する
---

# モデルリスト spec 見直し

`jupyter_mynerva/models.json` の allow/deny パターンを最新の OpenAI / Anthropic モデルカタログと照合し、必要な更新を提示する。

## 背景

ProvidersHandler は実行時に `OpenAI().models.list()` / `Anthropic().models.list()` を呼び、`models.json` の glob allow/deny で絞り込んでフロントに返す。allow パターンが古いと新モデルが UI に出ず、deny が甘いとストリーミング非対応モデルや古い snapshot が混じる。本 skill はパターンの妥当性を半自動で検証する。

## 実施手順

1. **現状把握**
   - `jupyter_mynerva/models.json` を読む
   - `jupyter_mynerva/routes.py` の `_filter_models()` の挙動を確認

2. **両プロバイダーの最新モデル情報を取得**
   - OpenAI: <https://platform.openai.com/docs/models> / リリースノート
   - Anthropic: <https://docs.anthropic.com/en/docs/about-claude/models> / リリースノート
   - 可能なら `models.list()` API を直接叩いた結果（`MYNERVA_OPENAI_API_KEY` / `MYNERVA_ANTHROPIC_API_KEY` がある環境で）

3. **絞り込み結果を検証**
   - 現 allow/deny を実 API レスポンスに適用した結果のリストを表示
   - 各モデルについて以下を確認:
     - **チャットモデルか**（embeddings / tts / whisper / dall-e / moderation を排除）
     - **ストリーミング対応か**（OpenAI は `o1`(無印) など非対応の歴史あり）
     - **廃止予定でないか**（deprecation スケジュール確認）
     - **alias と snapshot の重複がないか**（後述「snapshot 重複ルール」参照）

4. **更新候補を提示**
   - allow に追加すべき新モデル
   - deny に追加すべき非チャット / 非ストリーミング / 廃止予定モデル / alias で代替された snapshot
   - allow から外すべき廃止モデル
   - 既存パターンの調整（例: `gpt-5*` を `gpt-5.2`, `gpt-5-mini` に分割するか）

### Anthropic の alias-only 方針

Anthropic は同一系統が **alias 形式**（`claude-opus-4-7`）と **日付付き snapshot 形式**（`claude-opus-4-1-20250805`）の両方で API から返る。UI に混在するのを避けるため、現行 spec は **alias 形式のみ採用** している:

```json
"anthropic": {
  "allow": ["claude-sonnet-4-*", "claude-haiku-4-*", "claude-opus-4-*"],
  "deny": ["*-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"]
}
```

- `*-[0-9]×8` は末尾が `-` + 8 桁の数字（YYYYMMDD）にマッチ → snapshot 形式を一律除外
- `?` は「任意 1 文字」で数字限定ではない点に注意（`claude-opus-4-6` のような ID も誤マッチする）
- alias が未公開の family（過去事例: haiku 4.5）は **一時的に UI から消える**。Anthropic が alias を出せば自動復活するので、こちら側で吸収しない
- マイナーバージョン単位の細かい絞り込みはしない（最新 alias を全部見せる方針）

OpenAI 側は allow が exact ID 列挙なので snapshot 混入の問題は発生しない。

5. **diff を生成**
   - `models.json` の修正案を diff 形式で提示
   - ユーザーが承認したら反映

## 注意点

- allow / deny は **glob パターン**（`fnmatch` 準拠）。`*`, `?`, `[seq]` が使える
- パターン例:
  - `gpt-5*` — `gpt-5` で始まる全モデル
  - `claude-*-4-*` — `claude-` の後に何か、`-4-` を挟んで何か続くモデル
  - `*-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]` — 末尾が日付 snapshot（`-20251001` 形式）。**`?` は「任意 1 文字」で数字限定ではない**ので、数字に絞る場合は `[0-9]` を使う
- **Mynerva はストリーミング一択**。非対応モデルは絶対に入れない
- Enki Gate プロバイダーは別経路（`_fetch_openai_models`）で動的取得しており本 skill の対象外

## 完了基準

- 現行 allow/deny で漏れている新モデル / 残存する廃止モデルを洗い出した
- ストリーミング非対応モデルが allow にマッチしないことを確認した
- 必要なら `models.json` の差分を提示し、ユーザー承認後に反映した
