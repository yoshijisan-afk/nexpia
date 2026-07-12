# TripoSplat 自己ホストProvider — Modal / RunPod 比較PoCランブック

Part BBB（Gaussian Splatパイプライン拡張）BBB.13・ADR-320（v0.2追記、v0.3でAPI実測反映）の
PoC (J-6a) 実行手順。対応する設計書:
`docs/design/SisliR_v10_1_partBBB_GaussianSplatPipelineExpansion_v0.1.md` BBB.13.3。

> **TripoSplatのAPI契約自体は実機検証済みです。ただしModal/RunPodへの実デプロイは未検証です。**
> 2026-07-12、プロジェクトオーナーの依頼によりこのセッションでRunPod **L4 Pod**（常時起動の
> 検証用インスタンス）上で実際にTripoSplatを動かし、モデルのロード方法・`pipe.run()`の
> 入出力契約・依存パッケージ・HuggingFace認証不要・実測レイテンシを確認しました（詳細は
> 「1.5 実機検証済みのTripoSplat API」節）。この結果に基づき、本ディレクトリの
> `modal/app.py`・`runpod/handler.py`・`runpod/Dockerfile`・`runpod/requirements.txt`は
> プレースホルダー実装から実処理へ更新済みです。
>
> 一方で、**ローカルPod実機では動作確認済みだが、Modal/RunPodのコンテナ環境での実デプロイは
> まだ未検証です**。`apps/`・`lib/`配下のTypeScriptと異なり、このセッションではModal/RunPod
> アカウントへの実デプロイ（`modal deploy`・RunPod Serverless Endpointの作成）はできません
> でした（課金の発生するアカウント操作が必要なため）。コンテナビルド時間・ネットワーク制約
> （git clone・HuggingFace Hubへの到達性等）・Serverless固有のコールドスタート挙動は
> **ユーザー自身が実行して初めて確認できます**。

## 前提: このPoCで何を決めるか

BBB.13.4のとおり、Modal と RunPod Serverless のどちらにTripoSplatをデプロイするかは
**即断せず、両方で小規模PoC（各数十回の実生成）を行ってから決定する**方針です。
比較すべき指標は本ランブックの「比較記録テンプレート」を参照してください。

## 1. アカウント作成・APIトークン取得

### Modal

1. https://modal.com/ でアカウント作成（公式ドキュメント: https://modal.com/docs/guide ）
2. `pip install modal` （ローカル/開発機にPython環境が必要）
3. `modal token new` でCLI認証を行う（公式手順: https://modal.com/docs/guide/getting-started ）

詳細な操作手順はModal公式ドキュメントを参照してください（本ランブックではリンクのみ）。

### RunPod

1. https://www.runpod.io/ でアカウント作成
2. RunPod Serverless の公式ドキュメント（https://docs.runpod.io/serverless/overview ）に従い、
   APIキーを発行する
3. `pip install runpod` （handler.pyのローカルテスト用。デプロイ自体はDockerイメージ経由）

詳細な操作手順はRunPod公式ドキュメントを参照してください。

## 1.5 実機検証済みのTripoSplat API（2026-07-12・RunPod L4 Pod）

`modal/app.py`・`runpod/handler.py`は以下の実機確認済みAPIに基づいて実装されています。

**モデルのロードと推論**（プロセス起動時に1回だけロードすること。リクエストの都度は不可）:

```python
from triposplat import TripoSplatPipeline

pipe = TripoSplatPipeline(
    ckpt_path              = "ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path           = "ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path            = "ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path = "ckpts/vae/flux2-vae.safetensors",
    rmbg_path              = "ckpts/background_removal/birefnet.safetensors",
    device                 = "cuda",
)

gaussian, prepared = pipe.run(image_path, num_gaussians=262144, show_progress=True)
gaussian.save_ply("output.ply")
gaussian.save_splat("output.splat")
```

- `pipe.run()`の第一引数は**画像ファイルパス**（文字列）。URLではない。handler側でリクエストの
  `image_url`を一時ファイルへダウンロードしてから渡す必要がある（実装済み）
- `num_gaussians`は既定262144（32768/65536/131072/262144から選択可。値が大きいほど高品質・低速）

**モデル重みの取得**（HuggingFace認証トークン不要。コンテナビルド時に1回だけ実行し、
計3.6GBの重みをイメージへ焼き込む設計）:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='VAST-AI/TripoSplat',
    local_dir='ckpts/',
    allow_patterns=['diffusion_models/*','vae/*','clip_vision/*','background_removal/*'],
)
```

**依存パッケージ**（動作確認済みの組み合わせ）: `torch`(2.4.1+cu124で動作確認、新規コンテナ
では最新の安定版で問題ない)・`torchvision`・`numpy`・`safetensors`・`pillow`・`tqdm`・
`huggingface_hub`。`transformers`・`diffusers`は**不要**（TripoSplat独自実装）。

**実測レイテンシ**: 262144 Gaussianの生成で**約28〜30秒**（20ステップのdiffusion sampling）。
GPU使用量はピーク時のみ約6.6GB、処理完了後は解放される。

> **重要な限定事項**: 上記はすべて**RunPod L4 Pod**（常時起動の検証用インスタンス）上での
> 実測です。**Modal / RunPod Serverlessへの実デプロイ・当該環境でのコールドスタート込み
> レイテンシはこのセッションでは未検証**です。Serverlessのコールドスタート秒数は
> BBB.13.4の未検証事項のままであり、本節の28〜30秒という数値には**含まれていません**
> （コールドスタートは別途加算されます）。

## 2. デプロイコマンド

### Modal

```bash
# 一時起動（テスト用。プロセスを止めるとエンドポイントも消える）
modal serve infra/triposplat-selfhost/modal/app.py

# 永続デプロイ
modal deploy infra/triposplat-selfhost/modal/app.py
```

`modal deploy` 完了後に表示されるエンドポイントURL（`https://<workspace>--sislir-triposplat-selfhost-triposplatmodel-infer.modal.run` 相当）を
`TRIPOSPLAT_INFERENCE_ENDPOINT_URL` に設定します（下記4節）。

**着手前に確認しておくこと**: `TripoSplatModel.load_model()` / `run_inference()` は
「1.5 実機検証済みのTripoSplat API」節の実測APIに基づいて実装済みです（プレースホルダーでは
ありません）。ただし**Modal環境そのものへの実デプロイはこのセッションでは未検証**のため、
`modal serve`での一時起動時に、コンテナビルド（git clone・依存インストール・
`snapshot_download`によるモデル重みの取得）が問題なく完了するか、初回リクエストが
成功するかを必ず確認してください。TripoSplatリポジトリの実際のディレクトリ構造・
モジュール名が本実装の想定（`sys.path`に`/opt/triposplat`を追加し`from triposplat import
TripoSplatPipeline`でロード）と異なる場合は、`load_model()`のimport文を調整してください。

### RunPod

```bash
# 1. イメージをビルドしてコンテナレジストリへpush（例: Docker Hub）
docker build -t <your-dockerhub-username>/sislir-triposplat-selfhost:latest \
  infra/triposplat-selfhost/runpod/
docker push <your-dockerhub-username>/sislir-triposplat-selfhost:latest

# 2. RunPod ConsoleでServerless Endpointを作成し、上記イメージを指定する
#    (Web UI操作。詳細はRunPod公式ドキュメント参照:
#    https://docs.runpod.io/serverless/workers/deploy )
```

RunPod ConsoleでEndpoint作成後に発行されるエンドポイントURL
（`https://api.runpod.ai/v2/<endpoint-id>/runsync` 相当）を
`TRIPOSPLAT_INFERENCE_ENDPOINT_URL` に設定します。

**着手前に確認しておくこと**: Modal側と同じく、`infra/triposplat-selfhost/runpod/handler.py` の
`load_model()` / `run_inference()` は「1.5 実機検証済みのTripoSplat API」節の実測APIに
基づいて実装済みです。ただし**RunPod Serverlessへの実デプロイはこのセッションでは未検証**
のため、`docker build`でのイメージビルド（`Dockerfile`内の`snapshot_download`によるモデル
重み取得を含む）が問題なく完了するか、Endpoint作成後の初回リクエストが成功するかを
必ず確認してください。

> **注意**: RunPodの標準レスポンス形式は `{"output": {...}}` でラップされる場合があります
> （公式ドキュメントで要確認）。その場合、`lib/mesh3d/src/providers/TripoSplatSelfHostProvider.ts`
> が期待する `{ ply_url, gaussian_count }` の直下形式に合わせて、RunPod Endpoint側の
> レスポンス整形（またはProvider側のパース処理）を追加調整する必要があります
> （このセッションでは実デプロイしていないため未確認・要PoC確認）。

## 3. 比較記録テンプレート

PoC実行のたびに以下の表を埋めてください（`api_cost_logs`への本番記録とは別に、
PoC結果はこのファイルまたはチームの記録先に残すことを推奨します）。

| 実行日 | ホスティング | コールドスタート時間(s) | 1回あたりレイテンシ p50(s) | 1回あたりレイテンシ p95(s) | 1回あたり実測コスト(円) | Gaussian数 | 出力品質メモ |
|---|---|---|---|---|---|---|---|
| | Modal (L4) | | | | | 262144 or 32768 | |
| | RunPod Serverless (L4) | | | | | 262144 or 32768 | |

**比較すべき指標（BBB.13.4/BBB.13.7）**:

1. **コールドスタート時間**: RunPodは20〜60秒という確認済み情報がある（BBB.13.4）。
   Modal側は未検証のため実測すること。
2. **1回あたりのレイテンシ**（コールドスタート込み・ウォーム状態それぞれ記録推奨）。
   ウォーム状態の参考ベースライン: **262144 Gaussianで約28〜30秒**
   （2026-07-12・RunPod L4 **Pod**での実測。Serverlessのコールドスタートはこれに別途加算される
   ため、Serverless実測値がこのベースラインより大幅に長い場合はコールドスタート起因の可能性が
   高い）
3. **1回あたりの実測コスト（円換算）**: 実行秒数×秒単価×為替（BBB.13.7）。
   仮単価（L4 $0.80/hr・60秒と仮定で約¥2/回。BBB.13.7の未検証の仮定値）と実測値を比較すること。
4. **262,144 GaussiansとGaussian数を絞った場合（例: 32,768）の品質/速度トレードオフ**:
   TripoSplatが生成数を制御可能なパラメータを持つ場合、精細さと生成速度・コストの
   トレードオフを記録する。
5. **失敗率**: 数十回の実行中の失敗回数（画像入力形式起因の失敗・タイムアウト・
   OOM等を区別して記録することを推奨）。
6. **物件写真での出力品質**: TripoSplatはオブジェクト中心の生成が想定されており
   （BBB.13.3）、物件外観・室内シーンでの適合性はPoCで見極める。

## 4. 結線方法（PoC後の本接続への切り替え）

PoCの結果、ModalとRunPodのどちらか一方を採用先として決定したら、
`TRIPOSPLAT_INFERENCE_ENDPOINT_URL` 環境変数にそのエンドポイントURLを設定するだけで、
`lib/mesh3d/src/providers/TripoSplatSelfHostProvider.ts` が自動的にプレースホルダー動作
（`isPlaceholder:true`）から実接続へ切り替わります（環境変数未設定時は安全にプレースホルダー
のまま縮退する設計、BBB.13.5）。

```bash
# .env（環境変数の追記先はデプロイ環境に応じて調整すること。CLAUDE.mdの環境変数節への
# 反映は本タスクのスコープ外。ユーザー自身の判断で行うこと）
TRIPOSPLAT_INFERENCE_ENDPOINT_URL=https://xxx.modal.run/infer   # or RunPod endpoint URL
TRIPOSPLAT_INFERENCE_API_TOKEN=xxxxx   # 任意。エンドポイント保護用トークンを設定した場合のみ
```

さらに、`migrations/007_seed_triposplat_selfhost.sql` でseedした
`engine_provider_settings` の `('splat_gen', 'triposplat_selfhost')` 行は初期
`enabled=false` です。PoC完了・本採用確定後は、管制盤（Part AAA）または直接SQLで
`enabled=true` へ更新してください（`qa_status` の `pass` への昇格運用はBBB.3.1・
Part UUのQAゲート思想を踏襲し、人間承認を経ること）。

## 5. 未実装・要フォローアップ事項（正直な棚卸し）

- `infra/triposplat-selfhost/modal/app.py` / `infra/triposplat-selfhost/runpod/handler.py` の
  `load_model()` / `run_inference()` は「1.5 実機検証済みのTripoSplat API」節の実測APIに
  基づいて実装済みです（プレースホルダーではありません）。ただし**Modal/RunPodのコンテナ
  環境での実デプロイはこのセッションでは未実施**であり、ビルド時間・ネットワーク制約
  （git clone・HuggingFace Hubへの到達性等）・依存関係の解決可否はユーザー自身の実行で
  初めて確認できます。
- 生成したPLYのCloudflare R2への永続化アップロードは未実装です（現状はbase64データURLを
  返す簡易実装）。本採用時にR2アップロード処理を追加してください。
- `SPLATGEN_SELFHOST_MONTHLY_CALL_LIMIT`（BBB.13.6の環境変数草案）による月間呼び出し回数
  ガード、および`api_cost_logs`（provider='modal'|'runpod'）への実コスト記録は、本PoC PR
  ([J-6a](../../docs/design/SisliR_v10_1_partBBB_GaussianSplatPipelineExpansion_v0.1.md))
  のスコープ外です（BBB.13.7の運用実装は本採用確定後の別タスク）。
- RunPodのレスポンス形式（`{"output": {...}}`ラップの有無）は上記3節の注意書きのとおり
  要PoC確認です。
- Modal/RunPod **Serverless**でのコールドスタート込みレイテンシ・p50/p95・失敗率・
  実測コスト（円換算）は未検証です（BBB.13.4の未検証事項1〜4がそのまま該当）。
  「1.5」節の28〜30秒はあくまでPod（常時起動）環境でのウォーム状態の参考値です。
