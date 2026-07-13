# TRELLIS.2-4B 自己ホストProvider — RunPod デプロイランブック

Part AAA（パイプライン管制盤＋プリセット）AAA.11.1・ADR-325の実装。対応する設計書:
`docs/design/SisliR_v10_1_partAAA_PipelineControlPanel_v1.0.md` AAA.11.1/AAA.11.4。

姉妹実装 `infra/triposplat-selfhost/`（Part BBB・ADR-320、RunPod Serverlessでエンドツーエンド
動作確認済み）と同じ構成・同じ教訓を踏襲する。TripoSplatと異なりModalは対象外
（本タスクではRunPodのみを実装。Modal対応が必要になった場合は
`infra/triposplat-selfhost/modal/app.py`を参考に別途追加すること）。

> **2026-07-13時点のステータス**: RunPod **Pod**（常時起動インスタンス）上で画像→3D生成の
> エンドツーエンド動作を実機確認済み（推論167.4秒・ピークVRAM 7.09GB・GLB出力41.5MB。
> 設計書AAA.11.1参照）。**本Dockerfile/handler.pyの実ビルド・RunPod Serverlessへの実デプロイは
> まだ行っていない**（RunPodアカウント・課金操作を伴うため、実装担当エージェントの権限では
> 実行不可）。TripoSplat（Part BBB）がPod実機検証→Dockerfile作成→Serverless実デプロイまで
> 完走したのと同じ手順を、本ランブックに沿って次工程として実施すること。

## 1. アカウント作成・APIトークン取得

### RunPod

1. https://www.runpod.io/ でアカウント作成（TripoSplatのPoCで既に取得済みのアカウントを流用可）
2. RunPod Serverless の公式ドキュメント（https://docs.runpod.io/serverless/overview ）に従い、
   APIキーを発行する
3. `pip install runpod` （handler.pyのローカルテスト用。デプロイ自体はDockerイメージ経由）

### HuggingFace（TripoSplatには無かった新規手順・必読）

TRELLIS.2は2件のgated（アクセス承認制）HuggingFaceリポジトリに依存する
（AAA.11.1実機検証済み。当初「認証不要」としていた設計上の推定は誤りだった）:

1. `facebook/dinov3-vitl16-pretrain-lvd1689m`（画像条件付けエンコーダ）: モデルページで
   アクセスリクエストを送信する。**Meta側の手動レビューが必要で、承認まで即時ではなく
   変動する時間を要する**（2026-07-13のPoCで実際に体験済み）。早めに申請しておくこと。
2. `briaai/RMBG-2.0`（背景除去モデル）: 同様にアクセスリクエストが必要だが、
   2026-07-13のPoCでは**即時承認**だった。
3. 両方への承認取得後、read権限のHuggingFaceトークン（`HF_TOKEN`）を発行する
   （https://huggingface.co/settings/tokens ）。

## 1.5 実機検証済みのTRELLIS.2 API（2026-07-13・RunPod A40 Pod）

`runpod/handler.py`は以下の実機確認済みAPIに基づいて実装されています。

**モデルのロードと推論**（プロセス起動時に1回だけロードすること。リクエストの都度は不可）:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")  # またはローカルパス
pipeline.cuda()

image = Image.open(input_image_path)
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrastの制約

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
    coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096, remesh=True,
    remesh_band=1, remesh_project=0, verbose=True,
)
glb.export("output.glb", extension_webp=True)
```

- `pipeline.run(image)`の第一引数は**PIL.Image**（TripoSplatは画像ファイルパス文字列だったのと
  異なる点に注意）。handler側ではリクエストの`image_url`を一時ファイルへは書き出さず、
  `PIL.Image.open(io.BytesIO(...))`で直接デコードして渡している。
- `mesh.simplify(16777216)`はnvdiffrastの頂点数上限に合わせた簡略化。省略すると
  レンダリング/UV展開ステージで失敗しうる。

**実測結果**（2026-07-13・RunPod A40・Ampere sm_86・付属サンプル画像
`assets/example_image/T.png`使用）:

| 指標 | 実測値 |
|---|---|
| 推論時間（画像入力→メッシュ生成完了まで、`pipeline.run()`のみ） | 167.4秒 |
| ピークVRAM使用量 | 7.09GB（GPU要件表の「24GB推奨」を大幅に下回る） |
| 生成メッシュ（簡略化前） | 564万頂点・1145万面 |
| 生成メッシュ（簡略化後・実運用相当） | 48.5万頂点・97.6万面 |
| GLB出力サイズ | 41.5MB |

> 上記の167.4秒は`pipeline.run()`単体の実測。remeshing・simplify・xatlas UV展開の
> オーバーヘッドを含めた合計は200秒超（設計書AAA.11.1参照）。
> `TRELLIS2_SELFHOST_INFERENCE_TIMEOUT_MS`（`lib/mesh3d/src/providers/TrellisSelfHostProvider.ts`）
> はこの実測値＋コールドスタート安全マージンを見込んで600,000ms(10分)としている。

**GPU世代要件（最重要・実機検証済み）**: Ampere/Hopper世代(sm_80/86/90。A40・RTX A5000・
RTX A6000・A100・H100・L4等)限定。**Blackwell世代(sm_120。RTX 5090・RTX PRO 4500/5000/6000)は
使用不可**。`torch==2.6.0+cu124`がBlackwellのsm_120カーネルを持たず、単純な行列積すら
`CUDA error: no kernel image is available for execution on the device`で実行できないことを
直接確認済み（警告ではなく実行不能）。RunPod ConsoleでEndpoint作成時、**「Enabled GPU types」で
Blackwell系を明示的に除外し、Ampere/Hopper系のみに限定すること**。

## 2. 依存関係インストールの3つの罠（2026-07-13実機で発見。`Dockerfile`に反映済み）

公式`setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm`
（`--new-env`無し。condaは不要、system Pythonへ直接インストールで動作した）を素直に実行すると
以下3点で必ずつまずく。`Dockerfile`はこれらすべてに対処済み。

1. **`--flexgemm`ステップがtorchを意図せず`2.13.0`(cu13.0ビルド)へ上書きする**。
   `torch.utils.cpp_extension`のCUDAバージョン不一致で`o_voxel`のビルドが
   「detected CUDA version (12.4) mismatches version used to compile PyTorch (13.0)」で失敗する。
   → `setup.sh`完了後（o_voxelのビルド失敗でsetup.sh自体がエラー終了しても）、
   `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`
   で明示的に再インストールし、`cp -r TRELLIS2/o-voxel /tmp/extensions/o-voxel && pip install
   /tmp/extensions/o-voxel --no-build-isolation`でo_voxelを再ビルドする。
2. **ベースイメージ由来の`torchaudio`がtorchとABI不整合でクラッシュする**（`transformers`の
   import chainで`undefined symbol`エラー）。画像→3D生成にtorchaudioは不要
   （`transformers`の`is_torchaudio_available()`がその使用箇所をガードしているため、
   単純な`pip uninstall -y torchaudio`で解消する）。
3. **`transformers==5.13.1`のDINOv3実装とTRELLIS.2バンドルコードの構造不一致**。
   TRELLIS.2の`trellis2/modules/image_feature_extractor.py`の`DinoV3FeatureExtractor
   .extract_features()`は`for i, layer_module in enumerate(self.model.layer):`という
   平坦な構造を前提にしているが、実際のこのtransformersバージョンでは
   `DINOv3ViTModel.model`自体が`DINOv3ViTEncoder`（`.layer`を持つラッパー）であるため、
   正しいパスは`self.model.model.layer`。`sed -i 's/self\.model\.layer/self.model.model.layer/'
   trellis2/modules/image_feature_extractor.py`でパッチする（クローン直後・パッケージ化前に
   適用すること）。

## 3. デプロイコマンド

```bash
# 1. イメージをビルドしてコンテナレジストリへpush（例: Docker Hub）
# HF_TOKENはビルドシークレットとして渡す（イメージレイヤー・ビルドログへ値を残さない）。
export HF_TOKEN=<あなたのHuggingFace read権限トークン>
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t <your-dockerhub-username>/sislir-trellis2-selfhost:latest \
  -f infra/trellis2-selfhost/runpod/Dockerfile \
  infra/trellis2-selfhost/
docker push <your-dockerhub-username>/sislir-trellis2-selfhost:latest

# 2. RunPod ConsoleでServerless Endpointを作成し、上記イメージを指定する
#    「Enabled GPU types」で必ずAmpere/Hopper系(A40・A5000・A6000・A100・H100・L4)のみに限定し、
#    Blackwell系(RTX 5090・RTX PRO 4500/5000/6000)を除外すること（本README「GPU世代要件」参照）。
#    (Web UI操作。詳細はRunPod公式ドキュメント参照: https://docs.runpod.io/serverless/workers/deploy )
```

RunPod ConsoleでEndpoint作成後に発行されるエンドポイントURL
（`https://api.runpod.ai/v2/<endpoint-id>/runsync` 相当）を
`TRELLIS2_SELFHOST_INFERENCE_ENDPOINT_URL` に設定します。

**着手前に確認しておくこと**（TripoSplatのデプロイ実績から得た教訓。同じ落とし穴が起こりうる）:

- `runpod/base:1.0.7-cuda1240-ubuntu2204`（本Dockerfileの`FROM`タグ）は
  **Docker Hub上での実在確認ができていない**（TripoSplatのタグ`1.0.7-cuda1290-ubuntu2404`から
  の類推）。実ビルド前に`hub.docker.com/v2/repositories/runpod/base/tags`で実在タグを
  必ず確認すること。
- RunPodのGitHubビルド機能を使う場合、ビルドコンテキストのルート設定（RunPod Console上の
  「Root Directory」設定）と`COPY runpod/handler.py .`等の相対パスの整合を確認すること
  （TripoSplat Dockerfileの「2.5」節の教訓と同じ）。
- ビルド時焼き込みステップ（`Trellis2ImageTo3DPipeline.from_pretrained(...)`を1回実行）が、
  gated repoへのアクセス承認完了前に実行されると401/403で失敗する。HuggingFaceの
  アクセス承認（特に`facebook/dinov3-vitl16-pretrain-lvd1689m`のMeta手動レビュー）が
  完了していることを確認してからビルドすること。401は「トークン無効/未設定」、
  403で"awaiting a review"は「トークンは有効だが未承認」を意味する（AAA.11.1実機検証済み）。
- モデル重み(計16GB・22ファイル)のビルド時ダウンロードは、TripoSplat(3.6GB)より遥かに
  大きい。イメージビルド時間・レジストリpush時間・Serverlessコールドスタート時間への
  影響が大きくなる見込み（未検証・要実測）。

## 4. R2アップロード必須（RunPod payload size limit教訓の再確認）

**生成物（GLB）はhandler内でCloudflare R2へ実アップロードし、レスポンスにはURLのみを返す
実装を必須とする。** TripoSplatデプロイ（`infra/triposplat-selfhost/README.md`「2.5」節）で、
base64ペイロード返却がRunPodのpayload size limit（`/run`=10MB・`/runsync`=20MB）に抵触し
`output`が黙って欠落するという重大な問題が実際に発生・修正済みである。TRELLIS.2のGLB出力は
41.5MB（AAA.11.1実測）であり、この制限を大幅に超過するため、`runpod/handler.py`は最初から
R2への実アップロード実装（TripoSplatの`upload_ply_to_r2`と同一パターンの`upload_glb_to_r2`）
になっている。

必要な環境変数（RunPod Endpointの環境変数として設定。値はコードへ書かない）:

```bash
CLOUDFLARE_R2_ENDPOINT=
CLOUDFLARE_R2_ACCESS_KEY=
CLOUDFLARE_R2_SECRET_KEY=
CLOUDFLARE_R2_BUCKET=
CLOUDFLARE_R2_PUBLIC_URL=
```

## 5. 結線方法（TypeScript側 Provider）

デプロイ後、`TRELLIS2_SELFHOST_INFERENCE_ENDPOINT_URL` 環境変数にそのエンドポイントURLを
設定するだけで、`lib/mesh3d/src/providers/TrellisSelfHostProvider.ts` が自動的に
プレースホルダー動作（`isPlaceholder:true`）から実接続へ切り替わります（環境変数未設定時は
安全にプレースホルダーのまま縮退する設計、AAA.11.4）。

```bash
# .env（環境変数の追記先はデプロイ環境に応じて調整すること。CLAUDE.mdの環境変数節への
# 反映は本タスクのスコープ外。ユーザー自身の判断で行うこと）
TRELLIS2_SELFHOST_INFERENCE_ENDPOINT_URL=https://api.runpod.ai/v2/<endpoint-id>/runsync
TRELLIS2_SELFHOST_INFERENCE_API_TOKEN=xxxxx   # 任意。エンドポイント保護用トークンを設定した場合のみ
```

さらに、`migrations/006_seed_trellis2_selfhost.sql`でseedした`engine_provider_settings`の
`('mesh_engine', 'trellis2_selfhost')`行は初期`enabled=false`です。**本タスクではこの
migrationの実DBへの適用は行っていません**（プロジェクトオーナーが別途承認・実行すること）。
PoC完了・本採用確定後は、管制盤（Part AAA）または直接SQLで`enabled=true`へ更新してください
（`qa_status`の`pass`への昇格運用はAAA.2・Part UUのQAゲート思想を踏襲し、人間承認を経ること）。

## 6. 未実装・要フォローアップ事項（正直な棚卸し・2026-07-13時点）

- **本Dockerfile/handler.pyの実ビルド・RunPod Serverlessへの実デプロイはまだ行っていない**。
  TripoSplatが辿った「ベースイメージタグ実在確認」「COPYパスの基準」「クラッシュループ」
  「pipの`uninstall-no-record-file`エラー」等の落とし穴が、本Dockerfileでも再発する可能性がある。
- `runpod/base:1.0.7-cuda1240-ubuntu2204`タグの実在はDocker Hub API等で未確認（本README
  「3. デプロイコマンド」節参照）。
- コールドスタート時間はTripoSplat（3.6GB重み）より大きいモデル重み（16GB）のため、
  TripoSplatの実測（約7〜41秒）より大幅に長くなる見込みだが未実測。
- 1回あたりの実測コスト（円換算）は未計測。
- `MESH_SELFHOST_MONTHLY_CALL_LIMIT`（AAA.11.4(d)の環境変数草案）による月間呼び出し回数
  ガード、および`api_cost_logs`（provider='runpod'）への実コスト記録は未実装
  （本採用確定後の別タスク）。
- 物件写真（外観・室内）での出力品質は未評価。TRELLIS.2はオブジェクト中心の生成が
  想定されており、不動産シーン全体の再構成への適合性はPoCで見極める必要がある。
- `hunyuan3d_selfhost`/`sf3d_selfhost`/`spar3d_selfhost`（ADR-326〜327）は設計のみで、
  本タスクのスコープ外（別タスク・保留、AAA.11.6参照）。
