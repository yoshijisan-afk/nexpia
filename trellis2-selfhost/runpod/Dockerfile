# syntax=docker/dockerfile:1
# infra/trellis2-selfhost/runpod/Dockerfile
#
# Part AAA（パイプライン管制盤＋プリセット）AAA.11.1・ADR-325: RunPod Serverless用の
# TRELLIS.2-4B（microsoft/TRELLIS.2、MIT License）推論コンテナイメージ。
#
# 実機検証済み情報（2026-07-13・RunPod A40 Pod上での実行で確認。設計書AAA.11.1参照）:
#   TRELLIS.2の推論API・依存関係インストールの3つの罠・GPU世代要件（Ampere/Hopper限定）・
#   gated HuggingFaceリポジトリ2件の認証要否は、以下のとおり実機確認済み。
#   ただし本Dockerfile自体のビルド・RunPod **Serverless**への実デプロイはしていない
#   （RunPodアカウント・課金操作が必要なため、実装担当エージェントの権限では実行不可）。
#   実測はRunPod **Pod**（常時起動インスタンス、直接python実行）上でのものであり、
#   Serverlessのコンテナビルド環境固有の問題（ビルド時間・ネットワーク制約等）や、
#   本Dockerfile自体の`docker build`成功可否はまだ検証できていない
#   （infra/triposplat-selfhost/のDockerfileが実際に辿った「ベースイメージタグが実在しない」
#   「COPYパスがビルドコンテキスト基準」等の落とし穴がここにも起こりうる。
#   実際のビルド前に必ずDocker Hub API等でタグの実在を再確認すること）。
#
# ベースイメージ: `runpod/base`系列を使う（`infra/triposplat-selfhost/runpod/Dockerfile`・
# README「2.5」節で実証済みの理由と同じ: `runpod/pytorch`はPod=常時起動の対話開発用イメージで
# あり、`RUNPOD_WEBHOOK_GET_JOB`環境変数の伝搬を妨げServerlessワーカーがexit code 1で
# クラッシュループする問題が実際に発生した。公式Serverless専用テンプレート
# （`runpod-workers/worker-template`）が使う`runpod/base`系列を使うこと）。
#
# 2026-07-14訂正: 当初の`1.0.7-cuda1240-ubuntu2204`は類推で書いた未検証タグで、実際には
# Docker Hub上に存在しないことを確認した（`cuda1240`/`cuda1241`系タグは`runpod/base`の
# 1.0.7系には存在せず、cuda1281/cuda1290/cuda1300のみが存在する。2026-07-14実際に
# `hub.docker.com/v2/repositories/runpod/base/tags`をAPI照会して確認済み）。TRELLIS.2は
# `torch==2.6.0+cu124`を要求する（AAA.11.1実機検証済み）が、TripoSplat
# （`infra/triposplat-selfhost/runpod/Dockerfile`）も同様にbase imageのCUDAツールキット
# バージョンと実際にpipインストールするtorchのCUDAバージョンが完全一致しない状態で実デプロイに
# 成功しているため（拡張ビルドに使うnvccはtorchが要求するcu124系ヘッダとの互換性さえあれば
# 良く、実行時ランタイムはpipでインストールしたtorchが自前でバンドルするcu124系共有ライブラリを
# 使うため）、TripoSplatで実際に動作確認済みの`1.0.7-cuda1290-ubuntu2404`をそのまま使う
# （新規に未検証のタグの組み合わせを増やさない、実績のあるものを流用する方針）。
FROM runpod/base:1.0.7-cuda1290-ubuntu2404

# runpod/base公式テンプレートに倣い、python3を/usr/local/bin/python・python3として
# 明示的にシンボリックリンクする（runpod/pytorchと異なり、既定で`python`コマンドが
# 用意されている保証がないため。TripoSplat Dockerfileと同じ対処）。
RUN ln -sf $(which python3) /usr/local/bin/python && \
    ln -sf $(which python3) /usr/local/bin/python3

# git・pipを明示的にインストールする（runpod/baseが最小構成の可能性があり、確実に
# 使える保証がないため。TripoSplat Dockerfileと同じ対処）。
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# microsoft/TRELLIS.2リポジトリのクローン（AAA.11.1: リポジトリ名はハイフンではなく
# ピリオド「TRELLIS.2」。ハイフン表記「TRELLIS-2」で検索すると誤って401になった経緯が
# あるため注意）。
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS2

WORKDIR /opt/TRELLIS2

# ── 既知の罠(3) の対処: DINOv3属性パスのパッチ (AAA.11.1実機検証済み) ──
# transformers==5.13.1のDINOv3実装は`DINOv3ViTModel.model.layer`という入れ子構造だが、
# TRELLIS.2の image_feature_extractor.py は `self.model.layer` という平坦な旧構造を
# 前提にしており AttributeError になる。setup.sh実行前にパッチを適用する
# （クローン直後・パッケージ化前、依頼どおりの適用タイミング）。
RUN sed -i 's/self\.model\.layer/self.model.model.layer/' \
    trellis2/modules/image_feature_extractor.py

# ── 既知の罠(4)の対処: ビルド環境にGPUが無いためCUDAアーキテクチャの自動検出が失敗する ──
# 2026-07-14実際のRunPod GitHubビルド(GPU無しのビルド専用マシン)で新たに発覚:
# o_voxel/cumesh/flex_gemmのビルド時、torchが`torch.cuda.get_device_capability()`で
# 現在のGPUのアーキテクチャを自動検出しようとするが、ビルド環境にGPUが接続されていないため
# 検出結果が空リストになり、`torch/utils/cpp_extension.py`の`_get_cuda_arch_flags()`内で
# `arch_list[-1] += '+PTX'`が`IndexError: list index out of range`でクラッシュする
# （RunPod **Pod**上での対話的な実機検証ではGPUが常時接続されているため発覚しなかった、
# ビルド専用マシン特有の問題）。`TORCH_CUDA_ARCH_LIST`を明示的に設定し、GPU検出に頼らず
# 対象アーキテクチャ（Ampere sm_80/86・Hopper sm_90。AAA.11.1のGPU世代要件と一致させる）を
# 指定することで回避する。
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# ── 既知の罠(5)の対処: setup.shがGPU不在のビルド環境で即座に失敗する ──
# 2026-07-14実際のビルドで新たに発覚: setup.shは冒頭で`command -v nvidia-smi`のみを見て
# CUDA/ROCmプラットフォームを判定しており、存在しない場合は"Error: No supported GPU found"で
# 即座にexit 1する（--flash-attn/--nvdiffrast/--nvdiffrec等の個別フラグ処理に到達する前に
# スクリプト全体が終了する。setup.sh自体は`nvidia-smi`の出力を一切パースしておらず、
# 単なる存在確認のみのため、実体を伴わないダミーの`nvidia-smi`をPATH上に置くだけで
# この判定を通過できる）。setup.sh実行後、実行時に本物の`nvidia-smi`（RunPod
# ServerlessワーカーがGPUと共に提供する）と衝突しないよう、このダミーは直後に削除する。
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi

# ── 公式インストールスクリプト実行 (--new-envなし。condaは不要、system Pythonへ直接) ──
# AAA.11.1実機検証済み: --flexgemm ステップがtorchを意図せず2.13.0(cu13.0ビルド)へ上書きする
# 副作用があり、o_voxelのビルドがCUDAバージョン不一致で失敗する。setup.sh自体がここで
# エラー終了しても後続ステップで torch を明示的に再インストールし o_voxel を再ビルドするため、
# `|| true` でビルドを継続させる。
RUN bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm || true

# ダミーのnvidia-smiを削除し、実行時に本物と衝突しないようにする。
RUN rm -f /usr/local/bin/nvidia-smi

# ── 既知の罠(1) の対処: torch/torchvisionをcu124ビルドへ明示的に再ピン留め ──
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ── 既知の罠(2) の対処: ABI不整合を起こすtorchaudioを除去 ──
# 画像→3D生成にtorchaudioは不要（transformersのis_torchaudio_available()がその使用箇所を
# ガードしているため、単純な削除で問題ない。AAA.11.1実機検証済み）。
RUN python -m pip uninstall -y torchaudio || true

# ── 既知の罠(1) 続き: o_voxel拡張をcu124 torchに対して再ビルド ──
RUN mkdir -p /tmp/extensions && \
    cp -r /opt/TRELLIS2/o-voxel /tmp/extensions/o-voxel && \
    python -m pip install /tmp/extensions/o-voxel --no-build-isolation

# ── モデル重みはビルド時に焼き込まない（2026-07-14方針転換）──
# 当初はTripoSplatと同じくビルド時焼き込み（Dockerビルドシークレット経由でHF_TOKENを渡し
# パイプラインを1回構築させて強制ダウンロード）を狙っていたが、RunPodのGitHub連携ビルドが
# Docker BuildKitのビルド時シークレット（`--mount=type=secret`/`--secret`）をサポートしている
# という公式ドキュメント上の裏付けが取れなかった（2026-07-14調査）。ここでシークレットの
# 値をビルド引数(`--build-arg`)等で代用するとイメージレイヤー・ビルド履歴に平文で残る
# リスクがあるため採用しない。
# 安全側に倒し、モデル重み（TRELLIS.2-4B本体16GB + facebook/dinov3-vitl16-pretrain-lvd1689m
# + briaai/RMBG-2.0の2件のgated補助モデル）は**コンテナ起動時（handler.pyのload_model()内、
# 初回job実行時）に遅延ダウンロードする**方式に変更する。HF_TOKENはRunPod
# Serverlessエンドポイントの「Environment Variables」（実行時環境変数。CLOUDFLARE_R2_*と
# 同じ仕組み・既に実績あり）として設定する。
# トレードオフ: TripoSplat（3.6GB・認証不要・ビルド時焼き込み）と異なり、ワーカーの
# コールドスタート時に16GB超のダウンロードが発生するため起動が大幅に遅くなる
# （実測値は本番デプロイ後に確認すること）。将来的にRunPodのビルド時シークレット対応が
# 確認できた場合、または永続ネットワークボリュームでキャッシュを共有する設計に切替できた
# 場合は、ビルド時焼き込みへ戻すことを検討する。

# 2026-07-14訂正: RunPodの「Deploy from a GitHub repository」画面では、Dockerfile pathを
# 指定する単一のフィールドのみが存在し(ビルドコンテキストを別途指定するRoot Directory相当の
# フィールドは無い)、ビルドコンテキストは常にリポジトリルートになる（実際のUI確認・
# 2026-07-14）。デプロイ先(nexpia)の`trellis2-selfhost/runpod/Dockerfile`に対し、
# Dockerfile path欄には`trellis2-selfhost/runpod/Dockerfile`を指定する。
# よってCOPY元パスもリポジトリルートからのフルパスで指定する必要がある。
COPY trellis2-selfhost/runpod/requirements.txt .
# runpod/baseにOS(apt)経由で事前導入済みのパッケージがpipのRECORDファイルを持たず
# `uninstall-no-record-file`エラーになる問題への対処（TripoSplat Dockerfileと同じ理由・対処）。
RUN python -m pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY trellis2-selfhost/runpod/handler.py .

CMD ["python", "-u", "handler.py"]
