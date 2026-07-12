# infra/triposplat-selfhost/runpod/Dockerfile
#
# Part BBB（Gaussian Splatパイプライン拡張）BBB.13・ADR-320 PoC (J-6a): RunPod Serverless
# 用のTripoSplat推論コンテナイメージ。
#
# 実機検証済み情報（2026-07-12・RunPod L4 Pod上での実行で確認）: TripoSplatのAPI契約・
# 依存パッケージ・モデル重み取得方法（HuggingFace認証トークン不要）はhandler.pyのとおり
# 実機確認済み。ただし本Dockerfile自体のビルド・RunPod **Serverless**への実デプロイは
# していない（RunPodアカウント・課金が必要なため）。ローカルPod実機では動作確認済みだが、
# Serverlessのコンテナビルド環境固有の問題（ビルド時間・ネットワーク制約等）までは
# 検証できていない。infra/triposplat-selfhost/README.md のデプロイ手順を参照すること。
#
# runpod/pytorchベースイメージを使う（依頼内容記載の標準パターン）。
#
# 2026-07-12訂正: 当初指定していた`2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04`はDocker Hub上に
# 実在しないタグで、RunPodの実ビルドで
# 「failed to resolve source metadata ... not found」として失敗した（実機PodではPyTorch
# 2.4.1+cu124が動いていたが、それはPodの既定環境であり、runpod/pytorch Hubイメージの
# タグとは別物だった）。Docker Hub API(`hub.docker.com/v2/repositories/runpod/pytorch/tags`)
# で実在するタグを直接確認し、現行の命名規則(`X.X.X-dev-nix-cuXXXX-torchXXX-ubuntuXXXX`)に
# 沿った、検証時点で最も古い(=2.4.1に最も近い)組み合わせである torch 2.6.0 / CUDA 12.8.1 /
# Ubuntu 22.04 のタグへ差し替えた。TripoSplatはPyTorchの厳密なバージョン固定を要求しない
# （README「install torch and torchvision according to your environment」）ため、
# 2.4.1→2.6.0の差は問題にならない想定。ただし本タグでの実ビルド確認はまだ行っていない。
FROM runpod/pytorch:1.0.7-dev-nix-cu1281-torch260-ubuntu2204

WORKDIR /app

# TripoSplatリポジトリのクローン先（2026-07-12実機検証済み: このリポジトリで
# `from triposplat import TripoSplatPipeline` が解決することを確認済み）。
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSplat.git /opt/triposplat

# 2026-07-12訂正: RunPodのGitHubビルドはリポジトリルートをビルドコンテキストとして
# `docker buildx build --file runpod/Dockerfile <repoルート>` の形で実行するため（実ビルドログの
# コマンドで確認）、COPYの参照元はDockerfileからの相対パスではなく、コンテキストルート
# （＝リポジトリルート）からの相対パスで指定する必要がある。
COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデル重み(計3.6GB)をビルド時に1回だけダウンロードし、イメージへ焼き込む（実行時に都度
# ダウンロードすると起動が遅くなるため）。HuggingFace認証トークンは不要（2026-07-12実機検証済み）。
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='VAST-AI/TripoSplat', \
    local_dir='/opt/triposplat/ckpts', \
    allow_patterns=['diffusion_models/*', 'vae/*', 'clip_vision/*', 'background_removal/*'], \
)"

COPY runpod/handler.py .

CMD ["python", "-u", "handler.py"]
