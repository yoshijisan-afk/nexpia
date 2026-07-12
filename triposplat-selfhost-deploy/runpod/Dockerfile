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
# runpod/pytorchベースイメージを使う（依頼内容記載の標準パターン）。torch 2.4.1+cu124相当の
# タグを使用（2026-07-12実機検証済みの組み合わせに近いもの。厳密なパッチバージョン一致は
# 不要で、最新の安定版でも問題ないことをREADME側で確認済み）。
FROM runpod/pytorch:2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# TripoSplatリポジトリのクローン先（2026-07-12実機検証済み: このリポジトリで
# `from triposplat import TripoSplatPipeline` が解決することを確認済み）。
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSplat.git /opt/triposplat

COPY requirements.txt .
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

COPY handler.py .

CMD ["python", "-u", "handler.py"]
