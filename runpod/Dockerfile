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
# 2026-07-12再訂正（クラッシュループの根本原因）: `runpod/pytorch`ベースイメージへの差し替え後、
# ビルドは成功したが、実際のServerlessワーカーが起動直後に「exit code 1」でクラッシュループする
# 問題が発生した。RunPod Pod上でhandler.pyを直接実行して原因を特定した結果、
# runpod Python SDK(worker.py `_is_local()`)が
# 「`RUNPOD_WEBHOOK_GET_JOB`環境変数が未設定 → ローカルテストモードとみなし、
# test_input.jsonが無いのでexit(1)」という判定をしていることが判明した。
# `RUNPOD_WEBHOOK_GET_JOB`はRunPod基盤が本番Serverlessコンテナへ自動注入するはずの変数だが、
# `runpod/pytorch`はPod(常時起動・対話開発。Jupyter/SSH向け)用の汎用イメージであり、
# 公式のServerless用テンプレート(`runpod-workers/worker-template`)が使う
# `runpod/base`(Serverless専用の軽量イメージ、ENTRYPOINT無し)とは別系統である。
# `runpod/pytorch`側の初期化処理(ENTRYPOINT相当)が環境変数の伝搬を妨げている可能性が高いと
# 判断し、公式テンプレートと同じ`runpod/base`系列へ切り替えた。
# Docker Hub API(`hub.docker.com/v2/repositories/runpod/base/tags`)で実在確認済みのタグ
# （CUDA 12.9.0・Ubuntu 24.04。dev-nix系ではない方＝Serverless向けの想定）を使用する。
# `runpod/base`はPyTorchを含まないため、requirements.txtにtorch/torchvisionを再度追加した。
# ただし本切り替え後の実ビルド・実デプロイでの動作確認はまだ行っていない（このコミット時点）。
FROM runpod/base:1.0.7-cuda1290-ubuntu2404

# runpod/base公式テンプレート(runpod-workers/worker-template)に倣い、python3を
# /usr/local/bin/python・python3として明示的にシンボリックリンクする（runpod/pytorchと異なり
# 既定で`python`コマンドが用意されている保証がないため）。
RUN ln -sf $(which python3) /usr/local/bin/python && \
    ln -sf $(which python3) /usr/local/bin/python3

# runpod/pytorchより最小構成の可能性があるため、git・pipが確実に使える保証がなく、
# 念のため明示的にインストールする（既に入っていれば何もしない）。
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TripoSplatリポジトリのクローン先（2026-07-12実機検証済み: このリポジトリで
# `from triposplat import TripoSplatPipeline` が解決することを確認済み）。
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSplat.git /opt/triposplat

# 2026-07-12訂正: RunPodのGitHubビルドはリポジトリルートをビルドコンテキストとして
# `docker buildx build --file runpod/Dockerfile <repoルート>` の形で実行するため（実ビルドログの
# コマンドで確認）、COPYの参照元はDockerfileからの相対パスではなく、コンテキストルート
# （＝リポジトリルート）からの相対パスで指定する必要がある。
COPY runpod/requirements.txt .
# 2026-07-12再訂正: `runpod/base`にはOS(Debian/apt)経由で`cryptography`等が事前導入されており、
# pipのRECORDファイルを持たないため、pipが「安全にアンインストールできない」として
# `uninstall-no-record-file`エラーで失敗した（実ビルドログで確認）。`--ignore-installed`で
# 既存インストールとの衝突チェックをスキップし、pip管理下のバージョンをsite-packages内に
# 新規インストールさせる（OS側のパッケージを壊さない、pipエコシステムでの定番回避策）。
RUN python -m pip install --no-cache-dir --ignore-installed -r requirements.txt

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
