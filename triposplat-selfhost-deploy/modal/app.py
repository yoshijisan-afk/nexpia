# infra/triposplat-selfhost/modal/app.py
#
# Part BBB（Gaussian Splatパイプライン拡張）BBB.13・ADR-320 PoC (J-6a):
# TripoSplat（github.com/VAST-AI-Research/TripoSplat・MIT License・HuggingFace
# VAST-AI/TripoSplat）を Modal 上にサーバーレスGPUとしてデプロイするスクリプト。
#
# ============================================================================
# 実機検証済み情報（2026-07-12・RunPod L4 Pod上での実行で確認）:
#   TripoSplatの実際のAPI契約・依存パッケージ・モデル重みの取得方法・実測レイテンシは、
#   本セッションでRunPod L4 Pod上で実際にTripoSplatを動かして確認済み（下記コード参照）。
#   ただし Modal 環境そのものへの実デプロイ・実行検証はしていない（Modalアカウント・
#   課金が必要なため、実装担当エージェントの権限では実行不可）。
#   ローカルPod実機では動作確認済みだが、Modal固有のコンテナビルド時間・ネットワーク制約
#   （例: git clone/HuggingFace Hubへの到達性、ビルド時snapshot_downloadの所要時間）までは
#   検証できていない。デプロイ手順は infra/triposplat-selfhost/README.md を参照すること。
# ============================================================================
#
# レスポンス契約（lib/mesh3d/src/providers/TripoSplatSelfHostProvider.ts と対になる自作の
# シンプルなJSON契約。デプロイ側=本ファイルがこの契約に従う。TS側は変更不要・契約一致確認済み）:
#   POST /infer
#   body: { "image_url": string, "num_gaussians"?: number }
#   200 response: { "ply_url": string, "gaussian_count": number }
#
# R2アップロードについて: 本実装はシンプル化のため、生成したPLYファイルを
# Modalのコンテナ内エフェメラルボリューム(/tmp)に一時保存し、Modalが提供する
# 一時公開URL相当の仕組み（`modal.Volume`経由のダウンロードURLなど）を使わず、
# base64データURLとして返す簡易実装に留める（依頼内容どおりスコープ外のまま）。本番運用で
# Cloudflare R2へ永続化する処理（lib/mesh3d/src/SplatSupplyService.tsが期待するplyUrl/spzUrlの
# 永続ストレージURL化）は別途実装すること（PoC完了後・本採用が決まった段階の宿題として明記する）。
#
# モデルロードはModalの標準パターンに従い、コンテナ起動時（GPU起動時）に1回だけ行う
# （@app.cls + @modal.enter()）。
#
# CLAUDE.md 絶対ルール(TypeScript側コードに適用。本Pythonスクリプトは対象外)。
# ただし可読性のため、SisliR側の設計ドキュメント参照コメント規約は踏襲する。

import base64
import io
import os
import tempfile
from typing import Any

import modal

app = modal.App("sislir-triposplat-selfhost")

# BBB.13.3: 生成処理自体で約6.6GB VRAM・モデル込みで計8GB以上（consumer-grade GPU 8GB+で
# 動作、L4クラスで十分）。BBB.13.4の比較方針どおりModalの最安帯（L4・$0.80/hr）を使う。
# 2026-07-12実機検証（RunPod L4 Pod）でもピークVRAM使用量は約6.6GBで、L4で十分動作することを
# 確認済み。
GPU_TYPE = "L4"

# TripoSplatのモデル重み格納先。ビルド時（download_triposplat_weights）と実行時
# （load_model内のckpt_path指定）で一致させる必要がある。実機検証したAPI例（依頼内容記載）は
# カレントディレクトリからの相対パス "ckpts/..." を渡す契約だったため、実行時にこのディレクトリへ
# chdirしてから TripoSplatPipeline を構築する。
TRIPOSPLAT_DIR = "/opt/triposplat"


def download_triposplat_weights() -> None:
    """
    ビルド時（イメージレイヤーとして）にTripoSplatのモデル重み(計3.6GB)を取得し、
    イメージへ焼き込む。2026-07-12実機検証済み: HuggingFace認証トークンは不要
    （snapshot_downloadの引数はそのまま実機で成功した呼び出しと同一）。
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="VAST-AI/TripoSplat",
        local_dir=f"{TRIPOSPLAT_DIR}/ckpts",
        allow_patterns=[
            "diffusion_models/*",
            "vae/*",
            "clip_vision/*",
            "background_removal/*",
        ],
    )


# 依存パッケージは2026-07-12にRunPod L4 Pod上で実機動作確認済みの組み合わせ
# （torch 2.4.1+cu124で確認。新規コンテナでは最新の安定版で問題ない）。
# READMEに明記の通り transformers・diffusers は不要（TripoSplat独自実装）。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "safetensors",
        "pillow",
        "tqdm",
        "huggingface_hub",
        "requests",
        "fastapi[standard]",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/VAST-AI-Research/TripoSplat.git {TRIPOSPLAT_DIR}",
    )
    # モデル重み(計3.6GB)をビルド時に1回だけダウンロードし、イメージへ焼き込む
    # （実行時に都度ダウンロードすると起動が遅くなるため。依頼内容の設計方針どおり）。
    .run_function(download_triposplat_weights)
)


@app.cls(gpu=GPU_TYPE, image=image, timeout=300, scaledown_window=120)
class TripoSplatModel:
    """
    Modalの標準パターン: モデルロードをコンテナ(GPU)起動時の1回に限定する
    （@modal.enter()フック）。同一コンテナへの後続リクエストはロード済みモデルを再利用する。
    """

    @modal.enter()
    def load_model(self) -> None:
        # 2026-07-12実機検証済み（RunPod L4 Pod）: TripoSplatPipelineのコンストラクタ引数・
        # ckpt_path等の相対パス契約は依頼内容記載のコードで確認済み。ckpt_pathがビルド時に
        # chdir("/opt/triposplat")したカレントディレクトリからの相対パスであるため、
        # ロード前に明示的にchdirする。
        import sys

        sys.path.insert(0, TRIPOSPLAT_DIR)
        os.chdir(TRIPOSPLAT_DIR)

        from triposplat import TripoSplatPipeline

        self.pipeline = TripoSplatPipeline(
            ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
            decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
            dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
            flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
            rmbg_path="ckpts/background_removal/birefnet.safetensors",
            device="cuda",
        )

    def run_inference(self, image_bytes: bytes, num_gaussians: int) -> tuple[bytes, int]:
        """
        画像バイト列を受け取りPLYバイト列とGaussian数を返す。
        2026-07-12実機検証済み（RunPod L4 Pod）: pipe.run()は画像ファイルパス(文字列)を
        受け取る契約であり、URLやバイト列を直接は受け取らない。そのためここで一時ファイルへ
        書き出してから渡す。
        """
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            image.save(tmp_in.name)
            input_path = tmp_in.name

        try:
            gaussian, _prepared = self.pipeline.run(
                input_path, num_gaussians=num_gaussians, show_progress=True
            )
        finally:
            os.unlink(input_path)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_out:
            output_path = tmp_out.name
        try:
            gaussian.save_ply(output_path)
            with open(output_path, "rb") as f:
                ply_bytes = f.read()
        finally:
            os.unlink(output_path)

        # BBB.13.3: num_gaussiansは262144が既定で、生成されるGaussian数の上限指定。
        # gaussianオブジェクトが実際の生成数を返す属性を持つかは未確認のため、リクエストで
        # 指定したnum_gaussiansをそのまま返す（実際の生成数と厳密には異なる可能性がある点に注意。
        # 正確なGaussian数の取得方法が判明した場合は本行を更新すること）。
        return ply_bytes, num_gaussians

    @modal.fastapi_endpoint(method="POST", docs=True)
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        lib/mesh3d/src/providers/TripoSplatSelfHostProvider.ts が呼び出すエンドポイント。
        契約: { "image_url": string, "num_gaussians"?: number }
              -> { "ply_url": string, "gaussian_count": number }
        """
        import requests

        image_url = payload.get("image_url")
        if not image_url or not isinstance(image_url, str):
            return {"error": "image_url is required"}

        # BBB.13.3: 262144が既定（32768/65536/131072/262144から選択可）。
        num_gaussians = payload.get("num_gaussians", 262_144)
        if num_gaussians not in (32_768, 65_536, 131_072, 262_144):
            return {
                "error": "num_gaussians must be one of 32768, 65536, 131072, 262144",
            }

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        ply_bytes, gaussian_count = self.run_inference(response.content, num_gaussians)

        # 簡易実装: R2アップロードは前回同様スコープ外（本ファイル冒頭の注記参照）。
        # ここではdata URLとして返す（PoC比較用途のみ。本番運用ではR2永続化URLへ置換すること）。
        ply_base64 = base64.b64encode(ply_bytes).decode("ascii")
        ply_url = f"data:application/octet-stream;base64,{ply_base64}"

        return {"ply_url": ply_url, "gaussian_count": gaussian_count}


@app.local_entrypoint()
def main() -> None:
    """
    ローカル動作確認用エントリポイント（`modal run infra/triposplat-selfhost/modal/app.py`）。
    Modal環境への実デプロイ・実行検証はしていない。デプロイ手順は README.md を参照。
    """
    print(
        "This is a placeholder local entrypoint. Use `modal deploy "
        "infra/triposplat-selfhost/modal/app.py` to deploy the /infer endpoint, "
        "then set TRIPOSPLAT_INFERENCE_ENDPOINT_URL to the deployed URL. "
        "See infra/triposplat-selfhost/README.md."
    )
