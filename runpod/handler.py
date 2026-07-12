# infra/triposplat-selfhost/runpod/handler.py
#
# Part BBB（Gaussian Splatパイプライン拡張）BBB.13・ADR-320 PoC (J-6a):
# TripoSplat（github.com/VAST-AI-Research/TripoSplat・MIT License）を RunPod Serverless
# のjobハンドラとしてデプロイするスクリプト。
#
# ============================================================================
# 実機検証済み情報（2026-07-12・このセッションでRunPod L4 Pod上で実際にTripoSplatを動かして
# 確認済み）:
#   TripoSplatPipelineのコンストラクタ引数・pipe.run()の入出力契約・モデル重みの取得方法
#   （huggingface_hub.snapshot_download・認証トークン不要）・実測レイテンシ（262144 Gaussianで
#   約28〜30秒、20ステップのdiffusion sampling、GPU使用量はピーク時約6.6GB）は、下記コードの
#   とおり実機確認済み。
#   ただし RunPod **Serverless**（本ハンドラのデプロイ先）への実デプロイ・実行検証はしていない
#   （RunPodアカウント・課金が必要なため、実装担当エージェントの権限では実行不可）。今回の実測は
#   RunPod **Pod**（常時起動インスタンス）上でのものであり、Serverlessのコールドスタート込みの
#   挙動は別途未検証（infra/triposplat-selfhost/README.md参照）。
# ============================================================================
#
# RunPodのjob契約（RunPod標準形式・BBB.13.5の依頼どおり）:
#   input:  {"input": {"image_url": string, "num_gaussians"?: number}}
#   output: {"ply_url": string, "gaussian_count": number}
#   （lib/mesh3d/src/providers/TripoSplatSelfHostProvider.ts が期待する
#    { ply_url, gaussian_count } 契約と同じキー名にして、Modal版とレスポンス形式を揃える）
#
# モデルロードは「RunPodのjobを受けてモデル推論、GPU上にモデルを一度ロードして
# 使い回す」パターン（依頼内容記載のとおり）。RunPod Serverlessではワーカープロセスの
# グローバルスコープでモデルをロードし、handler()呼び出しのたびに再ロードしない。
#
# R2アップロードについて: 2026-07-12実機デプロイで判明した重大な訂正。当初はbase64データURLで
# 返す簡易実装だったが、RunPod公式の payload size limit（/run=10MB、/runsync=20MB）に
# 262144 Gaussianのbase64化PLY（約24MB）が抵触し、handler()自体は正常完了(executionTime
# 実測23秒台で安定)するにもかかわらず、output全体がAPIレスポンスから黙って欠落するという
# 実害を実機で確認した(https://www.answeroverflow.com/m/1199970565381967982 等、RunPod公式の
# 「大きな結果はクラウドストレージへ保存しリンクを返すこと」という推奨に反していた)。
# そのためR2への実アップロードを実装した。CloudflareのR2はS3互換APIのため、boto3の
# S3クライアントをR2のエンドポイントへ向けて使う。認証情報はRunPodエンドポイントの
# 環境変数として別途設定する必要がある(CLOUDFLARE_R2_ENDPOINT/ACCESS_KEY/SECRET_KEY/
# BUCKET/PUBLIC_URL。CLAUDE.md記載のSisliR既存の命名規則と統一)。

import io
import os
import tempfile
import uuid
from typing import Any

import boto3
import requests
import runpod

# TripoSplatリポジトリのクローン先（Dockerfileでビルド時にclone・snapshot_download済み）。
# ckpt_pathの相対パス契約（modal/app.pyと同じ）に合わせ、このディレクトリをカレントに
# してからモデルをロードする。
TRIPOSPLAT_DIR = "/opt/triposplat"

# グローバルスコープで1回だけロードし、ワーカープロセスの生存期間中モデルを保持する
# （RunPod Serverlessの標準パターン: モデルロードをコールドスタート時の1回に限定する）。
_pipeline = None


def load_model() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import sys

    sys.path.insert(0, TRIPOSPLAT_DIR)
    os.chdir(TRIPOSPLAT_DIR)

    from triposplat import TripoSplatPipeline

    # 2026-07-12実機検証済み（RunPod L4 Pod）: コンストラクタ引数はこのままの契約で動作した。
    _pipeline = TripoSplatPipeline(
        ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
        decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
        dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
        rmbg_path="ckpts/background_removal/birefnet.safetensors",
        device="cuda",
    )
    return _pipeline


def run_inference(image_bytes: bytes, num_gaussians: int) -> tuple[bytes, int]:
    """
    画像バイト列を受け取りPLYバイト列とGaussian数を返す。
    2026-07-12実機検証済み（RunPod L4 Pod）: pipe.run()は画像ファイルパス(文字列)を受け取る
    契約であり、URLやバイト列を直接は受け取らない。そのため一時ファイルへ書き出してから渡す。
    """
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipeline = load_model()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        image.save(tmp_in.name)
        input_path = tmp_in.name

    try:
        gaussian, _prepared = pipeline.run(
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

    # BBB.13.3: num_gaussiansは実際の生成数上限指定。gaussianオブジェクトが実際の生成数を
    # 返す属性を持つかは未確認のため、リクエストで指定したnum_gaussiansをそのまま返す。
    return ply_bytes, num_gaussians


def upload_ply_to_r2(ply_bytes: bytes) -> str:
    """
    生成したPLYをCloudflare R2へアップロードし、公開URLを返す。
    R2はS3互換APIのため、boto3のS3クライアントをR2のエンドポイントへ向けて使う
    （2026-07-12実装。CLAUDE.md「環境変数」節のCLOUDFLARE_R2_*命名規則と統一）。
    """
    endpoint = os.environ["CLOUDFLARE_R2_ENDPOINT"]
    access_key = os.environ["CLOUDFLARE_R2_ACCESS_KEY"]
    secret_key = os.environ["CLOUDFLARE_R2_SECRET_KEY"]
    bucket = os.environ["CLOUDFLARE_R2_BUCKET"]
    public_url = os.environ["CLOUDFLARE_R2_PUBLIC_URL"]

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    key = f"triposplat-selfhost/{uuid.uuid4()}.ply"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=ply_bytes,
        ContentType="application/octet-stream",
    )
    return f"{public_url.rstrip('/')}/{key}"


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod Serverlessのjobエントリポイント。
    job = {"input": {"image_url": "https://...", "num_gaussians": 262144}}
    """
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    if not image_url or not isinstance(image_url, str):
        return {"error": "image_url is required"}

    # BBB.13.3: 262144が既定（32768/65536/131072/262144から選択可）。
    num_gaussians = job_input.get("num_gaussians", 262_144)
    if num_gaussians not in (32_768, 65_536, 131_072, 262_144):
        return {"error": "num_gaussians must be one of 32768, 65536, 131072, 262144"}

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        ply_bytes, gaussian_count = run_inference(response.content, num_gaussians)

        ply_url = upload_ply_to_r2(ply_bytes)

        return {"ply_url": ply_url, "gaussian_count": gaussian_count}
    except Exception as e:  # noqa: BLE001 - RunPod handlerはエラーをJSONで返す契約
        return {"error": str(e)}


# RunPod Serverlessのワーカーエントリポイント。
runpod.serverless.start({"handler": handler})
