# infra/trellis2-selfhost/runpod/handler.py
#
# Part AAA（パイプライン管制盤＋プリセット）AAA.11.1・ADR-325:
# TRELLIS.2-4B（github.com/microsoft/TRELLIS.2・MIT License）をRunPod Serverlessの
# jobハンドラとしてデプロイするスクリプト。
#
# ============================================================================
# 実機検証済み情報（2026-07-13・RunPod A40 Pod上で実際にTRELLIS.2を動かして確認済み。
# 設計書AAA.11.1参照）:
#   Trellis2ImageTo3DPipeline.from_pretrained()の呼び出し方・pipeline.run()の入出力契約・
#   mesh.simplify()・o_voxel.postprocess.to_glb()によるGLB変換は、下記コードのとおり
#   実機確認済み（推論167.4秒、ピークVRAM 7.09GB、簡略化後GLB出力41.5MB）。
#   ただしRunPod **Serverless**（本ハンドラのデプロイ先）への実デプロイ・実行検証はしていない
#   （RunPodアカウント・課金が必要なため、実装担当エージェントの権限では実行不可）。今回の実測は
#   RunPod **Pod**（常時起動インスタンス）上でのものであり、Serverlessのコールドスタート込みの
#   挙動は別途未検証（infra/trellis2-selfhost/README.md参照）。
# ============================================================================
#
# RunPodのjob契約（RunPod標準形式。infra/triposplat-selfhost/runpod/handler.pyと同じ
# {"input": {...}}ラッピング規約に従う）:
#   input:  {"input": {"image_url": string}}
#   output: {"glb_url": string, "time_s": number, "asset_mb": number}
#   （lib/mesh3d/src/providers/TrellisSelfHostProvider.ts が期待する
#    { glb_url, time_s?, asset_mb? } 契約と同じキー名にする）
#
# モデルロードは「RunPodのjobを受けてモデル推論、GPU上にモデルを一度ロードして
# 使い回す」パターン（TripoSplat handler.pyと同じ構造）。パイプライン構築(重い処理)は
# ワーカープロセスのグローバルスコープで1回だけ行い、handler()呼び出しのたびには行わない。
#
# R2アップロード必須（AAA.11.4(c)。TripoSplatのpayload size limit教訓と同じ理由）:
# GLB出力は41.5MB(AAA.11.1実測)であり、RunPod公式のpayload size limit
# （/run=10MB、/runsync=20MB）を大幅に超過する。base64等でinline返却するとoutputが
# 黙って欠落するため、Cloudflare R2へ実アップロードしURLのみを返す
# （infra/triposplat-selfhost/runpod/handler.py と同じboto3 S3互換クライアントパターンを
# そのまま再利用する。認証情報はRunPodエンドポイントの環境変数として別途設定する
# 必要がある。CLOUDFLARE_R2_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET/PUBLIC_URL、
# CLAUDE.md記載のSisliR既存の命名規則と統一。値は環境変数にのみ設定し、コードへは
# 一切書かない）。

import io
import os
import tempfile
import time
import uuid
from typing import Any

# AAA.11.1実機検証済み: expandable_segmentsを有効にしないとメモリ断片化で
# 長時間実行時にOOMしやすい（Trellis2ImageTo3Dpipeline公式サンプルどおり、torch importより
# 前に設定する必要がある）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import boto3  # noqa: E402
import requests  # noqa: E402
import runpod  # noqa: E402

# TRELLIS.2リポジトリのクローン先（Dockerfileでビルド時にclone・パッチ済み）。
TRELLIS2_DIR = "/opt/TRELLIS2"

# 2026-07-14方針転換: RunPodのGitHub連携ビルドがDockerビルド時シークレットに対応している
# という裏付けが取れなかったため、モデル重み（TRELLIS.2-4B本体16GB +
# facebook/dinov3-vitl16-pretrain-lvd1689m + briaai/RMBG-2.0の2件のgated補助モデル）は
# ビルド時に焼き込まず、初回job実行時（コールドスタート時）にここで遅延ダウンロードする。
# huggingface_hubは`HF_TOKEN`環境変数を自動的に読む（明示的なlogin()呼び出し不要。
# 2026-07-13の実機検証で`env HF_TOKEN=... python3 verify_core.py`のみで認証が通ることを
# 確認済み）。RunPod Serverlessエンドポイントの環境変数として`HF_TOKEN`
# （facebook/dinov3-vitl16-pretrain-lvd1689m・briaai/RMBG-2.0の両方へアクセス承認済みの
# アカウントのread権限トークン）を設定しておくこと（CLOUDFLARE_R2_*と同じ実行時環境変数の
# 仕組み）。未設定の場合、gated repoへのアクセスで401/403エラーになる
# （AAA.11.1で実際に遭遇したエラーと同じ）。
TRELLIS2_MODEL_REPO_ID = "microsoft/TRELLIS.2-4B"

# nvdiffrastの制約(AAA.11.1の実測結果・簡略化後48.5万頂点/97.6万面が実運用相当)。
NVDIFFRAST_SIMPLIFY_TARGET = 16_777_216

# グローバルスコープで1回だけロードし、ワーカープロセスの生存期間中モデルを保持する
# （RunPod Serverlessの標準パターン。TripoSplat handler.pyと同じ構造）。
_pipeline: Any = None


def load_model() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import sys

    sys.path.insert(0, TRELLIS2_DIR)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # 2026-07-14方針転換によりここで初回ダウンロードが発生する（上部コメント参照）。
    # HF_TOKEN環境変数が未設定、またはgated repo 2件へのアクセス未承認の場合、
    # AAA.11.1で実際に遭遇したのと同じ401/403 GatedRepoErrorがここで発生する。
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS2_MODEL_REPO_ID)
    pipeline.cuda()
    _pipeline = pipeline
    return _pipeline


def run_inference(image_bytes: bytes) -> tuple[bytes, float]:
    """
    画像バイト列を受け取りGLBバイト列と推論時間(秒)を返す。
    2026-07-13実機検証済み（RunPod A40）: pipeline.run(image)はPIL.Imageを受け取り、
    mesh.simplify()・o_voxel.postprocess.to_glb()でGLBへ変換する（AAA.11.1の
    「検証済み推論API」節どおり）。
    """
    import o_voxel
    from PIL import Image

    started_at = time.monotonic()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipeline = load_model()

    mesh = pipeline.run(image)[0]
    # AAA.11.1実測: 簡略化前564万頂点/1145万面 → nvdiffrastの制約(16,777,216)に合わせて
    # 簡略化すると実運用相当の48.5万頂点/97.6万面になる。
    mesh.simplify(NVDIFFRAST_SIMPLIFY_TARGET)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_out:
        output_path = tmp_out.name
    try:
        glb.export(output_path, extension_webp=True)
        with open(output_path, "rb") as f:
            glb_bytes = f.read()
    finally:
        os.unlink(output_path)

    time_s = time.monotonic() - started_at
    return glb_bytes, time_s


def upload_glb_to_r2(glb_bytes: bytes) -> str:
    """
    生成したGLBをCloudflare R2へアップロードし、公開URLを返す。
    R2はS3互換APIのため、boto3のS3クライアントをR2のエンドポイントへ向けて使う
    （infra/triposplat-selfhost/runpod/handler.pyのupload_ply_to_r2()と同一パターン。
    CLAUDE.md「環境変数」節のCLOUDFLARE_R2_*命名規則と統一。値は環境変数からのみ読み、
    コードへは一切書かない）。
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
    key = f"trellis2-selfhost/{uuid.uuid4()}.glb"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=glb_bytes,
        ContentType="model/gltf-binary",
    )
    return f"{public_url.rstrip('/')}/{key}"


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod Serverlessのjobエントリポイント。
    job = {"input": {"image_url": "https://..."}}
    """
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    if not image_url or not isinstance(image_url, str):
        return {"error": "image_url is required"}

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        glb_bytes, time_s = run_inference(response.content)
        asset_mb = len(glb_bytes) / (1024 * 1024)

        glb_url = upload_glb_to_r2(glb_bytes)

        return {"glb_url": glb_url, "time_s": round(time_s, 1), "asset_mb": round(asset_mb, 2)}
    except Exception as e:  # noqa: BLE001 - RunPod handlerはエラーをJSONで返す契約
        return {"error": str(e)}


# RunPod Serverlessのワーカーエントリポイント。
runpod.serverless.start({"handler": handler})
