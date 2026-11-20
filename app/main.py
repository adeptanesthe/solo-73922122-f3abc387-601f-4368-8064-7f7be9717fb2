"""FastAPI 入口：上传参考/待检 WAV，返回冲次核对结果。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioValidationError, MAX_DURATION_SEC, SAMPLE_RATE
from .decay import DecayValidationError, TimeWindow
from .service import MAX_UPLOAD_BYTES, Store

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/workspace/data"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="冲压机冲次声音核对", version="1.1.0")
store = Store(DATA_DIR)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 对比请求缺字段/类型错误同样属于「条件不足」：撤下旧对比并给出中文原因
    if request.url.path == "/api/compare":
        store.clear_comparison()
        fields: list[str] = []
        for err in exc.errors():
            loc = [str(part) for part in err.get("loc", []) if part != "body"]
            if loc:
                fields.append(loc[-1])
        hint = "、".join(dict.fromkeys(fields)) or "请求参数"
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"对比参数不完整或格式错误（{hint}），"
                "请填写待测频带与四个时段的起止秒数后重试"
            },
        )
    return JSONResponse(
        status_code=422, content={"detail": jsonable_encoder(exc.errors())}
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/uploads/{name}")
def get_upload(name: str) -> FileResponse:
    # 动态读取 store 的上传目录，更换 DATA_DIR 或测试夹具后仍可回传录音
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = store.uploads_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/result/last")
def last_result() -> dict:
    result = store.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效检查结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.get("/api/comparison/last")
def last_comparison() -> dict:
    result = store.get_last_comparison()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效衰减对比结果")
    return result


@app.delete("/api/comparison/last")
def clear_comparison() -> dict[str, str]:
    store.clear_comparison()
    return {"status": "cleared"}


async def _read_wav(file: UploadFile, label: str) -> bytes:
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail=f"{label}必须是 .wav 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label}文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )
    return raw


@app.post("/api/compare")
async def compare(
    before: UploadFile = File(..., description="维修前 WAV"),
    after: UploadFile = File(..., description="维修后 WAV"),
    band_low_hz: float = Form(..., description="待测频带低端 (Hz)"),
    band_high_hz: float = Form(..., description="待测频带高端 (Hz)"),
    before_impact_start: float = Form(...),
    before_impact_end: float = Form(...),
    before_background_start: float = Form(...),
    before_background_end: float = Form(...),
    after_impact_start: float = Form(...),
    after_impact_end: float = Form(...),
    after_background_start: float = Form(...),
    after_background_end: float = Form(...),
) -> dict:
    # 条件不足时撤下旧对比，避免页面继续展示已失效的结论
    store.clear_comparison()
    before_raw = await _read_wav(before, "维修前录音")
    after_raw = await _read_wav(after, "维修后录音")

    windows = {
        "维修前冲击时段": (before_impact_start, before_impact_end),
        "维修前背景时段": (before_background_start, before_background_end),
        "维修后冲击时段": (after_impact_start, after_impact_end),
        "维修后背景时段": (after_background_start, after_background_end),
    }
    try:
        return store.compare(
            before_raw,
            after_raw,
            band_low_hz,
            band_high_hz,
            TimeWindow(*windows["维修前冲击时段"]),
            TimeWindow(*windows["维修前背景时段"]),
            TimeWindow(*windows["维修后冲击时段"]),
            TimeWindow(*windows["维修后背景时段"]),
        )
    except (AudioValidationError, DecayValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inspect")
async def inspect(
    reference: UploadFile = File(..., description="正常循环参考 WAV"),
    test: UploadFile = File(..., description="待检循环 WAV"),
) -> dict:
    ref_raw = await _read_wav(reference, "参考录音")
    test_raw = await _read_wav(test, "待检录音")

    try:
        # 任何 AudioValidationError 都会在 store 写入前抛出，
        # 因此无效音频不会覆盖上一次有效结果
        return store.inspect(ref_raw, test_raw)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
