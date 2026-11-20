# 冲压机冲次声音核对

值班员上传**正常循环参考录音**与**待检循环录音**，系统从音频中提取冲击并按参考节奏匹配，自动标出**漏冲对应的参考冲次**与**多冲的实录时刻**，网页可定位播放标出的时段。

## 输入约定

- WAV（PCM）：16 kHz、16-bit、单声道，单个文件不超过 8 秒
- 参考录音：含 4～8 次清晰冲击，相邻冲击间隔至少 150 ms
- 待检录音：允许起始静音、整体响度变化、漏冲、多冲；正常冲次间隔相对参考允许 ±15% 变化

## 判定规则

1. **冲击检测**：短时能量包络 + 自适应门限（门限跟随录音自身峰值，对整体响度变化不敏感）+ 局部极大值拾取。
2. **回声处理**：同一冲击 40 ms 不应期内的峰（含衰减拖尾、回声）只保留最强一个，回声不计多冲。
3. **节奏匹配（动态规划）**：以参考冲次序号为基准做最优对齐，成本 = 漏冲数 + 多冲数；相邻匹配的累计时长须落在参考节奏 ±15% 窗口内。
4. **漏冲不错位**：漏冲只计一次成本，后续冲次仍按参考序号继续对齐，不会整体错位。
5. **歧义人工复听**：若多种归属具有相同的最优成本（同样满足节奏与最少漏/多冲），返回 `ambiguous`，网页给出醒目提示，不武断放行。
6. **无效音频不覆盖结果**：格式/时长/参考质量不合格时返回 400 且不写入状态，上一次有效结果原样保留。

## 启动（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

录音与结果数据保存在命名卷 `press-data`（容器内 `/data`）。

## 本地开发（无 Docker 时）

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 自定义数据目录：APP_DATA_DIR=/some/dir uvicorn app.main:app
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/inspect` | multipart 表单字段 `reference`、`test`（均为 .wav），返回核对结果 |
| `GET` | `/api/result/last` | 最近一次有效结果（无结果时 404） |
| `DELETE` | `/api/result/last` | 清除内存中的最近结果 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/data/uploads/{name}` | 回传已上传录音，供网页播放器定位播放 |

结果关键字段：`verdict`（`ok` / `defect` / `ambiguous`）、`missed[].ref_index`（漏冲对应的参考冲次，从 0 开始）、`missed[].expected_time`（待检中预计应出现的时刻）、`extras[].time`（多充实录时刻）、`matches`（匹配明细）、`warnings`（回声抑制、歧义等提示）。

## 测试

```bash
python3 -m pytest tests/ -q
```

合成 WAV 生成器见 `scripts/gen_demo.py`，可用于手工演示：

```bash
python3 scripts/gen_demo.py /tmp/demo
```

## 目录结构

```
app/
  audio.py       WAV 校验与冲击检测
  matching.py    节奏匹配动态规划（漏冲/多冲/歧义）
  service.py     检查服务与结果存储（无效输入不覆盖旧结果）
  main.py        FastAPI 路由
  static/        单页前端（上传、波形标注、点击定位播放）
tests/           检测、匹配、端到端 API 测试（28 项）
scripts/         演示 WAV 生成
```
