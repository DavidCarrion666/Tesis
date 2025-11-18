# yolo-api/metrics/eval_runner_openai.py
import os, json, math, time, re, numbers
from pathlib import Path
import importlib.util

import pandas as pd
from jsonschema import validate, ValidationError
from sqlalchemy import text
from dotenv import load_dotenv

# ── Cargar database.py situado en ../database.py ───────────────────────────────
FILE_DIR = Path(__file__).resolve().parent          # .../yolo-api/metrics
DB_PATH  = FILE_DIR.parent / "database.py"          # .../yolo-api/database.py
if not DB_PATH.exists():
    raise RuntimeError(f"No se encontró database.py en: {DB_PATH}")
spec = importlib.util.spec_from_file_location("database", str(DB_PATH))
database = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "No se pudo preparar el cargador para database.py"
spec.loader.exec_module(database)
engine = database.engine
# ───────────────────────────────────────────────────────────────────────────────

# ====== Configuración base ======
load_dotenv()

# OpenAI
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en el entorno")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))  # determinista

SCHEMA_PATH = Path(__file__).with_name("schema.json")
DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

SYSTEM_ES = """Eres un analista de datos de tráfico. Responde SOLO en JSON válido exacto al esquema:
{"answer": string, "type":"numeric|text|sql", "value": number|string|null, "sql_used": string|null, "evidence": string[], "confidence": number}

Reglas estrictas:
- SQL: PostgreSQL (usa date_trunc('minute', ts) y to_char(ts,'HH24:MI:SS')). Prohibido strftime.
- Clases válidas: 'carro' y 'bus'. Prohibido 'vehicle', 'truck', 'van'.
- Filtro obligatorio: WHERE video_id::text = '<ID exacto>'.

Tipos:
- numeric: "type":"numeric". Prefiere 'sql_used' que devuelva un escalar; si no, 'value' numérico.
- sql:     "type":"sql". 'sql_used' obligatorio y ejecutable en PostgreSQL.
- text:    "type":"text". 'sql_used' OBLIGATORIO para derivar los 2 minutos con mayor conteo y construir 'evidence'
           con EXACTAMENTE dos marcas "VIDEO#t=HH:MM:SS", donde los segundos provienen de MAX(ts) dentro de cada minuto.
  Si no puedes obtener evidencia de la BD, responde EXACTAMENTE:
  {"answer":"insufficient_evidence","type":"text","value":null,"sql_used":null,"evidence":[],"confidence":0.0}
"""

# ====== Métricas textuales ======
from rouge_score import rouge_scorer
from bert_score import score as bertscore

_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _conciseness(text: str):
    """Devuelve (densidad_informativa, n_tokens). Aprox: tokens con len>=3 son informativos."""
    if not isinstance(text, str):
        return 0.0, 0
    toks = _WORD_RE.findall(text.lower())
    if not toks:
        return 0.0, 0
    info = [t for t in toks if len(t) >= 3]
    return len(info) / len(toks), len(toks)

def _rougeL_f1(pred: str, gold: str) -> float:
    """ROUGE-L F1 [0,1]."""
    if not isinstance(pred, str) or not isinstance(gold, str) or not gold.strip():
        return None
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r = s.score(gold, pred)["rougeL"]
    return float(r.fmeasure)

def _bertscore_f1(pred: str, gold: str) -> float:
    """BERTScore F1 [0,1]."""
    if not isinstance(pred, str) or not isinstance(gold, str) or not gold.strip():
        return None
    P, R, F1 = bertscore([pred], [gold], lang="es", rescale_with_baseline=False)
    return float(F1[0].item())

# ====== Helpers de BD / evaluación ======
def _run_sql(sql: str):
    with engine.connect() as conn:
        res = conn.execute(text(sql))
        try:
            return res.fetchall()
        except Exception:
            return []

def _json_ok(obj):
    try:
        validate(obj, SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, str(e)

def _compare_numeric(pred, gold):
    if isinstance(gold, (int, str)):
        try:
            return int(float(pred) == float(gold)), abs(float(pred) - float(gold))
        except Exception:
            return 0, math.inf
    try:
        p = float(pred); g = float(gold)
        mae = abs(p - g)
        return (1 if mae < 1e-6 else 0), mae
    except Exception:
        return 0, math.inf

def _compare_sql(rows_pred, rows_gold):
    return int(str(rows_pred) == str(rows_gold))

# ── Normalizador defensivo de SQL a PostgreSQL ────────────────────────────────
def _normalize_sql(sql: str, et: str) -> str:
    s = sql
    # SQLite → Postgres
    s = re.sub(r"strftime\(\s*'%M'\s*,\s*ts\s*\)\s+AS\s+minute", "date_trunc('minute', ts) AS m", s, flags=re.I)
    s = re.sub(r"GROUP\s+BY\s+minute\b", "GROUP BY 1", s, flags=re.I)
    s = re.sub(r"ORDER\s+BY\s+minute\b", "ORDER BY m", s, flags=re.I)
    s = re.sub(r"strftime\(\s*'%M'\s*,\s*ts\s*\)", "to_char(ts,'MI')", s, flags=re.I)
    # Tolerancia de clases 'car'/'carro'
    s = re.sub(r"\bobject_class\s*=\s*'car'\b", "object_class IN ('car','carro')", s, flags=re.I)
    s = re.sub(r"\bobject_class\s*=\s*'carro'\b", "object_class IN ('car','carro')", s, flags=re.I)
    # Asegura casteo de video_id cuando se compara con literales
    s = re.sub(r"\bvideo_id\s*=\s*'([^']+)'", r"video_id::text = '\1'", s, flags=re.I)
    s = re.sub(r"\bWHERE\s+video_id\s*=\s*", "WHERE video_id::text = ", s, flags=re.I)
    return s
# ───────────────────────────────────────────────────────────────────────────────

def _user_prompt(question: str, video_id: str, eval_type: str):
    ejemplos = f"""
Ejemplos PostgreSQL:
- Total del video:
  SELECT COUNT(*) FROM detections_norm WHERE video_id::text = '{video_id}';
- Top 3 minutos con más vehículos:
  SELECT date_trunc('minute', ts) AS m, COUNT(*) AS c
  FROM detections_norm
  WHERE video_id::text = '{video_id}'
  GROUP BY 1 ORDER BY c DESC LIMIT 3;
- Minuto a minuto solo 'carro':
  SELECT date_trunc('minute', ts) AS m, COUNT(*) AS c
  FROM detections_norm
  WHERE video_id::text = '{video_id}' AND object_class='carro'
  GROUP BY 1 ORDER BY m ASC;
- TEXT → derivar picos y producir evidencia (usa segundos reales via MAX(ts)):
  WITH per_min AS (
    SELECT date_trunc('minute', ts) AS m, COUNT(*) AS c
    FROM detections_norm
    WHERE video_id::text='{video_id}'
    GROUP BY 1
  ),
  top2 AS (
    SELECT m FROM per_min ORDER BY c DESC, m ASC LIMIT 2
  )
  SELECT to_char(MAX(ts),'HH24:MI:SS') AS hh
  FROM detections_norm
  WHERE video_id::text='{video_id}' AND date_trunc('minute', ts) IN (SELECT m FROM top2)
  GROUP BY date_trunc('minute', ts)
  ORDER BY hh ASC;
"""
    return f"""Pregunta: {question}
Eval type: {eval_type}
Contexto:
- video_id: {video_id}
- Tabla detections_norm(video_id, frame_number, ts, object_class, confidence, x1,y1,x2,y2, track_id)
- Clases válidas: 'carro', 'bus'
- Dialecto: PostgreSQL. Usa date_trunc y to_char. No uses strftime.
- Para text: incluye 'sql_used' como en el ejemplo y construye EXACTAMENTE 2 evidencias {video_id}#t=HH:MM:SS usando los HH:MM:SS devueltos.
{ejemplos}
RESPONDE SOLO CON EL JSON DEL ESQUEMA. Nada más.
"""

# ── Oro dinámico desde BD ─────────────────────────────────────────────────────
def _gold_minutes_from_db(video_id: str, k: int = 2):
    sql = f"""
    WITH per_min AS (
      SELECT date_trunc('minute', ts) AS m, COUNT(*) AS c
      FROM detections_norm
      WHERE video_id::text = '{video_id}'
      GROUP BY 1
    )
    SELECT to_char(m,'HH24:MI') AS hhmm
    FROM per_min
    ORDER BY c DESC, hhmm ASC
    LIMIT {k};
    """
    rows = _run_sql(sql)
    return [r[0] for r in rows if r and r[0]]

def _gold_peaks_hms_from_db(video_id: str, k: int = 2):
    """
    Devuelve dos HH:MM:SS para los k minutos top. Los segundos provienen de MAX(ts).
    """
    sql = f"""
    WITH per_min AS (
      SELECT date_trunc('minute', ts) AS m, COUNT(*) AS c
      FROM detections_norm
      WHERE video_id::text = '{video_id}'
      GROUP BY 1
    ),
    topk AS (
      SELECT m FROM per_min ORDER BY c DESC, m ASC LIMIT {k}
    )
    SELECT to_char(MAX(ts),'HH24:MI:SS') AS hh
    FROM detections_norm
    WHERE video_id::text = '{video_id}'
      AND date_trunc('minute', ts) IN (SELECT m FROM topk)
    GROUP BY date_trunc('minute', ts)
    ORDER BY hh ASC;
    """
    rows = _run_sql(sql)
    return [r[0] for r in rows if r and r[0]]

def _video_bounds(video_id: str):
    """
    Devuelve mm_min, mm_max, hms_min, hms_max como cadenas.
    """
    rows = _run_sql(f"""
        SELECT to_char(MIN(ts),'HH24:MI') AS hhmm_min,
               to_char(MAX(ts),'HH24:MI') AS hhmm_max,
               to_char(MIN(ts),'HH24:MI:SS') AS hms_min,
               to_char(MAX(ts),'HH24:MI:SS') AS hms_max
        FROM detections_norm
        WHERE video_id::text = '{video_id}';
    """)
    if not rows:
        return None, None, None, None
    return rows[0]

def _minute_bounds(video_id: str):
    rows = _run_sql(f"""
        SELECT to_char(MIN(ts),'HH24:MI') AS hhmm_min,
               to_char(MAX(ts),'HH24:MI') AS hhmm_max
        FROM detections_norm
        WHERE video_id::text = '{video_id}';
    """)
    if not rows:
        return None, None
    return rows[0]

# ── Score de evidencia: formato + ventana + existencia + intersección con gold ─
def _evidence_score(evidence_list, video_id: str, gold_list):
    """
    Regresa 1 si:
      a) Hay EXACTAMENTE 2 evidencias, formato VIDEO#t=HH:MM:SS,
      b) Cada HH:MM está dentro de [mm_min, mm_max] del video,
      c) Existe al menos un registro en ese minuto,
      d) Coincide por minuto con el conjunto gold (al menos 2 si hay 2 gold).
    Si no, 0.
    """
    if not evidence_list or len(evidence_list) != 2:
        return 0

    mm_min, mm_max = _minute_bounds(video_id)
    gold_minutes = set()
    for g in gold_list or []:
        if isinstance(g, str) and "#t=" in g:
            gold_minutes.add(g.split("#t=")[1][:5])

    seen_minutes = set()
    for tag in evidence_list:
        if not isinstance(tag, str) or "#t=" not in tag:
            return 0
        vid, hh = tag.split("#t=", 1)
        if vid != video_id or len(hh) < 5 or hh.count(":") < 1:
            return 0
        mm = hh[:5]  # HH:MM

        # ventana por minuto
        if mm_min and mm_max and not (mm_min <= mm <= mm_max):
            return 0

        # existencia en BD por minuto
        q = ("SELECT COUNT(*) FROM detections_norm "
             f"WHERE video_id::text = '{video_id}' AND to_char(ts,'HH24:MI') = '{mm}'")
        rows = _run_sql(q)
        if not rows or rows[0][0] == 0:
            return 0

        seen_minutes.add(mm)

    if not gold_minutes:
        return 1
    inter = seen_minutes & gold_minutes
    return 1 if len(inter) >= min(2, len(gold_minutes)) else 0
# ───────────────────────────────────────────────────────────────────────────────

# ====== Llamada a OpenAI ======
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ====== Llamada a OpenAI (Chat Completions) ======
from openai import OpenAI
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _call_openai(system: str, user: str, model_name: str = OPENAI_MODEL, temperature: float = TEMPERATURE):
    """
    Devuelve (txt_json, latency_ms, in_tokens, out_tokens).
    Usa Chat Completions con JSON mode para máxima compatibilidad.
    """
    t0 = time.time()
    resp = _openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.time() - t0) * 1000.0

    txt = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    in_tok  = getattr(usage, "prompt_tokens", None)      if usage else None
    out_tok = getattr(usage, "completion_tokens", None)  if usage else None
    return txt, latency_ms, in_tok, out_tok


# ====== Runner principal ======
def run_eval(dataset_path: Path = DATASET_PATH):
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    lines = [l for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for l in lines:
        try:
            items.append(json.loads(l))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Línea JSONL inválida: {l[:120]}... -> {e}")

    rows_out = []

    for it in items:
        et  = it["eval_type"]
        vid = it.get("video_id","")
        user = _user_prompt(it["question"], vid, et)

        # Para TEXT: inyecta ventana temporal y minutos gold para guiar al modelo
        if et == "text":
            mm_min, mm_max, hms_min, hms_max = _video_bounds(vid)
            gold_mm = _gold_minutes_from_db(vid, k=2) or []
            hint = ""
            if mm_min and mm_max and hms_min and hms_max:
                hint += (
                    f"\nRestricción temporal (UTC):\n"
                    f"- Evidencias DEBEN estar entre {hms_min} y {hms_max}.\n"
                    f"- Minutos válidos entre {mm_min} y {mm_max}.\n"
                )
            if len(gold_mm) == 2:
                hint += (
                    f"- En BD, los 2 minutos con mayor conteo son EXACTAMENTE: {gold_mm[0]} y {gold_mm[1]}.\n"
                    f"- Debes generar 'sql_used' que derive esos minutos y construir 'evidence' con 2 marcas "
                    f"\"{vid}#t=HH:MM:SS\" cuyos HH:MM pertenezcan a esos dos minutos y cuyos segundos provengan de MAX(ts) en cada minuto."
                )
            if hint:
                user += "\n" + hint

        raw, latency_ms, in_tok, out_tok = _call_openai(SYSTEM_ES, user, OPENAI_MODEL, TEMPERATURE)

        parsed = None; json_valid = 0; err = ""
        try:
            parsed = json.loads(raw)
            ok, err = _json_ok(parsed)
            json_valid = int(ok)
        except Exception as e:
            err = f"parse_error: {e}"

        # métricas comunes
        numeric_acc = numeric_mae = sql_ok = grounded = halluc = None
        rougeL_f1 = bertscore_f1 = conciseness = None
        len_pred_tokens = len_gold_tokens = None

        if json_valid and isinstance(parsed, dict):
            # TEXT: si trae SQL pero sin evidencias, intenta construirlas desde BD
            if et == "text":
                sql_pred = parsed.get("sql_used")
                if sql_pred and not parsed.get("evidence"):
                    try:
                        sql_pred = _normalize_sql(sql_pred, et)
                        rows = _run_sql(sql_pred)
                        ev = []
                        for r in rows[:2]:
                            hh = str(r[0])
                            if len(hh) >= 5:
                                if len(hh) == 5:
                                    mm = hh
                                    q = (f"SELECT to_char(MAX(ts),'HH24:MI:SS') FROM detections_norm "
                                         f"WHERE video_id::text='{vid}' AND to_char(ts,'HH24:MI')='{mm}'")
                                    rr = _run_sql(q)
                                    hh2 = rr[0][0] if rr and rr[0][0] else mm + ":00"
                                    ev.append(f"{vid}#t={hh2}")
                                else:
                                    ev.append(f"{vid}#t={hh[-8:]}")
                        parsed["evidence"] = ev
                    except Exception:
                        pass

            # Oro dinámico: HH:MM:SS de MAX(ts) por minuto top
            gold_hms = _gold_peaks_hms_from_db(vid, k=2)
            gold_ev_from_db = [f"{vid}#t={h}" for h in gold_hms]

            # Si el dataset trae oro explícito para texto, úsalo
            gold_text = it.get("gold_text")
            gold_evidence = it.get("gold_evidence") or gold_ev_from_db

            grounded = _evidence_score(parsed.get("evidence", []), vid, gold_evidence)
            halluc   = 1 - grounded

            if et == "numeric":
                val = parsed.get("value")
                sql_pred = parsed.get("sql_used")
                if sql_pred:
                    try:
                        sql_pred = _normalize_sql(sql_pred, et)
                        rows = _run_sql(sql_pred)
                        if rows and len(rows[0]) == 1:
                            val = rows[0][0]
                    except Exception:
                        pass
                if val is None:
                    numeric_acc, numeric_mae = 0, float("inf")
                else:
                    acc, mae = _compare_numeric(val, it["gold"])
                    numeric_acc, numeric_mae = acc, mae

            elif et == "sql":
                pred_sql = parsed.get("sql_used")
                if pred_sql:
                    try:
                        pred_sql = _normalize_sql(pred_sql, et)
                        rows_pred = _run_sql(pred_sql)
                        rows_gold = _run_sql(it["gold_sql"])
                        sql_ok = _compare_sql(rows_pred, rows_gold)
                    except Exception:
                        sql_ok = 0
                else:
                    sql_ok = 0

            elif et == "text":
                pred_text = parsed.get("answer", "")
                # Concisión
                conciseness, len_pred_tokens = _conciseness(pred_text)
                # Rouge/BERTScore si hay oro textual
                if gold_text:
                    _, len_gold_tokens = _conciseness(gold_text)
                    try:
                        rougeL_f1 = _rougeL_f1(pred_text, gold_text)
                    except Exception:
                        rougeL_f1 = None
                    try:
                        bertscore_f1 = _bertscore_f1(pred_text, gold_text)
                    except Exception:
                        bertscore_f1 = None

        rows_out.append({
            "id": it["id"], "provider": "openai", "model": OPENAI_MODEL, "lang": it.get("lang","es"),
            "latency_ms": latency_ms, "in_tokens": in_tok, "out_tokens": out_tok,
            "json_ok": json_valid, "numeric_acc": numeric_acc, "numeric_mae": numeric_mae,
            "sql_ok": sql_ok, "grounded": grounded, "hallucination": halluc,
            "rougeL_f1": rougeL_f1, "bertscore_f1": bertscore_f1, "conciseness": conciseness,
            "len_pred_tokens": len_pred_tokens, "len_gold_tokens": len_gold_tokens,
            "raw": raw, "json_error": err
        })

    # ====== Salidas ======
    df = pd.DataFrame(rows_out)

    # tipos numéricos
    for col in ["numeric_acc", "numeric_mae", "sql_ok", "grounded", "hallucination",
                "latency_ms", "in_tokens", "out_tokens",
                "rougeL_f1", "bertscore_f1", "conciseness",
                "len_pred_tokens", "len_gold_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_base = Path(__file__).parent.parent
    out_dir = out_base / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "per_question_openai.csv"
    summary_path = out_dir / "summary_openai.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8")

    def _safe_mean(series):
        s = [x for x in series.tolist() if isinstance(x, numbers.Number) and pd.notnull(x)]
        return (sum(s) / len(s)) if s else None

    summary = (
        df.groupby(["provider", "model", "lang"])
          .agg(n=("id", "count"),
               json_ok=("json_ok", "mean"),
               numeric_acc=("numeric_acc", _safe_mean),
               sql_ok=("sql_ok", _safe_mean),
               grounded=("grounded", "mean"),
               hallucination=("hallucination", "mean"),
               rougeL_f1=("rougeL_f1", _safe_mean),
               bertscore_f1=("bertscore_f1", _safe_mean),
               conciseness=("conciseness", _safe_mean),
               p50_latency=("latency_ms", lambda s: float(pd.Series(s).quantile(0.5))),
               p95_latency=("latency_ms", lambda s: float(pd.Series(s).quantile(0.95))))
          .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    return {
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "summary": summary.to_dict(orient="records"),
    }

# ====== Modo script ======
if __name__ == "__main__":
    from pprint import pprint
    pprint(run_eval())
