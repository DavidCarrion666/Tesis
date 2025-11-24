import json
import math
from statistics import mean
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util

# Modelo para BERTScore / embeddings
embedder = SentenceTransformer("all-mpnet-base-v2")

# --------------------------
# Cargar dataset
# --------------------------
dataset = []
with open("eval_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))


# --------------------------
# Cargar respuestas del modelo
# --------------------------
outputs = {}
with open("model_outputs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        outputs[d["id"]] = d["answer_pred"]


# --------------------------
# Métricas numéricas
# --------------------------
numeric_mae = []
numeric_rmse = []
numeric_exact = 0
numeric_total = 0
range_overlap = []

def parse_range(r):
    lo, hi = r.split("-")
    return float(lo), float(hi)

for item in dataset:
    qid = item["id"]
    if item["type"] not in ["numeric", "range"]:
        continue

    true_ans = item["answer_true"]
    pred = outputs.get(qid, None)
    numeric_total += 1

    if pred is None:
        continue

    # EXACT
    if item["type"] == "numeric":
        true_val = float(true_ans)
        pred_val = float(pred)
        if true_val == pred_val:
            numeric_exact += 1
        numeric_mae.append(abs(true_val - pred_val))
        numeric_rmse.append((true_val - pred_val) ** 2)

    # RANGE
    elif item["type"] == "range":
        lo, hi = parse_range(true_ans)
        pred_val = float(pred)

        # overlap score
        if lo <= pred_val <= hi:
            range_overlap.append(1)
        else:
            range_overlap.append(0)


# --------------------------
# Métricas textuales
# --------------------------
textual_f1 = []
textual_rouge = []
bert_scores = []
hallucination_count = 0
text_total = 0

def token_f1(a, b):
    a_tok = a.lower().split()
    b_tok = b.lower().split()
    inter = len(set(a_tok) & set(b_tok))
    if inter == 0:
        return 0.0
    precision = inter / len(b_tok)
    recall = inter / len(a_tok)
    return 2 * precision * recall / (precision + recall)

for item in dataset:
    qid = item["id"]
    if item["type"] != "text":
        continue

    truth = item["answer_true"]
    pred = outputs.get(qid, "")

    text_total += 1

    # F1 textual
    textual_f1.append(token_f1(truth, pred))

    # BERTScore
    emb_true = embedder.encode(truth, convert_to_tensor=True)
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_true, emb_pred))
    bert_scores.append(sim)

    # Hallucination rule:
    # Si menciona algo que no está en la verdad → posible alucinación
    if any(word not in truth.lower() for word in pred.lower().split()):
        hallucination_count += 1


# --------------------------
# Reporte final
# --------------------------
print("\n================ MÉTRICAS NUMÉRICAS ================")
print("Exact Match:", numeric_exact / numeric_total)
print("MAE:", mean(numeric_mae))
print("RMSE:", math.sqrt(mean(numeric_rmse)))
print("Range Overlap:", mean(range_overlap) if range_overlap else 0)

print("\n================ MÉTRICAS TEXTUALES ================")
print("F1 textual promedio:", mean(textual_f1))
print("BERTScore promedio:", mean(bert_scores))

print("\n================ HALLUCINATION ================")
print("Hallucination Rate:", hallucination_count / text_total)
