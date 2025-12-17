import json
import math
from sklearn.metrics import f1_score

input_file = "model_outputs.jsonl"

true_vals = []
pred_vals = []
text_true = []
text_pred = []
hallucinations = 0
total = 0

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def extract_numeric_range(x):
    """
    Convierte rangos tipo '5-6' en el valor promedio (5.5)
    Si no es rango, lo deja igual.
    """
    if isinstance(x, str) and "-" in x:
        a, b = x.split("-")
        return (float(a) + float(b)) / 2
    return x

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        item = json.loads(line)

        expected = item["answer_true"]
        pred = item["llm_value"]

        # Normalizar rangos
        expected_norm = extract_numeric_range(expected)
        pred_norm = extract_numeric_range(pred)

        # --- Numeric evaluation ---
        if is_number(expected_norm) and is_number(pred_norm):
            true_vals.append(float(expected_norm))
            pred_vals.append(float(pred_norm))

        # --- Text evaluation ---
        elif isinstance(expected, str) and isinstance(pred, str):
            text_true.append(expected.strip().lower())
            text_pred.append(pred.strip().lower())
        
        # --- Hallucination ---
        if pred is None or pred == "":
            hallucinations += 1

# Numeric Metrics
if true_vals:
    acc = sum(1 for t, p in zip(true_vals, pred_vals) if t == p) / len(true_vals)
    mae = sum(abs(t - p) for t, p in zip(true_vals, pred_vals)) / len(true_vals)
    rmse = math.sqrt(sum((t - p)**2 for t, p in zip(true_vals, pred_vals)) / len(true_vals))
else:
    acc = mae = rmse = None

# Text Metrics
if text_true:
    f1 = f1_score(text_true, text_pred, average='macro')
else:
    f1 = None

# Hallucination rate
hall_rate = hallucinations / total

results = {
    "numeric_accuracy": acc,
    "numeric_mae": mae,
    "numeric_rmse": rmse,
    "text_f1_macro": f1,
    "hallucination_rate": hall_rate,
    "total_items": total
}

print(json.dumps(results, indent=4, ensure_ascii=False))