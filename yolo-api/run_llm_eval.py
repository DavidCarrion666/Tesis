import json
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000/multimodal/video-query"
VIDEO_ID = "d8ce9992-6feb-4926-a7fa-fcff39cc160d"   # <- pon aquí tu actual video_id

input_file = "eval_dataset.jsonl"
output_file = "model_outputs.jsonl"

def call_llm(question):
    payload = {
        "video_id": VIDEO_ID,
        "question": question,
        "step": 30
    }
    r = requests.post(BACKEND_URL, json=payload)
    return r.json()

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        q = item["question"]

        print("Consultando:", q)
        resp = call_llm(q)
        llm_answer = resp.get("llm_response", {})

        out = {
            "question": q,
            "answer_true": item["answer_true"],
            "llm_answer": llm_answer
        }

        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

        time.sleep(1)   # para evitar rate limiting