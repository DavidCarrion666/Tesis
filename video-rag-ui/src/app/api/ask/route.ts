import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// Tipos esperados
interface AskBody {
  question: string;
  video_id: string; // UUID en string
}
interface GeminiSuccess {
  answer: string;
  tool_calls?: unknown;
  latency_ms?: number;
  tokens?: {
    prompt?: number | null;
    candidates?: number | null;
    total?: number | null;
  };
}
interface HeuristicSuccess {
  answer: string;
  tool_calls?: unknown;
}

// Validación mínima (sin libs)
function isAskBody(v: unknown): v is AskBody {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.question === "string" &&
    typeof o.video_id === "string" &&
    o.question.length > 0 &&
    o.video_id.length > 0
  );
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export async function POST(req: Request) {
  try {
    const raw: unknown = await req.json();
    if (!isAskBody(raw)) {
      return NextResponse.json({ error: "Body inválido" }, { status: 400 });
    }
    const { question, video_id } = raw;

    // 1) Intentar Gemini
    try {
      const data = await postJson<GeminiSuccess>(`${API_BASE}/llm-gemini/ask`, {
        question,
        video_id,
      });
      return NextResponse.json({
        provider: "gemini",
        answer: data.answer,
        tool_calls: data.tool_calls ?? [],
        latency_ms: data.latency_ms ?? null,
        tokens: data.tokens ?? null,
      });
    } catch {
      // fallback abajo
    }

    // 2) Fallback heurístico
    const data = await postJson<HeuristicSuccess>(`${API_BASE}/llm/ask`, {
      question,
      video_id,
      provider: "heuristic",
    });
    return NextResponse.json({
      provider: "heuristic",
      answer: data.answer,
      tool_calls: data.tool_calls ?? [],
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Error inesperado en /api/ask";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
