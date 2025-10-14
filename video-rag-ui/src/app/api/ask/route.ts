import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { question } = await req.json();

    if (!question) {
      return NextResponse.json(
        { error: "Falta la pregunta." },
        { status: 400 }
      );
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.error("❌ No se encontró GEMINI_API_KEY");
      return NextResponse.json(
        { error: "API Key no configurada." },
        { status: 500 }
      );
    }

    // ✅ Conexión directa a la API REST v1 (ya no usa SDK)
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: question }] }],
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      console.error("❌ Error de la API:", data);
      return NextResponse.json(
        { error: data.error?.message || "Error desde la API de Gemini." },
        { status: response.status }
      );
    }

    const text =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No se recibió respuesta del modelo.";

    return NextResponse.json({ answer: text });
  } catch (err: unknown) {
    if (err instanceof Error) {
      console.error("💥 Error interno:", err.message);
      return NextResponse.json({ error: err.message }, { status: 500 });
    }

    console.error("💥 Error desconocido:", err);
    return NextResponse.json(
      { error: "Error interno del servidor." },
      { status: 500 }
    );
  }
}

// 👇 Esto es opcional, para evitar el error 405 si entras desde el navegador
export async function GET() {
  return NextResponse.json({
    message: "Usa POST para enviar preguntas a /api/ask",
    example: { question: "hola" },
  });
}
