"use client";
import { useState, FormEvent } from "react";
import MessageBubble from "./MessageBubble";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export default function ChatPanel({ videoId }: { videoId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    if (!videoId) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Primero sube un video (no hay video_id).",
        },
      ]);
      return;
    }

    const question = input;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/multimodal/video-query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    video_id: videoId,
    question: question,
    step: 30, // frames representativos
  }),
});
;
const data = await res.json();

const answer =
  data?.llm_response?.answer ??
  JSON.stringify(data.llm_response, null, 2) ??
  "Sin respuesta.";

setMessages((prev) => [
  ...prev,
  { role: "assistant", content: answer },
]);

    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error al conectar con el servidor." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="flex flex-col flex-1 bg-white h-full border-r border-gray-200">
      <header className="p-4 border-b bg-gray-50">
        <h2 className="text-lg font-semibold text-[#1E3A8A]">
          Chat de análisis
        </h2>
        <p className="text-sm text-gray-500">
          Pregunta usando el video activo. ID: {videoId || "—"}
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 bg-gray-50">
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} />
        ))}
        {loading && (
          <div className="text-center text-gray-400 italic mt-4 text-sm">
            Analizando…
          </div>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="p-4 bg-white border-t border-gray-200 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Escribe tu pregunta…"
          className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-[#1E3A8A] focus:outline-none text-sm text-black"
        />

        <button
          disabled={loading}
          type="submit"
          className={`px-6 py-3 rounded-xl font-semibold transition-colors text-white ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-[#1E3A8A] hover:bg-[#172554]"
          }`}
        >
          Enviar
        </button>
      </form>
    </section>
  );
}
