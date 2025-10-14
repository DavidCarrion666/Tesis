"use client";
import { useState, FormEvent } from "react";
import MessageBubble from "./MessageBubble";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      const data = await res.json();
      const answer =
        data.answer || "No se recibió respuesta del modelo Gemini.";

      const response: Message = { role: "assistant", content: answer };
      setMessages((prev) => [...prev, response]);
    } catch (error) {
      console.error(error);
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
          Pregunta lo que quieras sobre el video cargado o en vivo.
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-4 bg-gray-50">
        {messages.map((m, i) => (
          <MessageBubble key={i} {...m} />
        ))}

        {loading && (
          <div className="text-center text-gray-400 italic mt-4 text-sm">
            Analizando video...
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
          placeholder="Escribe tu pregunta aquí..."
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
