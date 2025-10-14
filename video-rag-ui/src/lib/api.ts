export async function askVideoRag(
  question: string,
  videoId: string
): Promise<string> {
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, video_id: videoId }),
    });

    const data = await res.json();
    return data.answer ?? "No se obtuvo respuesta del modelo.";
  } catch (error) {
    console.error("Error al conectar con el backend:", error);
    return "Error de conexión con el servidor.";
  }
}
