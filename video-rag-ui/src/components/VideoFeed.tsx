// components/VideoFeed.tsx
"use client";
import { useState } from "react";

type VideoFeedProps = {
  onVideoIdChange?: (id: string) => void;
};

export default function VideoFeed({ onVideoIdChange }: VideoFeedProps) {
  const [streamImg, setStreamImg] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log("Archivo subido:", data);

      // 🔹 Notifica al padre el video_id
      if (data?.video_id) {
        onVideoIdChange?.(data.video_id);
      }

      // Inicia el streaming
      startStream(data.filename.replace(/\\/g, "/"), data.video_id);
    } catch (err) {
      alert("Error subiendo el video.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const startStream = (filePath: string, videoId: string) => {
    const socket = new WebSocket(
      `ws://127.0.0.1:8000/ws/stream?file=${encodeURIComponent(
        filePath
      )}&video_id=${videoId}`
    );

    socket.onmessage = (event) => {
      if (event.data === "<<EOF>>") {
        console.log("Stream terminado");
        socket.close();
        setWs(null);
        return;
      }
      if (event.data.startsWith("ERROR")) {
        alert(event.data);
        socket.close();
        return;
      }
      setStreamImg(`data:image/jpeg;base64,${event.data}`);
    };

    socket.onopen = () => console.log("WebSocket conectado ✅");
    socket.onclose = () => console.log("WebSocket cerrado ❌");

    setWs(socket);
  };

  const stopStream = () => {
    ws?.close();
    setWs(null);
    setStreamImg(null);
  };

  return (
    <section className="flex flex-col w-[40%] min-w-[380px] bg-white border-l border-gray-200 items-center justify-center p-6">
      <h2 className="text-lg font-semibold text-[#2563EB] mb-4">
        Detección en tiempo real con YOLO 🚗
      </h2>

      {loading ? (
        <p className="text-gray-500 animate-pulse">Subiendo video...</p>
      ) : streamImg ? (
        <img
          src={streamImg}
          alt="stream"
          className="rounded-xl shadow-md w-full max-w-md aspect-video object-cover border border-gray-200"
        />
      ) : (
        <div className="text-gray-400 border-2 border-dashed border-gray-300 rounded-xl w-full max-w-md h-64 flex items-center justify-center text-sm mb-4">
          Sube un video para ver las detecciones
        </div>
      )}

      <div className="flex gap-3 w-full max-w-md mt-4">
        <label className="flex-1 px-4 py-3 rounded-xl bg-gray-200 text-gray-800 font-semibold text-center cursor-pointer hover:bg-gray-300 transition-colors">
          Subir video
          <input
            type="file"
            accept="video/*"
            onChange={handleUpload}
            className="hidden"
          />
        </label>

        <button
          onClick={stopStream}
          className="flex-1 px-4 py-3 rounded-xl bg-red-500 text-white font-semibold hover:bg-red-600 transition-colors"
        >
          Detener
        </button>
      </div>
    </section>
  );
}
