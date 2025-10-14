"use client";
import { useRef, useState } from "react";

export default function VideoFeed() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [streaming, setStreaming] = useState<boolean>(false);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setVideoUrl(URL.createObjectURL(file));
  };

  const handleLive = async () => {
    if (streaming) {
      const stream = videoRef.current?.srcObject as MediaStream;
      stream?.getTracks().forEach((track) => track.stop());
      videoRef.current!.srcObject = null;
      setStreaming(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreaming(true);
      }
    } catch {
      alert("No se pudo acceder a la cámara.");
    }
  };

  return (
    <section className="flex flex-col w-[40%] min-w-[380px] bg-white border-l border-gray-200">
      <header className="p-4 border-b bg-gray-50">
        <h2 className="text-lg font-semibold text-[#2563EB]">Cámara / Video</h2>
        <p className="text-sm text-gray-500">
          Puedes subir un video o activar la cámara en vivo.
        </p>
      </header>

      <div className="flex flex-col flex-1 items-center justify-center p-6">
        {videoUrl || streaming ? (
          <video
            ref={videoRef}
            src={videoUrl || undefined}
            autoPlay={streaming}
            controls={!streaming}
            className="rounded-xl shadow-md w-full max-w-md aspect-video object-cover border border-gray-200"
          />
        ) : (
          <div className="text-gray-400 border-2 border-dashed border-gray-300 rounded-xl w-full max-w-md h-64 flex items-center justify-center text-sm">
            No hay video cargado
          </div>
        )}

        <div className="flex gap-3 w-full max-w-md mt-6">
          <button
            onClick={handleLive}
            className={`flex-1 px-4 py-3 rounded-xl text-white font-semibold transition-colors ${
              streaming
                ? "bg-red-500 hover:bg-red-600"
                : "bg-[#2563EB] hover:bg-[#1E40AF]"
            }`}
          >
            {streaming ? "Detener cámara" : "Cámara en vivo"}
          </button>

          <label className="flex-1 px-4 py-3 rounded-xl bg-gray-200 text-gray-800 font-semibold text-center cursor-pointer hover:bg-gray-300 transition-colors">
            Subir video
            <input
              type="file"
              accept="video/*"
              onChange={handleUpload}
              className="hidden"
            />
          </label>
        </div>
      </div>
    </section>
  );
}
