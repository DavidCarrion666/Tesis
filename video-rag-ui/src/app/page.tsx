"use client";

import { useState } from "react";
import Navbar from "../components/Navbar";
import ChatPanel from "../components/ChatPanel";
import VideoFeed from "../components/VideoFeed";

export default function Page() {
  const [videoId, setVideoId] = useState<string>("");

  return (
    <main className="flex flex-col h-screen bg-[#F5F7FB]">
      <Navbar />
      <div className="flex flex-1">
        {/* Izquierda: Chat */}
        <ChatPanel videoId={videoId} />

        {/* Derecha: Video. Cuando subas/selecciones un video, llama a setVideoId */}
        <VideoFeed onVideoIdChange={(id: string) => setVideoId(id)} />
      </div>
    </main>
  );
}
