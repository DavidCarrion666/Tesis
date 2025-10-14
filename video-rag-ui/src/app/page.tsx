import Navbar from "../components/Navbar";
import ChatPanel from "../components/ChatPanel";
import VideoFeed from "../components/VideoFeed";

export default function Page() {
  return (
    <main className="flex flex-col h-screen bg-[#F5F7FB]">
      <Navbar />
      <div className="flex flex-1">
        <ChatPanel />
        <VideoFeed />
      </div>
    </main>
  );
}
