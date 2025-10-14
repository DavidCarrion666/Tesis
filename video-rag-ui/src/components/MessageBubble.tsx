interface MessageProps {
  role: "user" | "assistant";
  content: string;
}

export default function MessageBubble({ role, content }: MessageProps) {
  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`px-4 py-3 rounded-2xl shadow-sm text-sm max-w-[75%] leading-relaxed ${
          isUser
            ? "bg-[#1E3A8A] text-white rounded-br-none"
            : "bg-gray-100 text-gray-900 rounded-bl-none"
        }`}
      >
        {content}
      </div>
    </div>
  );
}
