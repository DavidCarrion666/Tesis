export default function Navbar() {
  return (
    <nav className="w-full bg-white border-b border-gray-200 shadow-sm px-8 py-4 flex justify-between items-center">
      <h1 className="text-2xl font-semibold text-[#1E3A8A] tracking-tight">
        VideoRAG
      </h1>
      <div className="text-gray-600 text-sm">
        Análisis inteligente de video en tiempo real
      </div>
    </nav>
  );
}
