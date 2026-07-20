export default function Background() {
  return (
    <>
      {/* Base Background */}
      <div className="absolute inset-0 bg-[#050B18]" />

      {/* Main Glow */}
      <div className="absolute left-1/2 top-1/2 h-[700px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-500/10 blur-[180px]" />

      {/* Left Glow */}
      <div className="absolute left-20 top-32 h-[350px] w-[350px] rounded-full bg-blue-500/10 blur-[140px]" />

      {/* Right Glow */}
      <div className="absolute right-20 bottom-20 h-[400px] w-[400px] rounded-full bg-cyan-400/10 blur-[160px]" />

      {/* Top Glow */}
      <div className="absolute right-1/3 top-10 h-[250px] w-[250px] rounded-full bg-indigo-500/10 blur-[120px]" />
    </>
  );
}