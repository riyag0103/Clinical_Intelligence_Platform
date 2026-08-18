export default function Background() {
  return (
    <div className="absolute inset-0 overflow-hidden">

      {/* Main Background */}
      <div className="absolute inset-0 bg-[#040816]" />

      {/* Center Glow */}
      <div
        className="
          absolute
          left-1/2
          top-[-250px]
          h-[700px]
          w-[700px]
          -translate-x-1/2
          rounded-full
          bg-cyan-500/15
          blur-[180px]
        "
      />

      {/* Left Glow */}
      <div
        className="
          absolute
          left-[-200px]
          top-[250px]
          h-[500px]
          w-[500px]
          rounded-full
          bg-sky-500/10
          blur-[170px]
        "
      />

      {/* Right Glow */}
      <div
        className="
          absolute
          right-[-200px]
          bottom-[120px]
          h-[500px]
          w-[500px]
          rounded-full
          bg-violet-500/10
          blur-[170px]
        "
      />

      {/* Grid */}
      <div
        className="
          absolute
          inset-0
          opacity-[0.04]
          [background-image:linear-gradient(rgba(255,255,255,.1)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.1)_1px,transparent_1px)]
          [background-size:60px_60px]
        "
      />
    </div>
  );
}