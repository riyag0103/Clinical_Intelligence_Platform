"use client";

import Background from "./Background";
import HeroText from "./HeroText";
import AIEngine from "./AIEngine";

export default function Hero() {
  return (
    <section className="relative min-h-screen overflow-hidden bg-[#050B18]">
      {/* Background */}
      <Background />

      {/* Main Content */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1440px] items-center justify-between px-8 pt-20">

        {/* Left */}
        <div className="w-full lg:w-[45%]">
          <HeroText />
        </div>

        {/* Right */}
        <div className="hidden lg:flex lg:w-[55%] items-center justify-center">
          <AIEngine />
        </div>

      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
        <div className="flex flex-col items-center gap-2 text-slate-400">
          <span className="text-xs uppercase tracking-[0.3em]">
            Scroll
          </span>

          <div className="flex h-10 w-6 justify-center rounded-full border border-cyan-400/30">
            <div className="mt-2 h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          </div>
        </div>
      </div>
    </section>
  );
}