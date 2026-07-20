import Background from "./Background";
import HeroText from "./HeroText";
import AIEngine from "./AIEngine";

export default function Hero() {
  return (
    <section className="relative min-h-screen overflow-hidden bg-slate-950">
      {/* Background Effects */}
      <Background />

      {/* Hero Content */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1440px] items-start justify-between px-6 pt-36">

        {/* Left Section */}
        <div className="w-full lg:w-1/2">
          <HeroText />
        </div>

        {/* Right Section */}
        <div className="hidden lg:flex w-1/2 justify-end">
          <AIEngine />
        </div>

      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs uppercase tracking-[0.25em] text-slate-400">
            Scroll
          </span>

          <div className="h-10 w-6 rounded-full border border-cyan-400/30 flex justify-center">
            <div className="mt-2 h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          </div>
        </div>
      </div>
    </section>
  );
}