import Background from "./Background";
import Navbar from "./Navbar";
import HeroText from "./HeroText";
import AIEngine from "./AIEngine";

export default function Hero() {
  return (
    <section className="relative min-h-screen overflow-hidden">
      <Background />

      <div className="relative z-10">
        <Navbar />

        <div className="mx-auto flex min-h-[calc(100vh-80px)] max-w-7xl items-center justify-between gap-16 px-8">
          <HeroText />
          <AIEngine />
        </div>
      </div>
    </section>
  );
}