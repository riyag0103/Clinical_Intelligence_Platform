import Hero from "@/components/landing/Hero";
import TrustedTechnologies from "@/components/landing/TrustedTechnologies";

export default function Home() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-[#040816] text-white">
      <Hero />
      <TrustedTechnologies />
    </main>
  );
}