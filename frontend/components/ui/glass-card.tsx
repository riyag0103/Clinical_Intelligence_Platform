import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

export default function GlassCard({
  children,
  className,
}: GlassCardProps) {
  return (
    <div
      className={cn(
        "rounded-3xl border border-white/10 bg-white/[0.04] backdrop-blur-2xl shadow-[0_0_80px_rgba(34,211,238,.08)]",
        className
      )}
    >
      {children}
    </div>
  );
}