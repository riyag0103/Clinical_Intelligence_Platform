"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface FloatingCardProps {
  title: string;
  subtitle: string;
  icon: LucideIcon;
  className?: string;
}

export default function FloatingCard({
  title,
  subtitle,
  icon: Icon,
  className = "",
}: FloatingCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{
        opacity: 1,
        y: [0, -8, 0],
      }}
      transition={{
        opacity: { duration: 0.5 },
        y: {
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
        },
      }}
      whileHover={{
        scale: 1.05,
        y: -6,
      }}
      className={`
        absolute
        rounded-2xl
        border
        border-white/10
        bg-white/5
        backdrop-blur-xl
        px-4
        py-3
        shadow-[0_0_40px_rgba(37,99,235,0.15)]
        transition-all
        duration-300
        hover:border-cyan-400/40
        hover:shadow-[0_0_50px_rgba(34,211,238,.35)]
        ${className}
      `}
    >
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-cyan-500/15 p-2">
          <Icon className="h-5 w-5 text-cyan-400" />
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <p className="text-xs text-slate-400">{subtitle}</p>
        </div>
      </div>
    </motion.div>
  );
}