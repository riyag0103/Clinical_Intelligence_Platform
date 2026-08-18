"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface OutputCardProps {
  title: string;
  subtitle: string;
  icon: LucideIcon;
  delay?: number;
}

export default function OutputCard({
  title,
  subtitle,
  icon: Icon,
  delay = 0,
}: OutputCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{
        delay,
        duration: 0.6,
      }}
      whileHover={{
        scale: 1.03,
        x: -8,
      }}
      className="
        group
        relative
        overflow-hidden
        rounded-2xl
        border
        border-white/10
        bg-white/5
        backdrop-blur-xl
        p-5
        transition-all
        hover:border-cyan-400/40
        hover:bg-cyan-500/10
        hover:shadow-[0_0_40px_rgba(34,211,238,.2)]
      "
    >
      <div className="absolute -left-12 -top-12 h-32 w-32 rounded-full bg-cyan-500/10 blur-3xl opacity-0 transition-opacity group-hover:opacity-100" />

      <div className="relative flex items-center gap-4">

        <div
          className="
            rounded-xl
            bg-cyan-500/15
            p-3
            transition-all
            group-hover:-rotate-6
            group-hover:bg-cyan-500/25
          "
        >
          <Icon className="h-7 w-7 text-cyan-300" />
        </div>

        <div>

          <h3 className="text-lg font-semibold text-white">
            {title}
          </h3>

          <p className="text-sm text-slate-400">
            {subtitle}
          </p>

        </div>

      </div>
    </motion.div>
  );
}