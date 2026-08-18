"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface Props {
  title: string;
  subtitle: string;
  icon: LucideIcon;
}

export default function TechChip({
  title,
  subtitle,
  icon: Icon,
}: Props) {
  return (
    <motion.div
      whileHover={{
        y: -6,
        scale: 1.03,
      }}
      transition={{
        duration: 0.25,
      }}
      className="
        group
        rounded-2xl
        border
        border-white/10
        bg-white/5
        backdrop-blur-xl
        p-5
        hover:border-cyan-400/40
        hover:bg-cyan-500/10
        hover:shadow-[0_0_40px_rgba(34,211,238,.15)]
        transition-all
      "
    >
      <div className="flex items-center gap-4">

        <div
          className="
            rounded-xl
            bg-cyan-500/15
            p-3
            group-hover:bg-cyan-500/25
            transition-all
          "
        >
          <Icon className="h-6 w-6 text-cyan-300" />
        </div>

        <div>
          <h3 className="font-semibold text-white">
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