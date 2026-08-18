"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface ModalityNodeProps {
  icon: LucideIcon;
  title: string;
  subtitle: string;
  className?: string;
  delay?: number;
}

export default function ModalityNode({
  icon: Icon,
  title,
  subtitle,
  className,
  delay = 0,
}: ModalityNodeProps) {
  return (
    <motion.div
      initial={{
        opacity: 0,
        scale: 0.8,
      }}
      animate={{
        opacity: 1,
        scale: 1,
        y: [0, -10, 0],
      }}
      transition={{
        opacity: {
          delay,
          duration: 0.6,
        },
        scale: {
          delay,
          duration: 0.6,
        },
        y: {
          duration: 3 + delay,
          repeat: Infinity,
          ease: "easeInOut",
        },
      }}
      className={className}
    >
      <div
        className="
          flex
          items-center
          gap-3
          rounded-2xl
          border
          border-white/10
          bg-white/5
          px-4
          py-3
          backdrop-blur-xl
          transition-all
          hover:border-cyan-400/40
          hover:bg-cyan-500/10
        "
      >
        <div className="rounded-xl bg-cyan-500/15 p-3">
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