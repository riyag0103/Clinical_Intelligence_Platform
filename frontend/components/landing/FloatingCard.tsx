"use client";

import { motion } from "framer-motion";
import GlassCard from "../ui/glass-card";
import { LucideIcon } from "lucide-react";

interface FloatingCardProps {
  icon: LucideIcon;
  title: string;
  subtitle: string;
  className?: string;
  duration?: number;
  delay?: number;
  distance?: number;
}

export default function FloatingCard({
  icon: Icon,
  title,
  subtitle,
  className,
  duration = 3,
  delay = 0,
  distance = 20,
}: FloatingCardProps) {
  return (
    <motion.div
  className={className}
  animate={{
    y: [0, -distance, 0],
    rotate: [0, 1.5, 0],
  }}
  transition={{
    duration,
    delay,
    repeat: Infinity,
    repeatType: "mirror",
    ease: "easeInOut",
  }}
      whileHover={{
        scale: 1.06,
      }}
      
    >
      <GlassCard className="flex items-center gap-4 p-4">
        <div className="rounded-2xl bg-cyan-500/15 p-3">
          <Icon className="h-6 w-6 text-cyan-300" />
        </div>

        <div>
          <h3 className="font-medium text-white">
            {title}
          </h3>

          <p className="text-sm text-slate-400">
            {subtitle}
          </p>
        </div>
      </GlassCard>
    </motion.div>
  );

}