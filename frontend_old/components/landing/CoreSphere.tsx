"use client";

import { motion } from "framer-motion";
import { Brain } from "lucide-react";

export default function CoreSphere() {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{
        scale: [1, 1.03, 1],
        opacity: 1,
      }}
      transition={{
        opacity: { duration: 0.6 },
        scale: {
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
        },
      }}
      className="relative z-20 flex h-[280px] w-[280px] items-center justify-center"
    >
      {/* Outer Glow */}
      <div className="absolute h-[360px] w-[360px] rounded-full bg-cyan-500/20 blur-[110px]" />

      {/* Secondary Glow */}
      <div className="absolute h-[290px] w-[290px] rounded-full bg-blue-500/10 blur-[70px]" />

      {/* Glass Ring */}
      <div className="absolute h-[270px] w-[270px] rounded-full border border-cyan-400/20 bg-white/[0.03] backdrop-blur-xl shadow-[0_0_60px_rgba(34,211,238,0.15)]" />

      {/* Main Sphere */}
      <div className="relative z-10 flex h-[240px] w-[240px] flex-col items-center justify-center rounded-full border border-cyan-400/15 bg-gradient-to-br from-slate-900 via-[#0B1220] to-black shadow-[0_20px_80px_rgba(0,0,0,.55)]">
        <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-cyan-400/20 bg-cyan-400/10">
          <Brain className="h-9 w-9 text-cyan-400" />
        </div>

        <h2 className="text-5xl font-bold tracking-wide text-white">
          EMCI
        </h2>

        <p className="mt-2 text-center text-sm leading-6 text-slate-400">
          Clinical Intelligence
          <br />
          Engine
        </p>

        <div className="mt-6 flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-medium text-emerald-300">
            AI Online
          </span>
        </div>
      </div>
    </motion.div>
  );
}