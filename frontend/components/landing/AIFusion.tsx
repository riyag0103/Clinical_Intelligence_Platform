"use client";

import { motion } from "framer-motion";
import { BrainCircuit } from "lucide-react";

export default function AIFusion() {
  return (
    <motion.div
      animate={{
        scale: [1, 1.05, 1],
      }}
      transition={{
        duration: 3,
        repeat: Infinity,
      }}
      className="relative flex items-center justify-center"
    >
      {/* Glow */}
      <div className="absolute h-60 w-60 rounded-full bg-cyan-500/10 blur-[100px]" />

      {/* Outer */}
      <div className="absolute h-40 w-40 rounded-full border border-cyan-400/20" />

      {/* Rotating Ring 1 */}
      <motion.div
        className="absolute h-44 w-44 rounded-full border border-cyan-400/15"
        animate={{ rotate: 360 }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Rotating Ring 2 */}
      <motion.div
        className="absolute h-56 w-56 rounded-full border border-cyan-400/10 border-dashed"
        animate={{ rotate: -360 }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Core */}
      <div
        className="
          relative
          flex
          h-24
          w-24
          flex-col
          items-center
          justify-center
          rounded-full
          bg-gradient-to-br
          from-cyan-400
          to-sky-500
          shadow-[0_0_80px_rgba(34,211,238,.45)]
        "
      >
        <BrainCircuit className="mb-1 h-8 w-8 text-white" />

        <p className="text-xs font-bold tracking-widest text-white">
          EMCI
        </p>
      </div>
    </motion.div>
  );
}