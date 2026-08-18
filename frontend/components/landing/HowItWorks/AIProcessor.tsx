"use client";

import { motion } from "framer-motion";
import { BrainCircuit, Sparkles } from "lucide-react";

export default function AIProcessor() {
  return (
    <div className="relative flex h-[420px] w-[420px] items-center justify-center">

      {/* Ambient Glow */}
      <div className="absolute h-[320px] w-[320px] rounded-full bg-cyan-500/10 blur-[120px]" />

      {/* Outer Ring */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: "linear",
        }}
        className="
          absolute
          h-[300px]
          w-[300px]
          rounded-full
          border
          border-cyan-400/10
        "
      />

      {/* Dashed Ring */}
      <motion.div
        animate={{ rotate: -360 }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
        className="
          absolute
          h-[240px]
          w-[240px]
          rounded-full
          border-2
          border-dashed
          border-cyan-400/20
        "
      />

      {/* Orbit Dots */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "linear",
        }}
        className="absolute h-[300px] w-[300px]"
      >
        <div className="absolute left-1/2 top-0 h-3 w-3 -translate-x-1/2 rounded-full bg-cyan-400 shadow-[0_0_15px_rgba(34,211,238,.8)]" />
        <div className="absolute bottom-0 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full bg-cyan-400 shadow-[0_0_15px_rgba(34,211,238,.8)]" />
      </motion.div>

      {/* Core */}
      <motion.div
        animate={{
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
        }}
        className="
          relative
          flex
          h-44
          w-44
          flex-col
          items-center
          justify-center
          rounded-full
          border
          border-cyan-400/30
          bg-gradient-to-br
          from-cyan-400
          via-sky-500
          to-blue-600
          shadow-[0_0_80px_rgba(34,211,238,.45)]
        "
      >
        {/* Sparkle */}

        <motion.div
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "linear",
          }}
          className="absolute inset-0"
        >
          <Sparkles className="absolute right-5 top-5 h-5 w-5 text-cyan-100/70" />
        </motion.div>

        <BrainCircuit className="h-12 w-12 text-white" />

        <h3 className="mt-4 text-lg font-bold text-white">
          EMCI AI
        </h3>

        <p className="mt-1 text-center text-xs text-cyan-100">
          Multimodal
          <br />
          Intelligence Engine
        </p>
      </motion.div>

      {/* Floating Labels */}

      <motion.div
        animate={{
          y: [0, -10, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
        }}
        className="
          absolute
          -left-2
          top-10
          rounded-full
          border
          border-cyan-400/20
          bg-cyan-500/10
          px-4
          py-2
          text-xs
          text-cyan-300
          backdrop-blur-xl
        "
      >
        Fusion
      </motion.div>

      <motion.div
        animate={{
          y: [0, 10, 0],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
        }}
        className="
          absolute
          -right-2
          bottom-12
          rounded-full
          border
          border-cyan-400/20
          bg-cyan-500/10
          px-4
          py-2
          text-xs
          text-cyan-300
          backdrop-blur-xl
        "
      >
        Explainable AI
      </motion.div>

    </div>
  );
}