"use client";

import { motion } from "framer-motion";
import Badge from "../ui/badge";
import GradientText from "../ui/gradient-text";

export default function HeroText() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -40 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.8 }}
      className="max-w-2xl"
    >
      <Badge>
        🚀 AI-Powered Clinical Intelligence Platform
      </Badge>

      <h1
        className="
          mt-8
          text-6xl
          font-bold
          leading-tight
          tracking-tight
        "
      >
        The Future of
        <br />

        <GradientText>
          Explainable Healthcare AI
        </GradientText>
      </h1>

      <p
        className="
          mt-8
          text-xl
          leading-9
          text-slate-400
        "
      >
        Analyze CT scans, MRI, laboratory reports,
        patient vitals, and clinical records using
        multimodal artificial intelligence that provides
        transparent predictions, explainable insights,
        and real-time decision support.
      </p>

      <div className="mt-12 flex gap-5">
        <button
          className="
            rounded-xl
            bg-cyan-500
            px-7
            py-4
            font-semibold
            text-black
            transition-all
            hover:scale-105
          "
        >
          Launch Platform
        </button>

        <button
          className="
            rounded-xl
            border
            border-white/10
            bg-white/5
            px-7
            py-4
            font-semibold
            backdrop-blur-xl
            transition-all
            hover:bg-white/10
          "
        >
          Watch Demo
        </button>
      </div>

      <div className="mt-14 flex gap-12">
        <div>
          <h2 className="text-3xl font-bold text-cyan-300">
            95%
          </h2>

          <p className="mt-1 text-slate-400">
            Diagnostic Accuracy
          </p>
        </div>

        <div>
          <h2 className="text-3xl font-bold text-cyan-300">
            5+
          </h2>

          <p className="mt-1 text-slate-400">
            Clinical Modalities
          </p>
        </div>

        <div>
          <h2 className="text-3xl font-bold text-cyan-300">
            Explainable
          </h2>

          <p className="mt-1 text-slate-400">
            AI Decisions
          </p>
        </div>
      </div>
    </motion.div>
  );
}