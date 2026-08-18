"use client";

import { motion } from "framer-motion";
import { Cpu } from "lucide-react";

export default function Navbar() {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7 }}
      className="mx-auto flex h-20 max-w-7xl items-center justify-between px-8"
    >
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-500/15 ring-1 ring-cyan-400/20">
          <Cpu className="h-6 w-6 text-cyan-400" />
        </div>

        <div>
          <h1 className="text-lg font-semibold tracking-wide">
            EMCI
          </h1>

          <p className="text-xs text-slate-400">
            Explainable Multimodal Clinical Intelligence
          </p>
        </div>
      </div>

      {/* Navigation */}
      <div className="hidden gap-10 text-sm text-slate-300 md:flex">
        <a href="#">Platform</a>
        <a href="#">Solutions</a>
        <a href="#">Technology</a>
        <a href="#">Research</a>
      </div>

      {/* CTA */}
      <button
        className="
          rounded-xl
          bg-cyan-500
          px-5
          py-2.5
          text-sm
          font-medium
          text-black
          transition-all
          hover:scale-105
        "
      >
        Get Started
      </button>
    </motion.nav>
  );
}