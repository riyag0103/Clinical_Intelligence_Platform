"use client";

import { motion } from "framer-motion";

const lines = [
  {
    d: "M380 120 Q380 220 380 320",
    delay: 0,
  },
  {
    d: "M140 380 Q250 380 320 380",
    delay: 0.2,
  },
  {
    d: "M620 380 Q510 380 440 380",
    delay: 0.4,
  },
  {
    d: "M380 640 Q380 520 380 440",
    delay: 0.6,
  },
];

export default function ConnectionLines() {
  return (
    <svg
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 760 760"
    >
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {lines.map((line, i) => (
        <g key={i}>
          <motion.path
            d={line.d}
            fill="none"
            stroke="rgba(34,211,238,.25)"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{
              duration: 1,
              delay: line.delay,
            }}
          />

          <circle
            r="4"
            fill="#22d3ee"
            filter="url(#glow)"
          >
            <animateMotion
              dur="3s"
              repeatCount="indefinite"
              path={line.d}
            />
          </circle>
        </g>
      ))}
    </svg>
  );
}