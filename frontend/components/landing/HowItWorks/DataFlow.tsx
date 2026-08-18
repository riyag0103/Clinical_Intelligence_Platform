"use client";

import { motion } from "framer-motion";

const paths = [
  // Inputs → AI
  "M220 120 C320 120 360 170 470 250",
  "M220 240 C320 240 360 250 470 290",
  "M220 360 C320 360 360 330 470 330",
  "M220 480 C320 480 360 410 470 370",

  // AI → Outputs
  "M630 250 C740 170 800 120 900 120",
  "M630 290 C740 250 800 240 900 240",
  "M630 330 C740 330 800 360 900 360",
  "M630 370 C740 410 800 480 900 480",
];

export default function DataFlow() {
  return (
    <svg
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 1100 600"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="flowGradient" x1="0" x2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.15" />
          <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.15" />
        </linearGradient>

        <filter id="flowGlow">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {paths.map((path, index) => (
        <g key={index}>
          {/* Connection Line */}
          <motion.path
            d={path}
            fill="none"
            stroke="url(#flowGradient)"
            strokeWidth="2.5"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            viewport={{ once: true }}
            transition={{
              duration: 1,
              delay: index * 0.12,
            }}
          />

          {/* Packet 1 */}
          <circle
            r="4"
            fill="#22d3ee"
            filter="url(#flowGlow)"
          >
            <animateMotion
              dur={`${2.5 + index * 0.2}s`}
              repeatCount="indefinite"
              path={path}
            />
          </circle>

          {/* Packet 2 */}
          <circle
            r="3"
            fill="#7dd3fc"
            opacity="0.8"
            filter="url(#flowGlow)"
          >
            <animateMotion
              begin={`${index * 0.4}s`}
              dur={`${3.5 + index * 0.2}s`}
              repeatCount="indefinite"
              path={path}
            />
          </circle>
        </g>
      ))}

      {/* Pulse at AI Core */}
      <motion.circle
        cx="550"
        cy="300"
        r="12"
        fill="#22d3ee"
        filter="url(#flowGlow)"
        animate={{
          r: [10, 22, 10],
          opacity: [0.8, 0.2, 0.8],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
        }}
      />
    </svg>
  );
}