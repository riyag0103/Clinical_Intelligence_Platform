"use client";

import { motion } from "framer-motion";
import TechChip from "./TechChip";
import { technologies } from "@/data/technologies";

export default function TrustedTechnologies() {
  return (
    <section className="relative py-28">

      <div className="mx-auto max-w-7xl px-6">

        <motion.div
          initial={{
            opacity: 0,
            y: 40,
          }}
          whileInView={{
            opacity: 1,
            y: 0,
          }}
          viewport={{ once: true }}
          transition={{
            duration: 0.7,
          }}
          className="text-center"
        >

          <span
            className="
              rounded-full
              border
              border-cyan-500/30
              bg-cyan-500/10
              px-4
              py-2
              text-sm
              text-cyan-300
            "
          >
            POWERING EMCI
          </span>

          <h2 className="mt-6 text-5xl font-bold text-white">
            Built with Modern AI Technologies
          </h2>

          <p className="mx-auto mt-6 max-w-3xl text-lg leading-8 text-slate-400">
            EMCI combines state-of-the-art deep learning, medical imaging,
            multimodal intelligence, and modern cloud technologies to deliver
            explainable clinical decision support.
          </p>

        </motion.div>

        <div className="mt-20 grid gap-6 md:grid-cols-2 lg:grid-cols-3">

          {technologies.map((tech, index) => (
            <motion.div
              key={tech.title}
              initial={{
                opacity: 0,
                y: 40,
              }}
              whileInView={{
                opacity: 1,
                y: 0,
              }}
              viewport={{ once: true }}
              transition={{
                delay: index * 0.08,
              }}
            >
              <TechChip {...tech} />
            </motion.div>
          ))}

        </div>

      </div>

    </section>
  );
}