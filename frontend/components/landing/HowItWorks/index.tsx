"use client";

import { motion } from "framer-motion";

import InputCard from "./InputCard";
import OutputCard from "./OutputCard";
import AIProcessor from "./AIProcessor";
import DataFlow from "./DataFlow";

import { inputs, outputs } from "./data";

export default function HowItWorks() {
  return (
    <section className="relative py-40 overflow-hidden">

      {/* Background */}

      <div className="absolute inset-0">

        <div className="absolute left-1/2 top-1/2 h-[700px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-500/5 blur-[180px]" />

      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6">

        {/* Heading */}

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: .7 }}
          className="text-center"
        >
          <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-5 py-2 text-sm text-cyan-300">
            HOW EMCI WORKS
          </span>

          <h2 className="mt-8 text-5xl font-bold text-white">
            Multimodal Clinical Intelligence
          </h2>

          <p className="mx-auto mt-6 max-w-3xl text-lg leading-8 text-slate-400">
            EMCI intelligently combines medical imaging, clinical reports,
            patient history, and real-time vitals to generate explainable
            clinical insights and predictive recommendations.
          </p>
        </motion.div>

        {/* Workflow */}

        <div className="relative mt-32">

          <DataFlow />

          <div className="grid lg:grid-cols-3 gap-14 items-center">

            {/* Inputs */}

            <div className="space-y-6">

              {inputs.map((item, index) => (
                <InputCard
                  key={item.title}
                  {...item}
                  delay={index * .15}
                />
              ))}

            </div>

            {/* AI */}

            <div className="flex justify-center">

              <AIProcessor />

            </div>

            {/* Outputs */}

            <div className="space-y-6">

              {outputs.map((item, index) => (
                <OutputCard
                  key={item.title}
                  {...item}
                  delay={index * .15}
                />
              ))}

            </div>

          </div>

        </div>

      </div>

    </section>
  );
}