"use client";

import {
  Activity,
  FileText,
  Microscope,
  ScanSearch,
} from "lucide-react";

import AIFusion from "./AIFusion";
import ModalityNode from "./ModalityNode";
import ConnectionLines from "./ConnectionLines";

export default function AIEngine() {
  return (
    <div className="relative h-[760px] w-[760px] mx-auto">

  {/* Background Glow */}
  <div className="absolute inset-0 rounded-full bg-cyan-500/5 blur-[140px]" />

  {/* Connections */}
  <ConnectionLines />

  {/* AI Core */}
  <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20">
    <AIFusion />
  </div>

  {/* TOP */}
  <ModalityNode
    icon={ScanSearch}
    title="CT Scan"
    subtitle="Tumor Detection"
    delay={0}
    className="absolute left-1/2 top-10 -translate-x-1/2"
  />

  {/* LEFT */}
  <ModalityNode
    icon={FileText}
    title="Clinical Notes"
    subtitle="Medical History"
    delay={0.2}
    className="absolute left-[-20px] top-1/2 -translate-y-1/2"
  />

  {/* RIGHT */}
  <ModalityNode
    icon={Microscope}
    title="Lab Reports"
    subtitle="Blood Analysis"
    delay={0.4}
    className="absolute right-[-20px] top-1/2 -translate-y-1/2"
  />

  {/* BOTTOM */}
  <ModalityNode
    icon={Activity}
    title="Patient Vitals"
    subtitle="Live Monitoring"
    delay={0.6}
    className="absolute left-1/2 bottom-10 -translate-x-1/2"
  />

</div>
  );
}