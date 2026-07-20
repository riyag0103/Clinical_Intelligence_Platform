import {
  Brain,
  HeartPulse,
  ScanSearch,
  Dna,
  FileText,
} from "lucide-react";

import FloatingCard from "./FloatingCard";
import ConnectionLines from "./ConnectionLines";
import CoreSphere from "./CoreSphere";

export default function AIEngine() {
  return (
    <div className="relative flex h-[650px] w-[650px] items-center justify-center">

      {/* Background Glow */}

      <div className="absolute h-[420px] w-[420px] rounded-full bg-cyan-500/20 blur-[120px]" />
      <ConnectionLines />
      <CoreSphere />

      {/* Cards */}

      <FloatingCard
        title="CT Scan"
        subtitle="DICOM Imaging"
        icon={ScanSearch}
        className="left-0 top-52"
      />

      <FloatingCard
        title="MRI"
        subtitle="3D Imaging"
        icon={Brain}
        className="left-44 top-0"
      />

      <FloatingCard
        title="Vitals"
        subtitle="Live Monitoring"
        icon={HeartPulse}
        className="left-10 bottom-20"
      />

      <FloatingCard
        title="Lab Reports"
        subtitle="Clinical Data"
        icon={FileText}
        className="right-0 top-44"
      />

      <FloatingCard
        title="Genomics"
        subtitle="DNA Analysis"
        icon={Dna}
        className="right-10 bottom-20"
      />
    </div>
  );
}