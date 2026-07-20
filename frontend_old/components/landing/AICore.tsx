import {
  Brain,
  HeartPulse,
  Activity,
  ScanSearch,
  Dna,
  FileText,
} from "lucide-react";

const nodes = [
  {
    icon: Brain,
    label: "MRI",
    className: "-top-6 left-1/2 -translate-x-1/2",
  },
  {
    icon: HeartPulse,
    label: "Vitals",
    className: "top-1/2 -left-8 -translate-y-1/2",
  },
  {
    icon: Activity,
    label: "ECG",
    className: "top-1/2 -right-8 -translate-y-1/2",
  },
  {
    icon: ScanSearch,
    label: "CT",
    className: "bottom-6 left-8",
  },
  {
    icon: Dna,
    label: "Genomics",
    className: "bottom-6 right-8",
  },
  {
    icon: FileText,
    label: "Reports",
    className: "top-10 right-12",
  },
];

export default function AICore() {
  return (
    <div className="relative flex h-[500px] w-[500px] items-center justify-center">

      {/* Background Glow */}
      <div className="absolute h-80 w-80 rounded-full bg-cyan-200 blur-[120px] opacity-40" />

      {/* Center Core */}
      <div className="relative z-10 flex h-36 w-36 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-2xl">

        <div className="text-center">

          <Brain size={40} />

          <p className="mt-2 text-sm font-semibold">
            AI Core
          </p>

        </div>

      </div>

      {/* Floating Nodes */}

      {nodes.map(({ icon: Icon, label, className }) => (
        <div
          key={label}
          className={`absolute ${className}`}
        >
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-white shadow-xl">

            <Icon size={28} className="text-blue-600" />

          </div>

          <p className="mt-2 text-center text-xs font-medium text-slate-600">
            {label}
          </p>

        </div>
      ))}

    </div>
  );
}