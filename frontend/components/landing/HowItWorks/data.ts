import {
  Activity,
  FileText,
  ScanSearch,
  Microscope,
  ShieldCheck,
  BrainCircuit,
  TrendingUp,
  Stethoscope,
} from "lucide-react";

export const inputs = [
  {
    title: "CT Scan",
    subtitle: "Medical Imaging",
    icon: ScanSearch,
  },
  {
    title: "Lab Reports",
    subtitle: "Clinical Biomarkers",
    icon: Microscope,
  },
  {
    title: "Clinical Notes",
    subtitle: "Patient History",
    icon: FileText,
  },
  {
    title: "Patient Vitals",
    subtitle: "Real-Time Monitoring",
    icon: Activity,
  },
];

export const outputs = [
  {
    title: "Tumor Detection",
    subtitle: "Computer Vision",
    icon: ShieldCheck,
  },
  {
    title: "Risk Prediction",
    subtitle: "Predictive AI",
    icon: TrendingUp,
  },
  {
    title: "Explainability",
    subtitle: "Transparent Decisions",
    icon: BrainCircuit,
  },
  {
    title: "Clinical Recommendation",
    subtitle: "Decision Support",
    icon: Stethoscope,
  },
];