export default function ConnectionLines() {
  return (
    <svg
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 650 650"
      fill="none"
    >
      <defs>
        <linearGradient id="cyanGradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset="100%" stopColor="#3B82F6" />
        </linearGradient>
      </defs>

      {/* MRI */}
      <path
        d="M325 90 Q325 170 325 220"
        stroke="url(#cyanGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity=".65"
      />

      {/* CT */}
      <path
        d="M120 260 Q220 270 270 300"
        stroke="url(#cyanGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity=".65"
      />

      {/* Reports */}
      <path
        d="M530 250 Q450 280 380 300"
        stroke="url(#cyanGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity=".65"
      />

      {/* Vitals */}
      <path
        d="M180 520 Q250 430 285 365"
        stroke="url(#cyanGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity=".65"
      />

      {/* Genomics */}
      <path
        d="M480 520 Q405 430 365 365"
        stroke="url(#cyanGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity=".65"
      />
    </svg>
  );
}