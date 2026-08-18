interface GradientTextProps {
  children: React.ReactNode;
}

export default function GradientText({
  children,
}: GradientTextProps) {
  return (
    <span
      className="
        bg-gradient-to-r
        from-cyan-300
        via-sky-400
        to-violet-400
        bg-clip-text
        text-transparent
      "
    >
      {children}
    </span>
  );
}