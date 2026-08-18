interface BadgeProps {
  children: React.ReactNode;
}

export default function Badge({ children }: BadgeProps) {
  return (
    <div
      className="
        inline-flex
        items-center
        rounded-full
        border
        border-cyan-400/20
        bg-cyan-500/10
        px-4
        py-2
        text-sm
        text-cyan-300
        backdrop-blur-xl
      "
    >
      {children}
    </div>
  );
}