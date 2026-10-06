import { cn } from "@/lib/cn";

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "accent" | "success" | "warn" | "danger";
  className?: string;
}) {
  const tones = {
    neutral: "bg-zinc-800 text-zinc-300 border-zinc-700",
    accent: "bg-cyan-950/80 text-cyan-200 border-cyan-800",
    success: "bg-emerald-950/80 text-emerald-200 border-emerald-800",
    warn: "bg-amber-950/80 text-amber-200 border-amber-800",
    danger: "bg-rose-950/80 text-rose-200 border-rose-800",
  };
  return (
    <span className={cn("inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium", tones[tone], className)}>
      {children}
    </span>
  );
}
