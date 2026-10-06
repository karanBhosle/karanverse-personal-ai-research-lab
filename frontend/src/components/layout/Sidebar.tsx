"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

const NAV = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/research", label: "Research" },
  { href: "/knowledge", label: "Knowledge" },
  { href: "/graph", label: "Graph" },
  { href: "/history", label: "History" },
  { href: "/evaluation", label: "Evaluation" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-56 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/80">
      <div className="border-b border-zinc-800 px-5 py-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400/90">karanVerse</p>
        <h1 className="mt-1 text-lg font-semibold text-zinc-50">Research Lab</h1>
        <p className="mt-1 text-xs text-zinc-500">Evidence-first intelligence</p>
      </div>
      <nav className="flex flex-1 flex-col gap-1 p-3">
        {NAV.map((item) => {
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active ? "bg-zinc-800 text-zinc-50" : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100",
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
