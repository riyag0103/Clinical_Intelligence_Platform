"use client";

import Link from "next/link";
import Button from "@/components/ui/Button";

const navLinks = [
  { name: "Features", href: "#features" },
  { name: "Technology", href: "#technology" },
  { name: "Solutions", href: "#solutions" },
  { name: "About", href: "#about" },
];

export default function Navbar() {
  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b border-slate-200/60 bg-slate-950/60 backdrop-blur-3xl">
      <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6">

        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 text-white shadow-lg">
            🧠
          </div>

          <div>
            <h1 className="text-xl font-bold text-slate-900">
              EMCI
            </h1>

            <p className="text-xs text-slate-500">
              Clinical Intelligence
            </p>
          </div>
        </Link>

        {/* Desktop Menu */}
        <nav className="hidden md:flex items-center gap-10">
          {navLinks.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className="text-sm font-medium text-slate-300 transition hover:text-cyan-400"
            >
              {item.name}
            </Link>
          ))}
        </nav>

        {/* Right Side */}
        <div className="hidden md:flex items-center gap-4">
          <button className="text-sm font-medium text-slate-700 hover:text-blue-600 transition">
            Login
          </button>

          <Button>
            Get Started
          </Button>
        </div>

      </div>
    </header>
  );
}