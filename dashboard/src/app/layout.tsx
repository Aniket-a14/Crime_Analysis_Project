import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CrimeWatch | Intelligence Portal",
  description: "Advanced Law Enforcement Analytical Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={cn(inter.variable, jetbrainsMono.variable, "dark")} suppressHydrationWarning>
      <body className="min-h-screen bg-slate-950 font-sans antialiased overflow-x-hidden">
        {/* Intelligence Layers */}
        <div className="fixed inset-0 dot-grid opacity-100 pointer-events-none z-0"></div>
        <div className="fixed inset-0 noise-overlay pointer-events-none z-0"></div>
        <div className="scanline z-50"></div>
        
        <div className="relative z-10 flex flex-col min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
