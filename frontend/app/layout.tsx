import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Demand Forecasting Co-Pilot",
  description:
    "Agentic demand forecasting: sense, forecast, validate, approve, plan.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
