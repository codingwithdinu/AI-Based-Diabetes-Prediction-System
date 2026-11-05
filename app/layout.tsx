import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GlucoWise | AI Health Screening",
  description:
    "An educational AI-powered diabetes risk screening demonstration using a Support Vector Machine model.",
  applicationName: "GlucoWise",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
