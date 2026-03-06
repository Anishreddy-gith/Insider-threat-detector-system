import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes safely – avoids conflicts like `p-2 p-4`. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Map a 0-100 risk score to a semantic colour class. */
export function riskColor(score: number): string {
  if (score >= 75) return "text-risk-critical";
  if (score >= 50) return "text-risk-high";
  if (score >= 25) return "text-risk-medium";
  return "text-risk-low";
}

export function riskBgColor(score: number): string {
  if (score >= 75) return "bg-risk-critical";
  if (score >= 50) return "bg-risk-high";
  if (score >= 25) return "bg-risk-medium";
  return "bg-risk-low";
}

export function riskLabel(score: number): string {
  if (score >= 75) return "Critical";
  if (score >= 50) return "High";
  if (score >= 25) return "Medium";
  return "Low";
}
