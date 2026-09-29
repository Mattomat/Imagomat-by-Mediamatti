/** Auswahl-Stufen fürs Culling (Start-Seite und Shoot-Ansicht). */
export type Selection = "locker" | "normal" | "streng" | "highlights";

export const SELECTIONS: [Selection, string][] = [
  ["locker", "Locker"],
  ["normal", "Normal"],
  ["streng", "Streng"],
  ["highlights", "Nur Highlights"],
];

export const SELECTION_PARAMS: Record<Selection, { keep_ratio: number; highlights: boolean }> = {
  locker: { keep_ratio: 0.35, highlights: false },
  normal: { keep_ratio: 0.2, highlights: false },
  streng: { keep_ratio: 0.1, highlights: false },
  highlights: { keep_ratio: 0.05, highlights: true },
};

export const SELECTION_HINT: Record<Selection, string> = {
  locker: "behält ca. 35 %",
  normal: "behält ca. 20 %",
  streng: "behält ca. 10 %",
  highlights: "nur die besten Momente: Zweikampf, Schuss, Parade, Jubel – ein Bild pro Spielszene (ca. 5 %)",
};

export function selectionFromSettings(settings: string | null | undefined): Selection {
  try {
    const c = JSON.parse(settings || "{}").culling ?? {};
    if (c.highlights) return "highlights";
    const r = c.keep_ratio ?? 0.2;
    return r >= 0.3 ? "locker" : r <= 0.12 ? "streng" : "normal";
  } catch {
    return "normal";
  }
}
