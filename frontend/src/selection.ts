/** Auswahl-Stufen fürs Culling (Start-Seite und Shoot-Ansicht). */
export type Selection = "locker" | "normal" | "streng" | "highlights";

export const SELECTIONS: [Selection, string][] = [
  ["locker", "Nur Fehler raus"],
  ["normal", "Normal"],
  ["streng", "Streng"],
  ["highlights", "Nur Highlights"],
];

type Params = { keep_ratio: number; highlights: boolean; burst_keep: number };
export const SELECTION_PARAMS: Record<Selection, Params> = {
  locker: { keep_ratio: 1, highlights: false, burst_keep: 4 },
  normal: { keep_ratio: 1, highlights: false, burst_keep: 2 },
  streng: { keep_ratio: 0.2, highlights: false, burst_keep: 0 },
  highlights: { keep_ratio: 0.05, highlights: true, burst_keep: 0 },
};

export const SELECTION_HINT: Record<Selection, string> = {
  locker: "nur Unscharfes, geschlossene Augen, Fehlbelichtung raus; aus einer Serie mit gleichem Motiv bleiben die 4 besten",
  normal: "Fehler raus und aus jeder Serie mit (fast) gleichem Motiv nur die 2 besten, keine Prozent-Quote",
  streng: "behält ca. 20 % (die besten)",
  highlights: "nur die besten Momente: Zweikampf, Schuss, Parade, Jubel – ein Bild pro Spielszene (ca. 5 %)",
};

export function selectionFromSettings(settings: string | null | undefined): Selection {
  try {
    const c = JSON.parse(settings || "{}").culling ?? {};
    if (c.highlights) return "highlights";
    if (c.burst_keep !== undefined && c.burst_keep > 0) return c.burst_keep >= 3 ? "locker" : "normal";
    const r = c.keep_ratio ?? 1;
    return r <= 0.3 ? "streng" : "normal";
  } catch {
    return "normal";
  }
}
