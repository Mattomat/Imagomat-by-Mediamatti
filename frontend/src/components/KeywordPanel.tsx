// Eigene Stichwörter im Shoot: anlegen (eigene oder Themenliste), Vorschläge suchen und bestätigen/verwerfen.
import { useCallback, useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, waitForJob } from "../api";

export interface ShootKeyword { id: number; name: string; prompt: string; theme: string | null; n: number; assigned: number; suggested: number }

const sortKws = (xs: ShootKeyword[]) => [...xs].sort((a, b) =>
  (b.assigned + b.suggested > 0 ? 1 : 0) - (a.assigned + a.suggested > 0 ? 1 : 0) || a.name.localeCompare(b.name));

export default function KeywordPanel({ ctx, shootId, items, onChanged, reviewId }: {
  ctx: AppCtx; shootId: number; items: ImageItem[]; onChanged: () => void; reviewId?: number | null;
}) {
  const [kws, setKws] = useState<ShootKeyword[]>([]);
  const [themes, setThemes] = useState<Record<string, string[]>>({});
  const [name, setName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState<number | null>(null);
  const [review, setReview] = useState<ShootKeyword | null>(null);
  const [editing, setEditing] = useState<number | null>(null);

  const load = useCallback(() => {
    api.get<ShootKeyword[]>(`/api/shoots/${shootId}/keywords`).then((r) => setKws(sortKws(r))).catch(() => undefined);
  }, [shootId]);
  useEffect(load, [load, ctx.tick]);
  // Nach "Bereich markieren": Vorschläge dieses Stichworts gleich zeigen
  useEffect(() => {
    const k = reviewId ? kws.find((x) => x.id === reviewId) : undefined;
    if (k && k.suggested > 0) setReview(k);
  }, [reviewId, kws]);
  useEffect(() => { api.get<{ themes: Record<string, string[]> }>("/api/keywords").then((r) => setThemes(r.themes)).catch(() => undefined); }, []);

  const add = async () => {
    if (!name.trim()) return;
    try {
      await api.post("/api/keywords", { name, prompt });
      setName(""); setPrompt("");
      load();
    } catch (e) { ctx.toast((e as Error).message, "error"); }
  };
  const addTheme = async (t: string) => {
    await api.post("/api/keywords", { theme: t });
    ctx.toast(`Liste „${t}“ hinzugefügt`);
    load();
  };
  const remove = async (k: ShootKeyword) => {
    await api.del(`/api/keywords/${k.id}`);
    load(); onChanged();
  };
  const savePrompt = async (k: ShootKeyword, p: string) => {
    await api.patch(`/api/keywords/${k.id}`, { prompt: p });
    setEditing(null); load();
  };
  const suggest = async (k: ShootKeyword) => {
    setBusy(k.id);
    try {
      const r = await api.post<{ job_id: number }>(`/api/shoots/${shootId}/keywords/${k.id}/suggest`, {});
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id);
      if (j.status !== "done") throw new Error(j.error?.split("\n").filter(Boolean).pop() ?? "Suche fehlgeschlagen");
      const fresh = await api.get<ShootKeyword[]>(`/api/shoots/${shootId}/keywords`);
      setKws(sortKws(fresh));
      const now = fresh.find((x) => x.id === k.id);
      if (now && now.suggested > 0) setReview(now);
      else ctx.toast(`Keine passenden Bilder für „${k.name}“ gefunden`);
    } catch (e) { ctx.toast((e as Error).message, "error"); } finally { setBusy(null); }
  };

  if (review) return <Review ctx={ctx} shootId={shootId} kw={review} items={items}
    onClose={() => { setReview(null); load(); onChanged(); }} />;

  return (
    <div className="who-panel kw-panel">
      <div className="who-head">
        <div>
          <h2>Stichwörter</h2>
          <p className="hint">Eigene Stichwörter für Objekte, Szenen oder Gruppen (z. B. „Fankurve“, „Maskottchen“). Mit Beschreibung
            schlägt Tagmatti passende Bilder vor. Von Hand: im Raster Bilder mit ⌘/Ctrl- oder Shift-Klick wählen und das Stichwort tippen.</p>
        </div>
      </div>
      <div className="kw-add">
        <input value={name} placeholder="Neues Stichwort, z. B. Maskottchen" onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") add(); e.stopPropagation(); }} />
        <input value={prompt} placeholder="Beschreibung für die Suche (optional, am besten Englisch: „costumed mascot“)"
          onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") add(); e.stopPropagation(); }} />
        <button className="primary" disabled={!name.trim()} onClick={add}>Hinzufügen</button>
      </div>
      <div className="kw-themes">
        <span className="muted">Fertige Liste hinzufügen:</span>
        {Object.keys(themes).map((t) => <button key={t} className="small" onClick={() => addTheme(t)}>{t}</button>)}
      </div>
      {kws.length === 0 && <div className="empty">Noch keine Stichwörter. Lege eines an oder füge eine Liste hinzu.</div>}
      <div className="kw-list">
        {kws.map((k) => (
          <div key={k.id} className="kw-row">
            <div className="kw-main">
              <b>{k.name}</b>
              {editing === k.id
                ? <input autoFocus defaultValue={k.prompt} placeholder="Beschreibung (Englisch)"
                    onBlur={(e) => savePrompt(k, e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") savePrompt(k, (e.target as HTMLInputElement).value); if (e.key === "Escape") setEditing(null); e.stopPropagation(); }} />
                : <button className="link kw-prompt" onClick={() => setEditing(k.id)}>{k.prompt || "Beschreibung hinzufügen"}</button>}
            </div>
            <span className="kw-count">{k.assigned} {k.assigned === 1 ? "Bild" : "Bilder"}</span>
            {k.suggested > 0
              ? <button onClick={() => setReview(k)}>{k.suggested} Vorschläge prüfen</button>
              : <button disabled={busy !== null || (!k.prompt && k.n === 0)} onClick={() => suggest(k)}
                  title={!k.prompt && k.n === 0 ? "Zuerst eine Beschreibung eingeben oder ein paar Bilder von Hand zuordnen" : "Passende Bilder in diesem Shoot suchen"}>
                  {busy === k.id ? "sucht …" : "Bilder vorschlagen"}</button>}
            <button className="ghost small" title="Stichwort löschen" onClick={() => remove(k)}>✕</button>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Vorschläge prüfen: alle sind vorausgewählt, Klick nimmt ein Bild raus. */
function Review({ ctx, shootId, kw, items, onClose }: { ctx: AppCtx; shootId: number; kw: ShootKeyword; items: ImageItem[]; onClose: () => void }) {
  const [sug, setSug] = useState<{ image_id: number; score: number }[]>([]);
  const [off, setOff] = useState<Set<number>>(new Set());
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    api.get<{ image_id: number; score: number }[]>(`/api/shoots/${shootId}/keywords/${kw.id}/suggestions`).then(setSug).catch(() => undefined);
  }, [shootId, kw.id]);
  const names = new Map(items.map((i) => [i.id, i.filename]));
  const toggle = (id: number) => setOff((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const save = async () => {
    setSaving(true);
    const yes = sug.filter((s) => !off.has(s.image_id)).map((s) => s.image_id);
    const no = sug.filter((s) => off.has(s.image_id)).map((s) => s.image_id);
    try {
      if (yes.length) await api.post("/api/images/keywords", { image_ids: yes, keyword_id: kw.id, state: "confirmed" });
      if (no.length) await api.post("/api/images/keywords", { image_ids: no, keyword_id: kw.id, state: "rejected" });
      ctx.toast(`„${kw.name}“: ${yes.length} Bilder übernommen`);
      onClose();
    } catch (e) { ctx.toast((e as Error).message, "error"); setSaving(false); }
  };
  return (
    <div className="who-panel kw-panel">
      <div className="who-head">
        <div>
          <h2>„{kw.name}“ – {sug.length} Vorschläge</h2>
          <p className="hint">Alle sind ausgewählt. Klicke die Bilder an, die nicht passen. Verworfene Bilder werden nie wieder vorgeschlagen.</p>
        </div>
        <div className="row">
          <button className="ghost" onClick={onClose}>Später</button>
          <button className="primary" disabled={saving || sug.length === 0} onClick={save}>{sug.length - off.size} übernehmen</button>
        </div>
      </div>
      <div className="kw-review">
        {sug.map((s) => (
          <button key={s.image_id} className={`kw-sug ${off.has(s.image_id) ? "off" : ""}`} onClick={() => toggle(s.image_id)}
            title={names.get(s.image_id) ?? ""} aria-pressed={!off.has(s.image_id)}>
            <img loading="lazy" draggable={false} src={api.img(`/api/images/${s.image_id}/thumb`)} />
            <span className="kw-mark">{off.has(s.image_id) ? "✕" : "✓"}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
