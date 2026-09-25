import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ImageItem, Shoot } from "../api";

const LABEL_COLORS: Record<string, string> = {
  rot: "#d9534f", red: "#d9534f", gelb: "#e6c229", yellow: "#e6c229", grün: "#4caf50", green: "#4caf50",
  blau: "#3f7fd9", blue: "#3f7fd9", lila: "#9c5fc7", violett: "#9c5fc7", purple: "#9c5fc7",
};

type Filter = { decision: "all" | "keep" | "reject"; minStars: number; person: string; reason: string; review: boolean };

export default function ReviewView({ shoot, tick }: { shoot: Shoot | null; tick: number }) {
  const [items, setItems] = useState<ImageItem[]>([]);
  const [sel, setSel] = useState(0);
  const [loupe, setLoupe] = useState(false);
  const [after, setAfter] = useState(true);
  const [split, setSplit] = useState(false);
  const [filter, setFilter] = useState<Filter>({ decision: "all", minStars: 0, person: "", reason: "", review: false });
  const gridRef = useRef<HTMLDivElement>(null);

  const load = useCallback(() => {
    if (shoot) api.get<ImageItem[]>(`/api/shoots/${shoot.id}/images`).then(setItems);
  }, [shoot]);
  useEffect(load, [load, tick]);

  const shown = useMemo(
    () =>
      items.filter(
        (i) =>
          (filter.decision === "all" || i.decision === filter.decision) &&
          (i.rating ?? 0) >= filter.minStars &&
          (!filter.person || i.people.includes(filter.person)) &&
          (!filter.reason || i.reasons.includes(filter.reason)) &&
          (!filter.review || (i.confidence !== null && i.confidence < 0.35)),
      ),
    [items, filter],
  );
  const cur = shown[Math.min(sel, shown.length - 1)];
  const people = useMemo(() => [...new Set(items.flatMap((i) => i.people))].sort(), [items]);
  const reasons = useMemo(() => [...new Set(items.flatMap((i) => i.reasons))].sort(), [items]);

  const patch = async (i: ImageItem, body: Partial<Pick<ImageItem, "decision" | "rating" | "label">>) => {
    await api.patch(`/api/images/${i.id}/culling`, body);
    setItems((xs) => xs.map((x) => (x.id === i.id ? { ...x, ...body, manual: true } : x)));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      if (!cur) return;
      const cols = gridRef.current ? Math.max(1, Math.floor(gridRef.current.clientWidth / 212)) : 6;
      const k = e.key;
      if (k === "ArrowRight") setSel((s) => Math.min(shown.length - 1, s + 1));
      else if (k === "ArrowLeft") setSel((s) => Math.max(0, s - 1));
      else if (k === "ArrowDown" && !loupe) setSel((s) => Math.min(shown.length - 1, s + cols));
      else if (k === "ArrowUp" && !loupe) setSel((s) => Math.max(0, s - cols));
      else if (/^[0-5]$/.test(k)) patch(cur, { rating: +k, decision: +k > 0 && cur.decision === "reject" ? "keep" : cur.decision ?? "keep" });
      else if (k === "x" || k === "X") patch(cur, { decision: "reject" });
      else if (k === "p" || k === "P") patch(cur, { decision: "keep" });
      else if (k === "Enter" || k === " " || k === "e" || k === "E") setLoupe((l) => !l);
      else if (k === "Escape" || k === "g" || k === "G") setLoupe(false);
      else if (k === "\\") setAfter((a) => !a);
      else if (k === "y" || k === "Y") setSplit((s) => !s);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cur, shown.length, loupe]);

  useEffect(() => {
    document.getElementById(`t-${cur?.id}`)?.scrollIntoView({ block: "nearest" });
  }, [cur?.id]);

  if (!shoot) return <div className="page"><p>Bitte zuerst unter „Import“ einen Shoot öffnen.</p></div>;

  const kept = items.filter((i) => i.decision === "keep").length;

  return (
    <div className="review">
      <aside className="filters">
        <h3>{shoot.name}</h3>
        <p>{kept} von {items.length} behalten</p>
        <label>Entscheidung</label>
        <select value={filter.decision} onChange={(e) => setFilter({ ...filter, decision: e.target.value as Filter["decision"] })}>
          <option value="all">alle</option>
          <option value="keep">behalten</option>
          <option value="reject">aussortiert</option>
        </select>
        <label>Mindestens Sterne</label>
        <select value={filter.minStars} onChange={(e) => setFilter({ ...filter, minStars: +e.target.value })}>
          {[0, 1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n === 0 ? "alle" : "★".repeat(n)}</option>)}
        </select>
        <label>Person</label>
        <select value={filter.person} onChange={(e) => setFilter({ ...filter, person: e.target.value })}>
          <option value="">alle</option>
          {people.map((p) => <option key={p}>{p}</option>)}
        </select>
        <label>Culling-Grund</label>
        <select value={filter.reason} onChange={(e) => setFilter({ ...filter, reason: e.target.value })}>
          <option value="">alle</option>
          {reasons.map((r) => <option key={r}>{r}</option>)}
        </select>
        <label className="check">
          <input type="checkbox" checked={filter.review} onChange={(e) => setFilter({ ...filter, review: e.target.checked })} />
          nur unsichere Entwicklungen
        </label>
        <div className="keys">
          <b>Tasten</b>
          <span>← → ↑ ↓ navigieren</span>
          <span>0–5 Sterne, X ablehnen, P behalten</span>
          <span>Enter Lupe, Esc Raster</span>
          <span>\ Vorher/Nachher, Y geteilt</span>
        </div>
      </aside>
      {!loupe ? (
        <div className="grid" ref={gridRef}>
          {shown.map((i, idx) => (
            <div
              key={i.id}
              id={`t-${i.id}`}
              className={`thumb ${idx === sel ? "sel" : ""} ${i.decision === "reject" ? "rej" : ""}`}
              onClick={() => setSel(idx)}
              onDoubleClick={() => { setSel(idx); setLoupe(true); }}
            >
              <img loading="lazy" src={api.img(`/api/images/${i.id}/preview`)} />
              <div className="meta">
                <span className="stars">{"★".repeat(i.rating ?? 0)}</span>
                {i.label && <span className="label" style={{ background: LABEL_COLORS[i.label.toLowerCase()] ?? "#888" }} />}
                {i.best && <span className="badge">Bestes</span>}
                {i.denoise ? <span className="badge dn">DN {i.denoise}</span> : null}
                {i.decision === "reject" && <span className="badge rej">✕ {i.reasons[0] ?? ""}</span>}
                {i.manual && <span className="badge man">manuell</span>}
              </div>
            </div>
          ))}
        </div>
      ) : (
        cur && (
          <div className="loupe">
            <div className="stage">
              {split ? (
                <div className="split">
                  <figure><img src={api.img(`/api/images/${cur.id}/preview`)} /><figcaption>Vorher (Kamera-JPEG)</figcaption></figure>
                  <figure><img src={api.img(`/api/images/${cur.id}/render?size=1600`)} /><figcaption>Nachher (Annäherung)</figcaption></figure>
                </div>
              ) : (
                <figure>
                  <img
                    key={`${cur.id}-${after}`}
                    src={api.img(after ? `/api/images/${cur.id}/render?size=2000` : `/api/images/${cur.id}/preview`)}
                  />
                  <figcaption>{after ? "Nachher" : "Vorher (Kamera-JPEG)"}</figcaption>
                </figure>
              )}
              <div className="approx">Vorschau ist eine Annäherung an Lightroom, nicht das Endergebnis.</div>
            </div>
            <aside className="info">
              <h3>{cur.filename}</h3>
              <p>{cur.decision === "keep" ? "Behalten" : "Aussortiert"} · {"★".repeat(cur.rating ?? 0) || "keine Sterne"}{cur.best ? " · Bestes der Serie" : ""}</p>
              {cur.reasons.length > 0 && <p className="reasons">Gründe: {cur.reasons.join(", ")}</p>}
              {cur.people.length > 0 && <p>Personen: {cur.people.join(", ")}</p>}
              <p>ISO {cur.iso ?? "?"} · Serie {cur.series ?? "–"} · Score {cur.score?.toFixed(2) ?? "–"}</p>
              <p>Preset: {cur.preset ?? "–"} · Vertrauen {cur.confidence !== null ? Math.round(cur.confidence * 100) + " %" : "–"}</p>
              {cur.denoise ? <p>Denoise: {cur.denoise}</p> : null}
              {cur.notes.length > 0 && <p>{cur.notes.join(", ")}</p>}
              <div className="row">
                <button onClick={() => patch(cur, { decision: "keep" })}>Behalten (P)</button>
                <button onClick={() => patch(cur, { decision: "reject" })}>Ablehnen (X)</button>
              </div>
            </aside>
          </div>
        )
      )}
    </div>
  );
}
