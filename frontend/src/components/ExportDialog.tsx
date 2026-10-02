// Export wie in Lightroom: Speicherort, Dateiformate (RAW + XMP, JPEG, WebP, PNG, TIFF), Grösse, Qualität,
// Dateinamen, KI-Entrauschen und was danach passieren soll.
import { useState } from "react";
import type { AppCtx } from "../App";
import { api, IS_APP, pickFolder, reveal, Shoot, waitForJob } from "../api";
import { Modal, Progress } from "../ui";

type Fmt = "xmp" | "jpeg" | "webp" | "png" | "tiff";
const FORMATS: [Fmt, string, string][] = [
  ["xmp", "RAW + XMP (für Lightroom)", "Originale RAWs mit allen Einstellungen, Masken, Sternen und Personen"],
  ["jpeg", "JPEG", "fertige Bilder, überall lesbar"],
  ["webp", "WebP", "kleiner als JPEG, gut fürs Web"],
  ["png", "PNG", "verlustfrei"],
  ["tiff", "TIFF", "verlustfrei, für Druck und Retusche"],
];
const stored = (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const store = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* egal */ } };

export default function ExportDialog({ ctx, shoot, kept, total, onClose }: {
  ctx: AppCtx; shoot: Shoot; kept: number; total: number; onClose: () => void;
}) {
  const inLibrary = (() => { try { return !!JSON.parse(shoot.settings || "{}").library; } catch { return false; } })();
  const parent = shoot.folder.replace(/\/[^/]+\/?$/, "");
  const [root, setRoot] = useState(stored("imagomat.exportRoot", parent));
  const [sub, setSub] = useState(shoot.name);
  const [fmts, setFmts] = useState<Fmt[]>(() => { try { return JSON.parse(stored("imagomat.exportFmts", '["xmp"]')); } catch { return ["xmp"]; } });
  const [inplace, setInplace] = useState(inLibrary);
  const [which, setWhich] = useState<"keep" | "all">(inLibrary ? "all" : "keep");
  const [minStars, setMinStars] = useState(0);
  const [sizeMode, setSizeMode] = useState<"orig" | "long">(stored("imagomat.exportSize", "orig") as "orig" | "long");
  const [longSide, setLongSide] = useState(+stored("imagomat.exportLong", "3000"));
  const [quality, setQuality] = useState(+stored("imagomat.exportQ", "90"));
  const [naming, setNaming] = useState<"original" | "custom">("original");
  const [base, setBase] = useState(shoot.name.replace(/[/:]/g, "-"));
  const [denoise, setDenoise] = useState<"off" | "edit" | "all">(stored("imagomat.exportDn", "edit") as "off" | "edit" | "all");
  const [openLr, setOpenLr] = useState(stored("imagomat.openLr", "1") !== "0");
  const [state, setState] = useState<"form" | "busy" | "done">("form");
  const [progress, setProgress] = useState(0);
  const [msg, setMsg] = useState("");
  const [final, setFinal] = useState<{ target: string; render?: string } | null>(null);

  const raw = fmts.includes("xmp");
  const rendered = fmts.filter((f) => f !== "xmp");
  const folder = sub.trim() ? `${root.replace(/\/$/, "")}/${sub.trim()}` : root;
  const toggle = (f: Fmt) => setFmts((xs) => { const n = xs.includes(f) ? xs.filter((x) => x !== f) : [...xs, f]; store("imagomat.exportFmts", JSON.stringify(n)); return n; });
  const count = which === "all" ? total : kept;

  const go = async () => {
    setState("busy");
    store("imagomat.exportSize", sizeMode); store("imagomat.exportLong", String(longSide)); store("imagomat.exportQ", String(quality));
    store("imagomat.exportDn", denoise);
    try {
      const useInplace = raw && inplace;
      const r = await api.post<{ job_id: number; target: string }>(`/api/shoots/${shoot.id}/export`, {
        target: useInplace ? shoot.folder : folder, formats: fmts, copy_mode: useInplace ? "inplace" : "hardlink",
        include_rejected: which === "all", min_rating: minStars, long_side: sizeMode === "orig" ? 0 : longSide, quality,
        naming, name_base: base, denoise, render_target: useInplace && rendered.length ? folder : undefined,
      });
      setFinal({ target: r.target, render: useInplace && rendered.length ? folder : undefined });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, (x) => { setProgress(x.total ? x.progress / x.total : 0); setMsg(x.message ?? ""); });
      if (j.status !== "done") throw new Error(j.error?.split("\n")[0] ?? "Export fehlgeschlagen");
      setState("done");
      if (raw && openLr && IS_APP) api.post("/api/lightroom/open", { path: r.target }).catch((e) => ctx.toast(`Lightroom: ${(e as Error).message}`, "error"));
    } catch (e) {
      ctx.toast((e as Error).message, "error");
      setState("form");
    }
  };

  return (
    <Modal title="Exportieren" onClose={onClose} wide>
      {state === "form" && (
        <div className="exp">
          <section>
            <h3>Dateiformat</h3>
            <div className="exp-fmts">
              {FORMATS.map(([f, l, hint]) => (
                <label key={f} className={`exp-fmt ${fmts.includes(f) ? "on" : ""}`}>
                  <input type="checkbox" checked={fmts.includes(f)} onChange={() => toggle(f)} />
                  <span><b>{l}</b><small>{hint}</small></span>
                </label>
              ))}
            </div>
            {rendered.length > 0 && <>
              <h3>Bildgrösse und Qualität</h3>
              <div className="exp-row">
                <label className="check"><input type="radio" checked={sizeMode === "orig"} onChange={() => setSizeMode("orig")} /> Originalgrösse</label>
                <label className="check"><input type="radio" checked={sizeMode === "long"} onChange={() => setSizeMode("long")} /> lange Kante</label>
                <input type="number" className="exp-num" min={300} max={12000} value={longSide} disabled={sizeMode !== "long"} onChange={(e) => setLongSide(+e.target.value)} /> px
              </div>
              {(fmts.includes("jpeg") || fmts.includes("webp")) && (
                <div className="exp-row"><span className="exp-lbl">Qualität</span>
                  <input type="range" min={50} max={100} value={quality} onChange={(e) => setQuality(+e.target.value)} /><b>{quality}</b></div>
              )}
              <h3>Dateinamen</h3>
              <div className="exp-row">
                <label className="check"><input type="radio" checked={naming === "original"} onChange={() => setNaming("original")} /> wie das Original (DSC00022.jpg)</label>
                <label className="check"><input type="radio" checked={naming === "custom"} onChange={() => setNaming("custom")} /> eigener Name</label>
                <input value={base} disabled={naming !== "custom"} onChange={(e) => setBase(e.target.value)} className="exp-base" />
                {naming === "custom" && <span className="muted">→ {base}-001.jpg</span>}
              </div>
            </>}
          </section>
          <section>
            <h3>Speicherort</h3>
            {raw && inLibrary && (
              <label className="check"><input type="checkbox" checked={inplace} onChange={(e) => setInplace(e.target.checked)} />
                RAW: Einstellungen (XMP) direkt neben die RAWs am Ablageort schreiben (empfohlen)</label>
            )}
            {(!raw || !inplace || rendered.length > 0) && <>
              <div className="row">
                <input value={root} onChange={(e) => { setRoot(e.target.value); store("imagomat.exportRoot", e.target.value); }} />
                <button onClick={async () => { const d = await pickFolder("Export-Ordner wählen"); if (d) { setRoot(d); store("imagomat.exportRoot", d); } }}>Wählen…</button>
              </div>
              <label>Unterordner</label>
              <input value={sub} onChange={(e) => setSub(e.target.value)} placeholder="leer = direkt in den Ordner" />
              <div className="hint">{raw && inplace ? "Fertige Bilder" : "Alles"} landet in: <b>{folder}</b></div>
            </>}
            <h3>Bilder</h3>
            <div className="exp-row">
              <label className="check"><input type="radio" checked={which === "keep"} onChange={() => setWhich("keep")} /> nur behaltene ({kept})</label>
              <label className="check"><input type="radio" checked={which === "all"} onChange={() => setWhich("all")} /> alle ({total}, aussortierte mit 1 Stern{rendered.length ? ", nur als RAW" : ""})</label>
            </div>
            <div className="exp-row"><span className="exp-lbl">Sterne</span>
              <select value={minStars} onChange={(e) => setMinStars(+e.target.value)}>
                <option value={0}>alle</option>{[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{"★".repeat(n)} und mehr</option>)}
              </select></div>
            <h3>KI-Entrauschen</h3>
            <select value={denoise} onChange={(e) => setDenoise(e.target.value as "off" | "edit" | "all")}>
              <option value="edit">wie pro Bild eingestellt (Details → KI-Entrauschen)</option>
              <option value="all">alle Bilder entrauschen</option>
              <option value="off">aus</option>
            </select>
            <div className="hint">{raw ? "Für Lightroom entsteht wie beim Lightroom-Entrauschen eine DNG „…-Enhanced-NR.dng“ neben dem RAW, mit deinen Einstellungen. " : ""}
              Rechnet mit dem KI-Modell (NAFNet) auf deinem Mac, ca. 20–60 s pro Bild.</div>
            {raw && IS_APP && <label className="check"><input type="checkbox" checked={openLr}
              onChange={(e) => { setOpenLr(e.target.checked); store("imagomat.openLr", e.target.checked ? "1" : "0"); }} /> danach in Lightroom importieren</label>}
          </section>
          <div className="modal-foot exp-foot">
            <span className="muted">{count} Bilder · {fmts.length ? fmts.map((f) => FORMATS.find((x) => x[0] === f)![1].split(" ")[0]).join(" + ") : "kein Format gewählt"}</span>
            <span className="spacer" />
            <button className="ghost" onClick={onClose}>Abbrechen</button>
            <button className="primary" disabled={!fmts.length} onClick={go}>Exportieren</button>
          </div>
        </div>
      )}
      {state === "busy" && <><Progress value={progress} label={msg || "Exportiere …"} /><p className="hint">Du kannst das Fenster schliessen, der Export läuft weiter.</p></>}
      {state === "done" && final && (
        <>
          <p className="success">Fertig exportiert.</p>
          {raw && <ol className="steps">
            <li>In Lightroom: <b>Importieren</b> → Ordner <code>{final.target.split("/").pop()}</code> → oben <b>Hinzufügen</b> → <b>Importieren</b>.</li>
            <li>Masken mit Motiv/Himmel: Bilder markieren → <b>Einstellungen → KI-Einstellungen aktualisieren</b>.</li>
          </ol>}
          <div className="modal-foot">
            <button onClick={() => reveal(final.render ?? final.target)}>Im Finder zeigen</button>
            <button className="primary" onClick={onClose}>Schliessen</button>
          </div>
        </>
      )}
    </Modal>
  );
}
