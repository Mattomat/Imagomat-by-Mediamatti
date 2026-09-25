import { useState } from "react";
import { api, pickFile, pickFolder, Shoot } from "../api";

export default function ExportView({ shoot, onJob }: { shoot: Shoot | null; onJob: () => void }) {
  const [f, setF] = useState({
    target: "", xmp: true, catalog: false, jpeg: false, copy_mode: "copy", include_rejected: true, template: "", side: 2048,
  });
  const [msg, setMsg] = useState("");
  if (!shoot) return <div className="page"><p>Kein Shoot gewählt.</p></div>;

  const go = async () => {
    const formats = [f.xmp && "xmp", f.catalog && "catalog", f.jpeg && "jpeg"].filter(Boolean);
    try {
      await api.post(`/api/shoots/${shoot.id}/export`, {
        target: f.target, formats, copy_mode: f.copy_mode, include_rejected: f.include_rejected,
        template: f.template || null, jpeg_side: f.side,
      });
      setMsg("Export gestartet.");
      onJob();
    } catch (e) {
      setMsg(String(e));
    }
  };

  return (
    <div className="page two-col">
      <section className="card">
        <h2>Export: {shoot.name}</h2>
        <label>Zielordner</label>
        <div className="row">
          <input value={f.target} onChange={(e) => setF({ ...f, target: e.target.value })} />
          <button onClick={async () => { const d = await pickFolder(f.target); if (d) setF({ ...f, target: d }); }}>Wählen…</button>
        </div>
        <label className="check"><input type="checkbox" checked={f.xmp} onChange={(e) => setF({ ...f, xmp: e.target.checked })} /> A: RAW + XMP (empfohlen)</label>
        <label className="check"><input type="checkbox" checked={f.catalog} onChange={(e) => setF({ ...f, catalog: e.target.checked })} /> B: Lightroom-Katalog (experimentell)</label>
        {f.catalog && (
          <>
            <label>Leerer Vorlagen-Katalog (in Lightroom: Datei › Neuer Katalog, dann Lightroom schliessen)</label>
            <div className="row">
              <input value={f.template} onChange={(e) => setF({ ...f, template: e.target.value })} />
              <button onClick={async () => { const p = await pickFile(["lrcat"]); if (p) setF({ ...f, template: p }); }}>Wählen…</button>
            </div>
          </>
        )}
        <label className="check"><input type="checkbox" checked={f.jpeg} onChange={(e) => setF({ ...f, jpeg: e.target.checked })} /> C: Galerie-JPEGs (Annäherung, nicht Lightroom-Qualität)</label>
        <label>Originale</label>
        <select value={f.copy_mode} onChange={(e) => setF({ ...f, copy_mode: e.target.value })}>
          <option value="copy">kopieren</option>
          <option value="hardlink">Hardlink (spart Platz, gleicher Datenträger)</option>
          <option value="inplace">nur XMP neben die Original-RAWs schreiben</option>
        </select>
        <label className="check"><input type="checkbox" checked={f.include_rejected} onChange={(e) => setF({ ...f, include_rejected: e.target.checked })} /> Aussortierte mit exportieren (1 Stern + Grund als Stichwort)</label>
        <button className="primary" disabled={!f.target || (!f.xmp && !f.catalog && !f.jpeg)} onClick={go}>Exportieren</button>
        {msg && <p className="hint">{msg}</p>}
      </section>
      <section className="card">
        <h2>Danach in Lightroom Classic</h2>
        <ol className="steps">
          <li>Ordner importieren (<i>Hinzufügen</i>) bzw. den exportierten Katalog öffnen.</li>
          <li>Alle Bilder im Raster auswählen › <b>Foto › Entwicklungseinstellungen › KI-Einstellungen aktualisieren</b>. Damit berechnet Lightroom die KI-Masken und Denoise.</li>
          <li>Optional das Plugin <i>Imagomat.lrplugin</i>: setzt Picks/Ablehnungen und legt Sammlungen an (Personen, Gründe, Denoise, Prüfen).</li>
          <li>Unsichere Bilder: Stichwort <i>Imagomat › Prüfen</i> bzw. gelbes Label.</li>
          <li>Korrigierst du Bilder und speicherst die Metadaten (Cmd+S), lernt das Profil daraus über „Korrekturen lernen“.</li>
        </ol>
        <p className="hint">Export-Log: <code>imagomat-log.csv</code> im Zielordner (Entscheidung, Gründe, Sterne, Personen, Denoise).</p>
      </section>
    </div>
  );
}
