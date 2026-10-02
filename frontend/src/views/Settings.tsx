import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";
import LogPanel from "../components/LogPanel";
import { Modal, More, Segmented } from "../ui";

type S = {
  keywords: { people_root: string; write_face_regions: boolean; person_keyword_style: string };
  shirt_color_check?: boolean;
  device: string;
  face_backend: string;
  ignored_shirt_words: string[];
};

export default function SettingsView({ ctx }: { ctx: AppCtx }) {
  const [s, setS] = useState<S | null>(null);
  const [resetting, setResetting] = useState<"no" | "ask" | "busy">("no");
  const [licenses, setLicenses] = useState<{ name: string; license: string; commercial_ok: boolean }[]>([]);

  useEffect(() => {
    api.get<S>("/api/settings").then(setS);
    api.get<typeof licenses>("/api/licenses").then(setLicenses);
  }, []);
  if (!s) return null;

  const save = async (next: S) => {
    setS(next);
    await api.put("/api/settings", next);
    ctx.toast("Gespeichert");
  };
  const set = <K extends keyof S>(k: K, v: Partial<S[K]>) => save({ ...s, [k]: { ...(s[k] as object), ...v } } as S);

  return (
    <div className="page settings">
      <h1>Einstellungen</h1>
      <div className="card">
        <div className="setting">
          <div>
            <b>Namen als Stichwort</b>
            <div className="hint">nur der Name (z. B. „Lena Meier“) oder mit Team darüber (Personen › Team › Name)</div>
          </div>
          <Segmented value={s.keywords.person_keyword_style} onChange={(v) => set("keywords", { person_keyword_style: v })}
            options={[["name", "nur Name"], ["team", "mit Team"]]} />
        </div>
        <div className="setting">
          <div><b>Gesichtsbereiche speichern</b><div className="hint">welcher Name zu welchem Gesicht gehört (Lightroom „Personen“)</div></div>
          <input type="checkbox" className="switch" checked={s.keywords.write_face_regions} onChange={(e) => set("keywords", { write_face_regions: e.target.checked })} />
        </div>
        <div className="setting">
          <div><b>Rückennummern nur auf rot/weiss/schwarzen Trikots</b><div className="hint">für Vereine in diesen Farben: Nummern von Gegnern zählen dann nicht</div></div>
          <input type="checkbox" className="switch" checked={!!s.shirt_color_check} onChange={(e) => save({ ...s, shirt_color_check: e.target.checked })} />
        </div>
      </div>

      <LogPanel toast={ctx.toast} />

      <div className="card">
        <div className="setting">
          <div>
            <b>Neu anfangen (Personen behalten)</b>
            <div className="hint">löscht alle Shoots und Bilder – deine Personen und Stichwörter bleiben</div>
          </div>
          <button className="danger" onClick={() => setResetting("ask")}>Zurücksetzen …</button>
        </div>
      </div>
      {resetting !== "no" && (
        <Modal title="Alles ausser Personen löschen?" onClose={() => resetting === "ask" && setResetting("no")}>
          <p><b>Gelöscht wird:</b> alle Shoots mit Vorschauen,
            unbenannte Gesichter („Wer ist das?“) und alle Zwischenspeicher.</p>
          <p><b>Bleibt:</b> deine Personen mit Name, Nummer und Team und die Gesichter, die du ihnen zugeordnet hast
            (damit die Erkennung weiter funktioniert).</p>
          <p className="hint">Deine Originalbilder und exportierten XMPs werden nie angerührt.</p>
          <div className="modal-actions">
            <button className="ghost" disabled={resetting === "busy"} onClick={() => setResetting("no")}>Abbrechen</button>
            <button className="danger" disabled={resetting === "busy"} onClick={async () => {
              setResetting("busy");
              try {
                const r = await api.post<{ shoots: number; images: number; styles: number; persons: number }>(
                  "/api/reset", { confirm: "personen-behalten" });
                ctx.toast(`Gelöscht: ${r.shoots} Shoots, ${r.styles} Stile. Behalten: ${r.persons} Personen.`);
                ctx.refreshJobs();
              } catch (e) {
                ctx.toast((e as Error).message, "error");
              }
              setResetting("no");
            }}>{resetting === "busy" ? "Lösche …" : "Ja, alles ausser Personen löschen"}</button>
          </div>
        </Modal>
      )}

      <More label="Erweitert">
        <div className="card">
          <div className="setting">
            <div><b>Stichwort für Personen</b></div>
            <input className="narrow" value={s.keywords.people_root} onChange={(e) => setS({ ...s, keywords: { ...s.keywords, people_root: e.target.value } })}
              onBlur={() => save(s)} />
          </div>
          <div className="setting">
            <div><b>Sponsoren auf Trikots ignorieren</b><div className="hint">diese Wörter sind nie ein Spielername (weitere erkennt Tagmatti selbst)</div></div>
            <input className="narrow" value={(s.ignored_shirt_words ?? []).join(", ")}
              onChange={(e) => setS({ ...s, ignored_shirt_words: e.target.value.split(",").map((w) => w.trim()).filter(Boolean) })}
              onBlur={() => save(s)} placeholder="z. B. Sponsorname" />
          </div>
          <div className="setting">
            <div><b>Gesichtserkennung</b><div className="hint">InsightFace ist am genauesten, aber nur privat nutzbar</div></div>
            <select className="narrow" value={s.face_backend} onChange={(e) => save({ ...s, face_backend: e.target.value })}>
              <option value="auto">Automatisch</option>
              <option value="insightface">InsightFace</option>
              <option value="yunet">YuNet (kommerziell ok)</option>
            </select>
          </div>
        </div>
        <h3>Lizenzen der KI-Modelle</h3>
        <table className="list">
          <tbody>
            {licenses.map((l) => <tr key={l.name}><td>{l.commercial_ok ? "✓" : "⚠︎"}</td><td>{l.name}</td><td>{l.license}</td></tr>)}
          </tbody>
        </table>
      </More>
    </div>
  );
}
