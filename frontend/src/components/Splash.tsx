import { useEffect, useState } from "react";
import { BASE } from "../api";

type SetupEvent = { stage: string; message: string; progress: number; done: boolean; error: boolean };

/** Zeigt die einmalige Einrichtung bzw. den Start des Backends, bis die API antwortet. */
export default function Splash({ onReady }: { onReady: () => void }) {
  const [evt, setEvt] = useState<SetupEvent>({ stage: "Starte", message: "", progress: 0, done: false, error: false });
  const [waited, setWaited] = useState(0);

  useEffect(() => {
    let stop = false;
    let unlisten: (() => void) | undefined;
    if (window.__TAURI_INTERNALS__) {
      import("@tauri-apps/api/event").then(({ listen }) =>
        listen<SetupEvent>("setup", (e) => setEvt(e.payload)).then((u) => (unlisten = u)),
      );
      import("@tauri-apps/api/core").then(({ invoke }) =>
        invoke<SetupEvent>("setup_status").then((e) => e.stage && setEvt(e)).catch(() => undefined),
      );
    }
    const poll = async () => {
      while (!stop) {
        try {
          const r = await fetch(BASE + "/api/health");
          if (r.ok) {
            onReady();
            return;
          }
        } catch {
          /* Backend startet noch */
        }
        setWaited((w) => w + 1);
        await new Promise((res) => setTimeout(res, 1000));
      }
    };
    poll();
    return () => {
      stop = true;
      unlisten?.();
    };
  }, [onReady]);

  const firstRun = evt.stage !== "Starte" && evt.stage !== "bereit";
  return (
    <div className="splash">
      <div className="splash-box">
        <div className="logo">Tagmatti</div>
        {evt.error ? (
          <>
            <h2>Einrichtung fehlgeschlagen</h2>
            <p className="error">{evt.message}</p>
            <p className="hint">Details: ~/Library/Application Support/Tagmatti/logs</p>
          </>
        ) : firstRun ? (
          <>
            <h2>Einmalige Einrichtung</h2>
            <p>Tagmatti lädt beim ersten Start seine KI-Bausteine (ca. 5–10 Minuten, braucht Internet). Danach startet die App in Sekunden.</p>
            <div className="bar big"><div style={{ width: `${Math.round(evt.progress * 100)}%` }} /></div>
            <p className="stage">{evt.stage}</p>
            <p className="hint mono">{evt.message}</p>
          </>
        ) : (
          <>
            <p>Wird gestartet …</p>
            {waited > 20 && !window.__TAURI_INTERNALS__ && (
              <p className="hint">Läuft der Server? Im Terminal: <code>tagmatti serve</code></p>
            )}
            {waited > 60 && window.__TAURI_INTERNALS__ && (
              <p className="hint">Dauert ungewöhnlich lange. Protokoll: ~/Library/Application Support/Tagmatti/logs/backend.log</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
