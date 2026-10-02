// "Wer ist das?" direkt im Shoot: unbenannte Gesichtsgruppen einmal benennen, gilt für alle Bilder der Gruppe.
import { useCallback, useEffect, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, Cluster, Person, waitForJob } from "../api";

export default function WhoPanel({ ctx, shootId, onChanged }: { ctx: AppCtx; shootId: number; onChanged: () => void }) {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [persons, setPersons] = useState<Person[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const inputs = useRef<Record<string, HTMLInputElement | null>>({});

  const load = useCallback(() => {
    api.get<Cluster[]>(`/api/shoots/${shootId}/clusters`).then((c) => setClusters(c.filter((x) => !x.person_id && x.cluster_id !== null)));
    api.get<Person[]>("/api/persons").then(setPersons);
  }, [shootId]);
  useEffect(load, [load, ctx.tick]);

  const body = (n: string) => {
    const p = persons.find((x) => x.name.toLowerCase() === n.trim().toLowerCase());
    return p ? { person_id: p.id } : { name: n.trim() };
  };
  const focusNext = (i: number) => setTimeout(() => {
    const next = clusters[i + 1];
    if (next) inputs.current[next.key]?.focus();
  }, 50);
  const assign = async (c: Cluster, i: number) => {
    const n = (names[c.key] ?? "").trim();
    if (!n || c.cluster_id === null) return;
    await api.post(`/api/shoots/${shootId}/clusters/${c.cluster_id}/assign`, body(n));
    setClusters((xs) => xs.filter((x) => x.key !== c.key));
    ctx.toast(`${c.count} Gesichter: ${n}`);
    focusNext(i - 1);
    onChanged();
  };
  const ignore = async (c: Cluster) => {
    if (c.cluster_id === null) return;
    await api.post(`/api/shoots/${shootId}/clusters/${c.cluster_id}/ignore`, {});
    setClusters((xs) => xs.filter((x) => x.key !== c.key));
  };
  const removeFace = async (c: Cluster, fid: number) => {
    await api.post(`/api/faces/${fid}/uncluster`, {});
    setClusters((xs) => xs.map((x) => (x.key === c.key ? { ...x, faces: x.faces.filter((f) => f.id !== fid), count: x.count - 1 } : x)));
  };
  const recognize = async () => {
    setBusy(true);
    try {
      const r = await api.post<{ job_id: number }>(`/api/shoots/${shootId}/run/people`, {});
      ctx.refreshJobs();
      await waitForJob(r.job_id);
      load(); onChanged();
      ctx.toast("Personen neu erkannt – benannte Personen werden jetzt auch in den übrigen Bildern gefunden");
    } finally { setBusy(false); }
  };

  return (
    <div className="who-panel">
      <div className="who-head">
        <div>
          <h2>Wer ist das?</h2>
          <p className="hint">Jede Gruppe ist eine Person. Name tippen, Enter – fertig für alle Bilder der Gruppe. Passt ein Gesicht nicht: ✕.
            Gegner oder Zuschauer: „Ignorieren“.</p>
        </div>
        <button disabled={busy} onClick={recognize} title="Mit den neu benannten Personen alle Bilder nochmals durchsuchen">
          {busy ? "erkenne …" : "Neu erkennen"}</button>
      </div>
      <datalist id="who-persons">{persons.map((p) => <option key={p.id} value={p.name} />)}</datalist>
      {clusters.length === 0 && <div className="empty">Alle Gruppen sind benannt. 🎉 Einzelne Gesichter benennst du im Bild (Doppelklick auf ein Bild).</div>}
      <div className="who-list">
        {clusters.map((c, i) => (
          <div key={c.key} className="who-card">
            <div className="faces">
              {c.faces.slice(0, 10).map((f) => (
                <div key={f.id} className="face-x">
                  <img loading="lazy" src={api.img(`/api/faces/${f.id}/crop`)} />
                  <button title="Gehört nicht dazu" onClick={() => removeFace(c, f.id)}>✕</button>
                </div>
              ))}
            </div>
            <div className="who-count">{c.count} Gesichter</div>
            <div className="row">
              <input ref={(el) => { inputs.current[c.key] = el; }} list="who-persons" placeholder="Name eingeben, Enter"
                value={names[c.key] ?? ""} autoFocus={i === 0}
                onChange={(e) => setNames({ ...names, [c.key]: e.target.value })}
                onKeyDown={(e) => { if (e.key === "Enter") assign(c, i); e.stopPropagation(); }} />
              <button className="ghost small" onClick={() => ignore(c)}>Ignorieren</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
