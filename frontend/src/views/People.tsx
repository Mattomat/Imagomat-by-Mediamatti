import { useEffect, useState } from "react";
import { api, Cluster, Person, Shoot } from "../api";

export default function PeopleView({ shoot, tick, onJob }: { shoot: Shoot | null; tick: number; onJob: () => void }) {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [persons, setPersons] = useState<Person[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [roster, setRoster] = useState({ team: "FC Winterthur 1. Mannschaft", url: "", csv: "" });
  const [preview, setPreview] = useState<{ name: string; number: string | null }[]>([]);
  const [sources, setSources] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState("");

  const load = () => {
    api.get<Person[]>("/api/persons").then(setPersons);
    if (shoot) api.get<Cluster[]>(`/api/shoots/${shoot.id}/clusters`).then(setClusters);
  };
  useEffect(load, [shoot, tick]);
  useEffect(() => {
    api.get<Record<string, string>>("/api/roster/sources").then(setSources);
  }, []);

  const assign = async (c: Cluster) => {
    const n = (names[c.key] ?? "").trim();
    if (!n || !shoot) return;
    const p = persons.find((x) => x.name === n);
    const body = p ? { person_id: p.id } : { name: n };
    if (c.cluster_id !== null && !c.person_id) await api.post(`/api/shoots/${shoot.id}/clusters/${c.cluster_id}/assign`, body);
    else for (const f of c.faces) await api.post(`/api/faces/${f.id}/assign`, body);
    load();
  };

  const loadRoster = async (save: boolean) => {
    setMsg("");
    try {
      const r = await api.post<{ entries: { name: string; number: string | null }[] }>("/api/roster", {
        team: roster.team, url: roster.url || undefined, csv: roster.csv || undefined, save,
      });
      setPreview(r.entries);
      setMsg(save ? `${r.entries.length} Personen gespeichert` : `${r.entries.length} Einträge gefunden, bitte prüfen`);
      if (save) load();
    } catch (e) {
      setMsg(String(e));
    }
  };

  return (
    <div className="page two-col">
      <section className="card wide">
        <h2>Gesichter im Shoot</h2>
        {!shoot && <p>Kein Shoot gewählt.</p>}
        {shoot && (
          <p className="hint">
            Unbekannte Gesichter sind nach Ähnlichkeit gruppiert. Gib einer Gruppe einen Namen: ab dann erkennt Imagomat
            die Person auch in künftigen Shoots.{" "}
            <button onClick={() => api.post(`/api/shoots/${shoot.id}/run/people`).then(onJob)}>Neu zuordnen</button>
          </p>
        )}
        <datalist id="persons">{persons.map((p) => <option key={p.id} value={p.name} />)}</datalist>
        {clusters.map((c) => (
          <div key={c.key} className="cluster">
            <div className="faces">
              {c.faces.map((f) => (
                <img key={f.id} src={api.img(`/api/faces/${f.id}/crop`)} title={f.assigned_by ?? "unbestätigt"} />
              ))}
            </div>
            <div className="assign">
              <b>{c.name ?? (c.cluster_id !== null ? `Gruppe ${c.cluster_id + 1}` : "Einzelgesichter")}</b> · {c.count}×
              <input list="persons" placeholder="Name" value={names[c.key] ?? ""} onChange={(e) => setNames({ ...names, [c.key]: e.target.value })} />
              <button onClick={() => assign(c)}>Zuordnen</button>
            </div>
          </div>
        ))}
      </section>
      <section className="card">
        <h2>Kader & Rückennummern</h2>
        <label>Team</label>
        <input list="teams" value={roster.team} onChange={(e) => setRoster({ ...roster, team: e.target.value })} />
        <datalist id="teams">{Object.keys(sources).map((t) => <option key={t} value={t} />)}</datalist>
        <label>Webseite (optional)</label>
        <input value={roster.url} placeholder={sources[roster.team] ?? "https://…"} onChange={(e) => setRoster({ ...roster, url: e.target.value })} />
        <label>oder CSV (Name;Nummer)</label>
        <textarea rows={5} value={roster.csv} onChange={(e) => setRoster({ ...roster, csv: e.target.value })} placeholder={"Max Muster;7\nLuca Beispiel;10"} />
        <div className="row">
          <button onClick={() => loadRoster(false)}>Vorschau</button>
          <button className="primary" onClick={() => loadRoster(true)}>Speichern</button>
        </div>
        {msg && <p className="hint">{msg}</p>}
        {preview.length > 0 && (
          <table className="list"><tbody>{preview.map((e) => <tr key={e.name}><td>{e.number}</td><td>{e.name}</td></tr>)}</tbody></table>
        )}
        <h3>Personen ({persons.length})</h3>
        <table className="list">
          <tbody>
            {persons.map((p) => (
              <tr key={p.id}>
                <td>{p.number}</td><td>{p.name}</td><td>{p.team}</td>
                <td><button onClick={() => api.del(`/api/persons/${p.id}`).then(load)}>✕</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
