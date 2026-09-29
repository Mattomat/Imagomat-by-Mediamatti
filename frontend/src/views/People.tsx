import { useEffect, useMemo, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, Cluster, IS_APP, JobInfo, Overview, Person, pickFile, pickFolder, Shoot, waitForJob } from "../api";
import PersonDetail from "../components/PersonDetail";
import { Progress } from "../ui";

export default function PeopleView({ ctx, setDrop }: { ctx: AppCtx; setDrop: (h: ((p: string[]) => void) | null) => void }) {
  const [persons, setPersons] = useState<Person[]>([]);
  const [team, setTeam] = useState("");
  const [teamFilter, setTeamFilter] = useState("");
  const [learning, setLearning] = useState<JobInfo | null>(null);
  const [shoots, setShoots] = useState<Shoot[]>([]);
  const [shootId, setShootId] = useState<number | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [faceV, setFaceV] = useState(0);
  const [catalogs, setCatalogs] = useState<string[]>([]);
  const [open, setOpen] = useState<Person | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = () => {
    api.get<Person[]>("/api/persons").then(setPersons);
    setFaceV((v) => v + 1);
  };
  useEffect(load, [ctx.tick]);
  useEffect(() => {
    api.get<Overview>("/api/overview").then((o) => setCatalogs(o.catalogs));
  }, []);
  useEffect(() => {
    api.get<Shoot[]>("/api/shoots").then((s) => {
      setShoots(s);
      if (s.length && shootId === null) setShootId(s[0].id);
    });
  }, [ctx.tick]);
  useEffect(() => {
    if (shootId) api.get<Cluster[]>(`/api/shoots/${shootId}/clusters`).then((c) => setClusters(c.filter((x) => !x.person_id)));
  }, [shootId, ctx.tick]);

  const teams = useMemo(() => [...new Set(persons.map((p) => p.team).filter((t): t is string => !!t))].sort(), [persons]);
  const shownPersons = persons.filter((p) => !teamFilter || p.team === teamFilter);

  const importCsvText = async (text: string, fileName: string) => {
    const teamName = team.trim() || fileName.replace(/\.(csv|txt)$/i, "").replace(/[_-]+/g, " ");
    try {
      const r = await api.post<{ entries: unknown[] }>("/api/roster", { team: teamName, csv: text, save: true });
      ctx.toast(`${r.entries.length} Personen gespeichert (${teamName})`);
      load();
    } catch (e) {
      ctx.toast((e as Error).message, "error");
    }
  };

  const learnFrom = async (body: { folder?: string; files?: string[]; catalog?: string }) => {
    try {
      const r = await api.post<{ job_id: number }>("/api/people/learn", { ...body, team: team.trim() || undefined });
      ctx.refreshJobs();
      const j = await waitForJob(r.job_id, setLearning);
      setLearning(null);
      if (j.status === "done") ctx.toast(j.message ?? "Fertig");
      else ctx.toast(j.error?.split("\n")[0] ?? "Fehler beim Einlesen", "error");
      load();
    } catch (e) {
      setLearning(null);
      ctx.toast((e as Error).message, "error");
    }
  };

  useEffect(() => {
    setDrop(async (paths) => {
      const csv = paths.find((p) => /\.(csv|txt)$/i.test(p));
      if (csv) {
        const teamName = team.trim() || csv.split("/").pop()!.replace(/\.\w+$/, "").replace(/[_-]+/g, " ");
        try {
          const r = await api.post<{ entries: unknown[] }>("/api/roster", { team: teamName, path: csv, save: true });
          ctx.toast(`${r.entries.length} Personen gespeichert (${teamName})`);
          load();
        } catch (e) {
          ctx.toast((e as Error).message, "error");
        }
        return;
      }
      const images = paths.filter((p) => /\.(jpe?g|tiff?|dng|arw|cr3|nef|heic)$/i.test(p));
      if (images.length) learnFrom({ files: images });
      else if (paths[0]) learnFrom({ folder: paths[0] });
    });
    return () => setDrop(null);
  }, [setDrop, team]);

  const assignCluster = async (c: Cluster) => {
    const n = (names[c.key] ?? "").trim();
    if (!n || !shootId) return;
    const p = persons.find((x) => x.name.toLowerCase() === n.toLowerCase());
    const body = p ? { person_id: p.id } : { name: n, team: team.trim() || undefined };
    if (c.cluster_id !== null) await api.post(`/api/shoots/${shootId}/clusters/${c.cluster_id}/assign`, body);
    else for (const f of c.faces) await api.post(`/api/faces/${f.id}/assign`, body);
    setClusters((cs) => cs.filter((x) => x.key !== c.key));
    ctx.toast(`${n} zugeordnet`);
    load();
  };

  return (
    <div className="page people">
      <h1>Personen</h1>
      <div className="action-cards">
        <div className="action-card main">
          <div className="ac-title">Aus Lightroom übernehmen</div>
          <div className="ac-text">Alle Personen, die du in Lightroom schon benannt hast, auf einmal lernen.</div>
          {learning ? (
            <Progress value={learning.total ? learning.progress / learning.total : 0.05} label={learning.message ?? "Lerne Personen …"} />
          ) : (
            <>
              {catalogs.length > 0 && (
                <button className="primary" onClick={() => learnFrom({ catalog: catalogs[0] })} title={catalogs[0]}>
                  Übernehmen aus „{catalogs[0].split("/").pop()}“
                </button>
              )}
              <button className={catalogs.length ? "link" : "primary"} onClick={async () => {
                const f = await pickFile(["lrcat"], "Lightroom-Katalog wählen");
                if (f) learnFrom({ catalog: f });
              }}>{catalogs.length ? "anderen Katalog wählen" : "Katalog wählen …"}</button>
            </>
          )}
        </div>
        <div className="action-card">
          <div className="ac-title">Aus fertigen Bildern</div>
          <div className="ac-text">
            Einen ganzen Ordner mit fertigen JPGs (oder RAW + XMP) wählen{IS_APP ? " oder hierher ziehen" : ""}.
            Namen kommen aus Stichwörtern, Gesichtern oder dem Dateinamen, alle auf einmal.
          </div>
          <button disabled={!!learning} onClick={async () => { const f = await pickFolder("Ordner mit fertigen Bildern"); if (f) learnFrom({ folder: f }); }}>Ordner wählen …</button>
        </div>
        <div className="action-card">
          <div className="ac-title">Kader (CSV)</div>
          <div className="ac-text">Name und Rückennummer, z. B. <code>Max Muster;7</code>. Wird sofort gespeichert.</div>
          <input ref={fileRef} type="file" accept=".csv,.txt" hidden onChange={async (e) => {
            const f = e.target.files?.[0];
            if (f) await importCsvText(await f.text(), f.name);
            e.target.value = "";
          }} />
          <button onClick={() => fileRef.current?.click()}>CSV wählen …</button>
        </div>
      </div>
      <div className="team-row">
        <label>Team für neue Personen</label>
        <input list="teams" value={team} placeholder="z. B. FC Winterthur Frauen" onChange={(e) => setTeam(e.target.value)} />
        <datalist id="teams">{teams.map((t) => <option key={t} value={t} />)}</datalist>
      </div>

      {clusters.length > 0 && (
        <>
          <div className="section-head">
            <h2>Wer ist das?</h2>
            <select value={shootId ?? ""} onChange={(e) => setShootId(+e.target.value)}>
              {shoots.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
          <datalist id="persons">{persons.map((p) => <option key={p.id} value={p.name} />)}</datalist>
          <div className="who-list">
            {clusters.slice(0, 20).map((c) => (
              <div key={c.key} className="who-card">
                <div className="faces">{c.faces.slice(0, 5).map((f) => <img key={f.id} src={api.img(`/api/faces/${f.id}/crop`)} />)}</div>
                <div className="who-count">{c.count} Bilder</div>
                <input list="persons" placeholder="Name eingeben, Enter" value={names[c.key] ?? ""}
                  onChange={(e) => setNames({ ...names, [c.key]: e.target.value })}
                  onKeyDown={(e) => e.key === "Enter" && assignCluster(c)} />
              </div>
            ))}
          </div>
        </>
      )}

      <div className="section-head">
        <h2>Bekannte Personen ({persons.length})</h2>
        {teams.length > 1 && (
          <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)}>
            <option value="">Alle Teams</option>
            {teams.map((t) => <option key={t}>{t}</option>)}
          </select>
        )}
      </div>
      {persons.length === 0 && <div className="empty">Noch keine Personen. Importiere einen Kader oder ziehe benannte Bilder hierher.</div>}
      <div className="person-grid">
        {shownPersons.map((p) => (
          <div key={p.id} className="person-card" onClick={() => setOpen(p)} title="Bilder ansehen und bearbeiten">
            <div className="avatar-wrap">
              <div className="avatar">
                <img src={api.img(`/api/persons/${p.id}/face?v=${faceV}`)} onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />
                <span>{p.name.split(" ").map((x) => x[0]).join("").slice(0, 2)}</span>
              </div>
              {p.number && <span className="num-badge">{p.number}</span>}
            </div>
            <div className="p-name">{p.name}</div>
            <div className="p-team">{p.team ?? ""}</div>
            <button className="ghost small" title="Entfernen" onClick={(e) => {
              e.stopPropagation();
              if (confirm(`${p.name} wirklich entfernen?`)) api.del(`/api/persons/${p.id}`).then(load);
            }}>✕</button>
          </div>
        ))}
      </div>
      {open && (
        <PersonDetail person={open} teams={teams} toast={ctx.toast} onChanged={load} onClose={() => setOpen(null)} />
      )}
    </div>
  );
}
