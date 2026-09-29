import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, ImageItem, Person } from "../api";

type FaceRow = { id: number; bbox: number[]; person_id: number | null; name: string | null; assigned_by: string | null };
type Other = { person_id: number; name: string; via: string };

/** Personen im Bild: Gesichter benennen/korrigieren, Personen ohne Gesicht hinzufügen, Widersprüche entscheiden. */
export default function PeoplePanel({ ctx, image, onChanged }: { ctx: AppCtx; image: ImageItem; onChanged: () => void }) {
  const [faces, setFaces] = useState<FaceRow[]>([]);
  const [others, setOthers] = useState<Other[]>([]);
  const [persons, setPersons] = useState<Person[]>([]);
  const [edit, setEdit] = useState<number | "add" | null>(null);
  const [name, setName] = useState("");

  const load = () => api.get<{ faces: FaceRow[]; others: Other[] }>(`/api/images/${image.id}/faces`).then((r) => {
    setFaces(r.faces);
    setOthers(r.others);
  });
  useEffect(() => {
    load();
    setEdit(null);
  }, [image.id]);
  useEffect(() => {
    api.get<Person[]>("/api/persons").then(setPersons);
  }, []);

  const body = (n: string) => {
    const p = persons.find((x) => x.name.toLowerCase() === n.trim().toLowerCase());
    return p ? { person_id: p.id } : { name: n.trim() };
  };
  const save = async () => {
    if (!name.trim() || edit === null) return;
    if (edit === "add") await api.post(`/api/images/${image.id}/persons`, body(name));
    else await api.post(`/api/faces/${edit}/assign`, body(name));
    setEdit(null);
    setName("");
    await load();
    onChanged();
  };
  const remove = async (pid: number) => {
    await api.del(`/api/images/${image.id}/persons/${pid}`);
    await load();
    onChanged();
  };
  const decide = async (faceId: number, personId: number, who: string | null) => {
    await api.post(`/api/images/${image.id}/people-check`, { face_id: faceId, person_id: personId });
    ctx.toast(`${who ?? "Person"} festgelegt`);
    await load();
    onChanged();
  };

  return (
    <div className="people-panel">
      {(image.people_check ?? []).map((c) => (
        <div key={c.face_id} className="conflict">
          <span>Gesicht sagt <b>{c.face_person}</b>, Trikot ({c.shirt.map((k) => (k === "name" ? "Name" : "Nummer")).join(" + ")}) sagt <b>{c.shirt_person}</b>.</span>
          <button className={c.chosen === "face" ? "on" : ""} onClick={() => decide(c.face_id, c.face_person_id, c.face_person)}>{c.face_person}</button>
          <button className={c.chosen === "shirt" ? "on" : ""} onClick={() => decide(c.face_id, c.shirt_person_id, c.shirt_person)}>{c.shirt_person}</button>
        </div>
      ))}
      <datalist id="pp-persons">{persons.map((p) => <option key={p.id} value={p.name} />)}</datalist>
      <div className="pp-row">
        {faces.map((f) => (
          <div key={f.id} className={`pp-face ${edit === f.id ? "sel" : ""}`} onClick={() => { setEdit(f.id); setName(f.name ?? ""); }}
            title="Klicken zum Benennen">
            <img src={api.img(`/api/faces/${f.id}/crop`)} />
            <span className={f.name ? "" : "unknown"}>{f.name ?? "Wer?"}</span>
          </div>
        ))}
        {others.map((o) => (
          <div key={`o${o.person_id}`} className="pp-chip" title={o.via === "#manuell" ? "manuell hinzugefügt" : "über Trikot erkannt"}>
            {o.name}{o.via !== "#manuell" ? " (Trikot)" : ""}
            <button onClick={() => remove(o.person_id)}>✕</button>
          </div>
        ))}
        <button className="ghost small" onClick={() => { setEdit("add"); setName(""); }}>+ Person</button>
      </div>
      {edit !== null && (
        <div className="pp-edit">
          <input autoFocus list="pp-persons" value={name} placeholder={edit === "add" ? "Wer ist noch im Bild?" : "Name, Enter"}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") setEdit(null); }} />
          <button className="primary small" onClick={save}>OK</button>
          {typeof edit === "number" && faces.find((f) => f.id === edit)?.person_id && (
            <button className="ghost small" onClick={() => remove(faces.find((f) => f.id === edit)!.person_id!)}>Entfernen</button>
          )}
          <button className="ghost small" onClick={() => setEdit(null)}>Abbrechen</button>
        </div>
      )}
    </div>
  );
}
