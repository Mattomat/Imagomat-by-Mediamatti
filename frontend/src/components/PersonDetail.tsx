import { useEffect, useState } from "react";
import { api, Person } from "../api";

type PImage = {
  image_id: number;
  filename: string;
  capture_time: number | null;
  shoot: string;
  face_id: number | null;
  bbox: number[] | null;
  via: "face" | "number";
  reference: boolean;
};

/** Alle Bilder einer Person + Name, Nummer und Team bearbeiten. */
export default function PersonDetail({ person, teams, onClose, onChanged, toast }: {
  person: Person;
  teams: string[];
  onClose: () => void;
  onChanged: () => void;
  toast: (m: string, k?: "ok" | "error") => void;
}) {
  const [images, setImages] = useState<PImage[] | null>(null);
  const [name, setName] = useState(person.name);
  const [number, setNumber] = useState(person.number ?? "");
  const [team, setTeam] = useState(person.team ?? "");
  const [big, setBig] = useState<PImage | null>(null);
  const [pid, setPid] = useState(person.id);

  const load = (id = pid) => api.get<{ images: PImage[] }>(`/api/persons/${id}/images`).then((r) => setImages(r.images));
  useEffect(() => {
    load();
  }, []);

  const dirty = name.trim() !== person.name || number.trim() !== (person.number ?? "") || team.trim() !== (person.team ?? "");
  const save = async () => {
    try {
      const r = await api.patch<Person & { merged: boolean }>(`/api/persons/${pid}`, {
        name: name.trim(), number: number.trim(), team: team.trim(),
      });
      toast(r.merged ? `Mit „${r.name}“ zusammengeführt` : "Gespeichert");
      setPid(r.id);
      load(r.id);
      onChanged();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const unassign = async (img: PImage) => {
    // Person aus dem Bild entfernen; Tagmatti merkt sich das und ordnet das Bild nicht wieder falsch zu
    await api.del(`/api/images/${img.image_id}/persons/${pid}`);
    setImages((xs) => xs?.filter((x) => x.image_id !== img.image_id) ?? null);
    setBig(null);
    onChanged();
  };

  const shoots = images ? [...new Set(images.map((i) => i.shoot))] : [];

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal person-detail" onClick={(e) => e.stopPropagation()}>
        <div className="pd-head">
          <div className="avatar big">
            <img src={api.img(`/api/persons/${pid}/face`)} onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />
            {number && <span className="num-badge">{number}</span>}
          </div>
          <div className="pd-fields">
            <input className="pd-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
            <div className="row">
              <input className="narrow" value={number} onChange={(e) => setNumber(e.target.value.replace(/\D/g, ""))} placeholder="Nr." />
              <input list="pd-teams" value={team} onChange={(e) => setTeam(e.target.value)} placeholder="Team (z. B. Herren 1)" />
              <datalist id="pd-teams">{teams.map((t) => <option key={t} value={t} />)}</datalist>
              <button className="primary" disabled={!dirty || !name.trim()} onClick={save}>Speichern</button>
            </div>
            <div className="hint">
              {images ? `${images.length} Bilder${shoots.length ? ` · ${shoots.length} Shoot${shoots.length > 1 ? "s" : ""}` : ""}` : "lädt …"}
              {" · "}Falsches Bild? Mit ✕ entfernen, Tagmatti lernt daraus.
            </div>
          </div>
          <button className="ghost" onClick={onClose}>✕</button>
        </div>

        {images && images.length === 0 && <div className="empty">Noch keine Bilder mit dieser Person.</div>}
        <div className="pd-grid">
          {images?.map((img) => (
            <div key={img.image_id} className="pd-thumb" onClick={() => setBig(img)} title={`${img.filename} · ${img.shoot}`}>
              <img loading="lazy" src={api.img(img.face_id ? `/api/faces/${img.face_id}/crop` : `/api/images/${img.image_id}/preview`)} />
              {img.via === "number" && <span className="pd-via">#</span>}
              <button className="pd-del" title={`Das ist nicht ${name.split(" ")[0]}`}
                onClick={(e) => { e.stopPropagation(); unassign(img); }}>✕</button>
            </div>
          ))}
        </div>

        {big && (
          <div className="pd-big" onClick={() => setBig(null)}>
            <div className="pd-big-img">
              <img src={api.img(`/api/images/${big.image_id}/preview`)} />
              {big.bbox && (
                <div className="pd-box" style={{
                  left: `${big.bbox[0] * 100}%`, top: `${big.bbox[1] * 100}%`,
                  width: `${(big.bbox[2] - big.bbox[0]) * 100}%`, height: `${(big.bbox[3] - big.bbox[1]) * 100}%`,
                }} />
              )}
            </div>
            <div className="pd-big-bar" onClick={(e) => e.stopPropagation()}>
              <span>{big.filename} · {big.shoot}{big.capture_time ? ` · ${new Date(big.capture_time * 1000).toLocaleDateString()}` : ""}</span>
              <span className="spacer" />
              <button onClick={() => unassign(big)}>Das ist nicht {name.split(" ")[0]}</button>
              <button className="ghost" onClick={() => setBig(null)}>Zurück</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
