//! Tauri-Hülle für Imagomat.
//!
//! Beim ersten Start (bzw. nach einem App-Update) richtet die App ihre Python-Laufzeit ein:
//! Der mitgelieferte Python-Code wird nach ~/Library/Application Support/Imagomat/runtime
//! kopiert, `uv` (liegt im App-Bundle) erstellt dort eine Python-Umgebung und installiert
//! die Pakete. Danach startet das Backend auf 127.0.0.1:8765.
//! Fortschritt geht als Event "setup" an die Oberfläche.
//!
//! Entwicklung: IMAGOMAT_DEV_BACKEND=1 -> Backend separat starten (`imagomat serve`).

use serde::Serialize;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

/// KI-Pakete: einzeln installiert, damit ein fehlgeschlagenes Paket nicht alles blockiert.
const ML_PACKAGES: &[&str] = &[
    "torch>=2.3",
    "torchvision>=0.18",
    "open_clip_torch>=2.24",
    "transformers>=4.44",
    "timm>=1.0",
    "einops>=0.8",
    "kornia>=0.7",
    "onnxruntime>=1.18",
    "mediapipe>=0.10.14",
    "ocrmac>=1.0",
    "insightface>=0.7.3",
];

#[derive(Clone, Serialize, Default)]
struct SetupEvent {
    stage: String,
    message: String,
    progress: f32,
    done: bool,
    error: bool,
}

#[derive(Default)]
struct AppState {
    backend: Mutex<Option<Child>>,
    last: Mutex<SetupEvent>,
}

fn emit(app: &AppHandle, stage: &str, message: &str, progress: f32, done: bool, error: bool) {
    let evt = SetupEvent {
        stage: stage.into(),
        message: message.chars().take(300).collect(),
        progress,
        done,
        error,
    };
    if let Some(state) = app.try_state::<AppState>() {
        *state.last.lock().unwrap() = evt.clone();
    }
    let _ = app.emit("setup", evt);
}

#[tauri::command]
fn setup_status(state: tauri::State<AppState>) -> SetupEvent {
    state.last.lock().unwrap().clone()
}

fn data_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("Library/Application Support/Imagomat")
}

/// PATH für GUI-Apps um Homebrew ergänzen (z. B. für ExifTool).
fn gui_path() -> String {
    let base = std::env::var("PATH").unwrap_or_default();
    format!("/opt/homebrew/bin:/usr/local/bin:{base}")
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "__pycache__" || name == ".venv" || name == "tests" {
            continue;
        }
        let target = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn run_logged(app: &AppHandle, stage: &str, progress: f32, cmd: &mut Command) -> Result<(), String> {
    cmd.env("PATH", gui_path()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("{stage}: {e}"))?;
    let stderr = child.stderr.take();
    let app2 = app.clone();
    let stage2 = stage.to_string();
    let reader = std::thread::spawn(move || {
        if let Some(err) = stderr {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                emit(&app2, &stage2, &line, progress, false, false);
            }
        }
    });
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            emit(app, stage, &line, progress, false, false);
        }
    }
    let _ = reader.join();
    let status = child.wait().map_err(|e| format!("{stage}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{stage} fehlgeschlagen ({status})"))
    }
}

fn bootstrap(app: &AppHandle) -> Result<PathBuf, String> {
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let uv = res.join("bin").join("uv");
    let runtime = data_root().join("runtime");
    let venv = runtime.join("venv");
    let python = venv.join("bin").join("python");
    let version = app.package_info().version.to_string();
    let marker = runtime.join("version.txt");
    let installed = fs::read_to_string(&marker).unwrap_or_default();
    if installed.trim() == version && python.exists() {
        return Ok(python);
    }
    fs::create_dir_all(&runtime).map_err(|e| e.to_string())?;
    emit(app, "Vorbereiten", "Programmdateien kopieren", 0.02, false, false);
    let src = runtime.join("src");
    let _ = fs::remove_dir_all(&src);
    copy_dir(&res.join("backend"), &src).map_err(|e| format!("Kopieren: {e}"))?;
    if !python.exists() {
        run_logged(
            app,
            "Python einrichten",
            0.08,
            Command::new(&uv).args(["venv", "--python", "3.12"]).arg(&venv),
        )?;
    }
    run_logged(
        app,
        "Grundpakete installieren",
        0.2,
        Command::new(&uv)
            .args(["pip", "install", "--python"])
            .arg(&python)
            .arg(format!("{}[scrape]", src.display())),
    )?;
    let n = ML_PACKAGES.len() as f32;
    for (i, pkg) in ML_PACKAGES.iter().enumerate() {
        let p = 0.3 + 0.65 * (i as f32) / n;
        let stage = format!("KI-Pakete ({}/{})", i + 1, ML_PACKAGES.len());
        if let Err(e) = run_logged(
            app,
            &stage,
            p,
            Command::new(&uv).args(["pip", "install", "--python"]).arg(&python).arg(pkg),
        ) {
            // Nicht fatal: die App nutzt dann Ersatzverfahren
            emit(app, &stage, &format!("übersprungen: {e}"), p, false, false);
        }
    }
    fs::write(&marker, &version).map_err(|e| e.to_string())?;
    Ok(python)
}

fn start_backend(python: &Path) -> Result<Child, String> {
    let logs = data_root().join("logs");
    fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
    let log = fs::File::create(logs.join("backend.log")).map_err(|e| e.to_string())?;
    let log2 = log.try_clone().map_err(|e| e.to_string())?;
    Command::new(python)
        .args(["-m", "imagomat.cli", "serve", "--port", "8765"])
        .env("PATH", gui_path())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log2))
        .spawn()
        .map_err(|e| format!("Backend-Start: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![setup_status])
        .setup(|app| {
            let handle = app.handle().clone();
            if std::env::var("IMAGOMAT_DEV_BACKEND").is_ok() {
                emit(&handle, "bereit", "Entwicklungsmodus", 1.0, true, false);
                return Ok(());
            }
            std::thread::spawn(move || {
                match bootstrap(&handle).and_then(|py| start_backend(&py)) {
                    Ok(child) => {
                        *handle.state::<AppState>().backend.lock().unwrap() = Some(child);
                        emit(&handle, "bereit", "Backend läuft", 1.0, true, false);
                    }
                    Err(e) => emit(&handle, "Fehler", &e, 1.0, true, true),
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Fehler beim Starten von Imagomat");
    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<AppState>().backend.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
