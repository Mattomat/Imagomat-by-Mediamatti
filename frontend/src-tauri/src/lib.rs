//! Tauri-Hülle für Tagmatti.
//!
//! Beim ersten Start (bzw. nach einem App-Update) richtet die App ihre Python-Laufzeit ein:
//! Der mitgelieferte Python-Code wird in den Datenordner kopiert (macOS: ~/Library/Application
//! Support/Tagmatti/runtime, Windows: %APPDATA%\\Tagmatti\\runtime), `uv` (liegt im App-Bundle) erstellt dort eine Python-Umgebung und installiert
//! die Pakete. Danach startet das Backend auf 127.0.0.1:8765.
//! Fortschritt geht als Event "setup" an die Oberfläche.
//!
//! Entwicklung: TAGMATTI_DEV_BACKEND=1 -> Backend separat starten (`tagmatti serve`).

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
    "ocrmac>=1.0",
    "pyobjc-framework-Quartz>=10.0",
    "insightface>=0.7.3",
];
/// Nur auf dem Mac sinnvoll (Apple Vision, Core Image).
const MAC_ONLY: &[&str] = &["ocrmac", "pyobjc"];

/// Befehl ohne Konsolenfenster (Windows öffnet sonst für jeden Aufruf ein schwarzes Fenster).
fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

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
    log_setup(&format!("[{stage}] {message}"));
    let _ = app.emit("setup", evt);
}

/// Einrichtungsprotokoll in <Datenordner>/logs/setup.log
fn log_setup(line: &str) {
    use std::io::Write;
    let dir = data_root().join("logs");
    if fs::create_dir_all(&dir).is_ok() {
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(dir.join("setup.log")) {
            let _ = writeln!(f, "{line}");
        }
    }
}

#[tauri::command]
fn setup_status(state: tauri::State<AppState>) -> SetupEvent {
    state.last.lock().unwrap().clone()
}

/// Gleicher Ordner wie im Backend (tagmatti.config.data_dir).
fn data_root() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("APPDATA").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into());
        return PathBuf::from(base).join("Tagmatti");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("Library/Application Support/Tagmatti")
}

/// Die App hiess früher "Imagomat": vorhandene Daten (Personen, Shoots, Einstellungen) einmalig übernehmen.
/// Die Python-Umgebung wird danach neu eingerichtet (sie kennt noch die alten Pfade).
fn migrate_old_name() {
    let new = data_root();
    let old = new.with_file_name("Imagomat");
    if new.exists() || !old.is_dir() {
        return;
    }
    if fs::rename(&old, &new).is_ok() {
        let _ = fs::remove_file(new.join("runtime").join("version.txt"));
        log_setup("[Vorbereiten] Daten von Imagomat übernommen");
    }
}

/// Python der App-Umgebung (Windows: Scripts\\python.exe).
fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// PATH für GUI-Apps um Homebrew ergänzen (z. B. für ExifTool).
fn gui_path() -> String {
    let base = std::env::var("PATH").unwrap_or_default();
    if cfg!(windows) {
        return base;
    }
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

/// uv aus dem App-Bundle in den Laufzeitordner kopieren, ausführbar machen und die
/// macOS-Quarantäne-Markierung entfernen (sonst kann Gatekeeper den Start blockieren).
fn prepare_uv(res: &Path, runtime: &Path) -> Result<PathBuf, String> {
    let src = res.join("bin").join("uv");
    let dst = runtime.join("bin").join(if cfg!(windows) { "uv.exe" } else { "uv" });
    fs::create_dir_all(runtime.join("bin")).map_err(|e| e.to_string())?;
    fs::copy(&src, &dst).map_err(|e| format!("uv kopieren: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&dst, fs::Permissions::from_mode(0o755));
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("/usr/bin/xattr").args(["-d", "com.apple.quarantine"]).arg(&dst).output();
    }
    Ok(dst)
}

/// Mitgelieferte Wheels installieren (z. B. rawpy mit der neuesten LibRaw: liest auch ganz neue Kameras
/// wie die Sony A7 V ohne Adobe DNG Converter). Ersetzt die Version aus dem Internet. Lässt sich das
/// Paket danach nicht laden, kommt die Version aus dem Internet zurück: die App muss immer starten.
fn install_bundled_wheels(app: &AppHandle, uv: &Path, python: &Path, dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let stage = "RAW-Leser (neueste Kameras)";
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("whl") {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let package = name.split('-').next().unwrap_or("").to_string();
        // rawpy ohne Abhängigkeiten (numpy ist schon da), andere Pakete (insightface unter Windows) mit
        let mut cmd = command(uv);
        cmd.args(["pip", "install", "--reinstall"]);
        if package == "rawpy" {
            cmd.arg("--no-deps");
        }
        let installed = run_logged(app, stage, 0.28, cmd.arg("--python").arg(python).arg(&path));
        if let Err(e) = &installed {
            emit(app, stage, &format!("{name}: {e}"), 0.28, false, false);
        }
        if package != "rawpy" {
            // z. B. insightface (Windows): braucht onnxruntime, das erst später kommt -> hier nicht laden
            continue;
        }
        let check = "import rawpy, importlib; importlib.import_module('rawpy._rawpy')";
        let loads = installed.is_ok()
            && command(python)
                .args(["-c", check])
                .env("PATH", gui_path())
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
        if !loads {
            emit(app, stage, "mitgelieferte Version lässt sich nicht laden, nehme Standardversion", 0.28, false, false);
            let _ = run_logged(
                app,
                stage,
                0.28,
                command(uv).args(["pip", "install", "--reinstall", "--python"]).arg(python).arg(&package),
            );
        }
    }
}

fn bootstrap(app: &AppHandle) -> Result<PathBuf, String> {
    migrate_old_name();
    let res = app.path().resource_dir().map_err(|e| e.to_string())?;
    let runtime = data_root().join("runtime");
    let venv = runtime.join("venv");
    let python = venv_python(&venv);
    let version = app.package_info().version.to_string();
    let marker = runtime.join("version.txt");
    let installed = fs::read_to_string(&marker).unwrap_or_default();
    if installed.trim() == version && python.exists() {
        return Ok(python);
    }
    fs::create_dir_all(&runtime).map_err(|e| e.to_string())?;
    emit(app, "Vorbereiten", "Programmdateien kopieren", 0.02, false, false);
    let uv = prepare_uv(&res, &runtime)?;
    let src = runtime.join("src");
    let _ = fs::remove_dir_all(&src);
    copy_dir(&res.join("backend"), &src).map_err(|e| format!("Kopieren: {e}"))?;
    if !python.exists() {
        run_logged(
            app,
            "Python einrichten",
            0.08,
            command(&uv).args(["venv", "--python", "3.12"]).arg(&venv),
        )?;
    }
    run_logged(
        app,
        "Grundpakete installieren",
        0.2,
        command(&uv)
            .args(["pip", "install", "--python"])
            .arg(&python)
            .arg(format!("{}[scrape]", src.display())),
    )?;
    install_bundled_wheels(app, &uv, &python, &res.join("wheels"));
    let packages: Vec<&str> = ML_PACKAGES
        .iter()
        .copied()
        .filter(|p| cfg!(target_os = "macos") || !MAC_ONLY.iter().any(|m| p.starts_with(m)))
        .collect();
    let n = packages.len() as f32;
    for (i, pkg) in packages.iter().enumerate() {
        let p = 0.3 + 0.65 * (i as f32) / n;
        let stage = format!("KI-Pakete ({}/{})", i + 1, packages.len());
        if let Err(e) = run_logged(
            app,
            &stage,
            p,
            command(&uv).args(["pip", "install", "--python"]).arg(&python).arg(pkg),
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
    command(python)
        .args(["-m", "tagmatti.cli", "serve", "--port", "8765"])
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
            if std::env::var("TAGMATTI_DEV_BACKEND").is_ok() {
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
        .expect("Fehler beim Starten von Tagmatti");
    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<AppState>().backend.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
