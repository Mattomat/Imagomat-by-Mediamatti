//! Tauri-Hülle: startet das Python-Backend (Sidecar) und beendet es beim Schliessen.
//!
//! Release: gebündeltes Backend `binaries/imagomat-server-<target>` (PyInstaller, siehe scripts/build_backend.sh).
//! Entwicklung: Umgebungsvariable IMAGOMAT_DEV_BACKEND=1 -> Backend separat starten
//! (`imagomat serve`), die App verbindet sich nur.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

struct Backend(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Backend(Mutex::new(None)))
        .setup(|app| {
            if std::env::var("IMAGOMAT_DEV_BACKEND").is_err() {
                let cmd = app
                    .shell()
                    .sidecar("imagomat-server")?
                    .args(["serve", "--port", "8765"]);
                let (_rx, child) = cmd.spawn()?;
                *app.state::<Backend>().0.lock().unwrap() = Some(child);
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(child) = window.state::<Backend>().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("Fehler beim Starten von Imagomat");
}
