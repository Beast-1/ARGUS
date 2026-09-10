// The Tauri shell doesn't spawn or supervise the FastAPI backend (run_app.bat
// starts both as siblings) — it just needs to hand the frontend the shared auth
// token, which run_app.bat generates once and exports as ARGUS_API_TOKEN before
// launching both processes, so this reads the value from its own inherited env
// rather than generating one itself.
#[tauri::command]
fn get_api_token() -> String {
    std::env::var("ARGUS_API_TOKEN").unwrap_or_default()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_api_token])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
