//! strixllama management over native IPC; the helper never exposes a network port.
use serde_json::Value;
use std::{io::Write, path::PathBuf, process::{Command, Stdio}};

/// The text of a document dropped on the chat box. An HTML5 drop hands the webview the file's bytes but not its path
/// (Jan turns Tauri's own drop handling off, since its image drops need the HTML5 events), so the bytes come here raw
/// and are read by the parser Jan's file picker reaches (tauri-plugin-rag), through a temporary file that is removed
/// again. The header x-file-type is the extension, as the picker passes it.
#[tauri::command]
pub async fn strixllama_parse_dropped(request: tauri::ipc::Request<'_>) -> Result<String, String> {
    let bytes: Vec<u8> = match request.body() {
        tauri::ipc::InvokeBody::Raw(bytes) => bytes.clone(),
        // the IPC's postMessage fallback carries them as a JSON array of numbers
        tauri::ipc::InvokeBody::Json(Value::Array(items)) => items.iter()
            .map(|v| v.as_u64().and_then(|n| u8::try_from(n).ok()))
            .collect::<Option<Vec<u8>>>().ok_or("Expected the file's bytes")?,
        _ => return Err("Expected the file's bytes".into()),
    };
    if bytes.len() as u64 > tauri_plugin_rag::MAX_PARSE_FILE_SIZE {
        return Err("File too large (max 200MB)".into());
    }
    // also the temporary file's extension, so nothing but letters and digits
    let file_type: String = request.headers().get("x-file-type").and_then(|v| v.to_str().ok()).unwrap_or("")
        .chars().filter(char::is_ascii_alphanumeric).take(16).collect::<String>().to_ascii_lowercase();
    tauri::async_runtime::spawn_blocking(move || {
        let dir = std::env::temp_dir().join("strixllama-drops");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("{}-{nanos}.{}", std::process::id(), if file_type.is_empty() { "bin" } else { &file_type }));
        std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
        let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || tauri_plugin_rag::parser::parse_document(&path.to_string_lossy(), &file_type)));
        let _ = std::fs::remove_file(&path);
        match parsed {
            Ok(result) => result.map_err(|e| e.to_string()),
            Err(_) => Err("Document parsing failed unexpectedly".into()),
        }
    }).await.map_err(|e| e.to_string())?
}

/// A file the page hands over for saving (issue #5): a code block's download button, a table's CSV, the Logs page. The
/// webview's own download of a blob link saved nowhere visible, so the bytes come here and go into the Downloads
/// folder under the name the page gave (x-file-name, percent-encoded UTF-8), numbered when that name is taken.
/// Returns the path written.
#[tauri::command]
pub async fn strixllama_save_download<R: tauri::Runtime>(app: tauri::AppHandle<R>, request: tauri::ipc::Request<'_>)
        -> Result<String, String> {
    use tauri::Manager;
    let bytes: Vec<u8> = match request.body() {
        tauri::ipc::InvokeBody::Raw(bytes) => bytes.clone(),
        tauri::ipc::InvokeBody::Json(Value::Array(items)) => items.iter()
            .map(|v| v.as_u64().and_then(|n| u8::try_from(n).ok()))
            .collect::<Option<Vec<u8>>>().ok_or("Expected the file's bytes")?,
        _ => return Err("Expected the file's bytes".into()),
    };
    let raw = request.headers().get("x-file-name").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let name = percent_decode(&raw);
    // a file name only: no folders, nothing Windows refuses in a name
    let mut clean: String = name.chars()
        .map(|c| if c.is_control() || "<>:\"/\\|?*".contains(c) { '_' } else { c }).collect();
    clean = clean.trim().trim_end_matches(['.', ' ']).to_string();
    if clean.is_empty() || clean.chars().all(|c| c == '.') { clean = "download".into(); }
    let dir = app.path().download_dir().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let (stem, ext) = match clean.rfind('.') {
            Some(i) if i > 0 => (clean[..i].to_string(), clean[i..].to_string()),
            _ => (clean.clone(), String::new()),
        };
        // create_new, so two saves of one name at once cannot land on the same file
        for n in 0..10000 {
            let path = if n == 0 { dir.join(&clean) } else { dir.join(format!("{stem} ({n}){ext}")) };
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    f.write_all(&bytes).map_err(|e| e.to_string())?;
                    return Ok(path.to_string_lossy().to_string());
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err("No free name for the file in Downloads".into())
    }).await.map_err(|e| e.to_string())?
}

fn percent_decode(s: &str) -> String {
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[tauri::command]
pub async fn strixllama_request(request: Value) -> Result<Value, String> {
    let op = request.get("op").and_then(Value::as_str).ok_or("Missing operation")?;
    if !["catalog", "status", "logs", "slots", "profile", "roots", "save", "start", "stop"].contains(&op) {
        return Err("Unsupported strixllama operation".into());
    }
    let input = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    if input.len() > 65536 { return Err("Request exceeds 64 KiB".into()); }
    tauri::async_runtime::spawn_blocking(move || {
        // Where the manager, runtime and models live. STRIX_ROOT if set; otherwise the runtime
        // bundle an installed copy carries beside its executable (tools/make_runtime_bundle.py,
        // shipped as Tauri resources under runtime/); otherwise, for a development build, the
        // repository this was compiled in.
        let root = std::env::var_os("STRIX_ROOT").map(PathBuf::from)
            .or_else(|| std::env::current_exe().ok()
                .and_then(|exe| exe.parent().map(|d| d.join("runtime")))
                .filter(|r| r.join("tools").join("manager.py").is_file()))
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).unwrap().to_path_buf());
        // STRIX_PYTHON if set; else the interpreter the bundle carries; else `python` on PATH. It
        // used to default to one machine's embedded interpreter, which no other machine has.
        let bundled_python = root.join("python").join("python.exe");
        let python = std::env::var_os("STRIX_PYTHON").map(PathBuf::from)
            .unwrap_or_else(|| if bundled_python.is_file() { bundled_python } else { PathBuf::from("python") });
        let helper = root.join("tools/manager.py");
        if !helper.is_file() {
            return Err(format!("The strixllama manager was not found at {}; STRIX_ROOT can point at the repository root", helper.display()));
        }
        let mut command = Command::new(&python);
        command.arg(helper).current_dir(root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().map_err(|e| format!(
            "Python ({}) could not be started: {e}. Put python on PATH, or set STRIX_PYTHON to an interpreter",
            python.display()))?;
        child.stdin.take().ok_or("Missing helper stdin")?.write_all(&input).map_err(|e| e.to_string())?;
        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        let result: Value = serde_json::from_slice(&output.stdout).map_err(|e| format!("Invalid manager response: {e}; {}", String::from_utf8_lossy(&output.stderr)))?;
        if result.get("ok").and_then(Value::as_bool) != Some(true) {
            // a coded error travels whole ({error, code, params}) so the pages can render it in
            // their own language; anything else is the plain message
            return Err(match result.get("code") {
                Some(_) => result.to_string(),
                None => result.get("error").and_then(Value::as_str).unwrap_or("The manager reported a failure").to_owned(),
            });
        }
        Ok(result["data"].clone())
    }).await.map_err(|e| e.to_string())?
}
