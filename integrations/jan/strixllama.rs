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
    if !["catalog", "status", "logs", "slots", "profile", "roots", "save", "start", "stop", "network", "save_network",
         // Downloads (0.6.0): the catalog, a repository's files, and the one download job
         "dl_catalog", "dl_search", "dl_repo", "dl_readme", "dl_info", "dl_start", "dl_status", "dl_pause", "dl_resume",
         "dl_cancel"].contains(&op) {
        return Err("Unsupported strixllama operation".into());
    }
    #[cfg(windows)]
    if matches!(op, "start" | "stop" | "status" | "save") {
        return crate::brain_host::inference_request(request).await;
    }
    tauri::async_runtime::spawn_blocking(move || helper(&request, None)).await
        .map_err(|_| "The manager stopped unexpectedly".to_owned())?
}

pub(crate) fn root() -> PathBuf {
    std::env::var_os("STRIX_ROOT").map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok()?.parent().map(|dir| dir.join("runtime"))
            .filter(|r| r.join("tools/manager.py").is_file()))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).unwrap().to_path_buf())
}

pub(crate) fn helper(request: &Value, context: Option<&Value>) -> Result<Value, String> {
    use std::io::Read;
    let input = serde_json::to_vec(request).map_err(|_| "Malformed request")?;
    let limit = if context.is_some_and(|v| v["store"] == true) { (16 << 20) + 8192 } else { 65536 };
    if input.len() > limit { return Err(r#"{"code":"request_too_large","error":"The request is too large.","params":{}}"#.into()); }
    let root = root();
    let bundled = root.join("python/python.exe");
    let python = std::env::var_os("STRIX_PYTHON").map(PathBuf::from)
        .unwrap_or_else(|| if bundled.is_file() { bundled } else { "python".into() });
    let mut cmd = Command::new(python);
    cmd.args(["-I","-B","-S"]).arg(root.join("tools/manager.py"));
    if context.is_some() { cmd.arg("--native"); }
    cmd.current_dir(&root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let mut child = cmd.spawn().map_err(|_| "The bundled manager could not start")?;
    let outcome = (|| {
        let mut pipe = child.stdin.take().ok_or("Manager pipe unavailable")?;
        if let Some(ctx) = context {
            serde_json::to_writer(&mut pipe,ctx).map_err(|_| "Manager pipe closed")?;
            pipe.write_all(b"\n").map_err(|_| "Manager pipe closed")?;
        }
        pipe.write_all(&input).map_err(|_| "Manager pipe closed")?;
        drop(pipe);
        let mut out = Vec::new();
        child.stdout.take().ok_or("Manager pipe unavailable")?.take((64<<20)+1)
            .read_to_end(&mut out).map_err(|_| "Manager pipe closed")?;
        if out.len()>64<<20 { return Err(r#"{"code":"response_too_large","error":"The response is too large.","params":{}}"#.into()); }
        let v: Value = serde_json::from_slice(&out).map_err(|_| "Malformed manager response")?;
        if v["ok"] != true { return Err(v.to_string()); }
        Ok(v["data"].clone())
    })();
    if outcome.is_err() { let _ = child.kill(); }
    let _ = child.wait();
    outcome
}
