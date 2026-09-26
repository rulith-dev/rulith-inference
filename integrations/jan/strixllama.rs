//! strixllama management over native IPC; the helper never exposes a network port.
use serde_json::Value;
use std::{io::Write, path::PathBuf, process::{Command, Stdio}};

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
