//! Windows desktop host for the shared Brain runtime.
use crate::strixllama::{helper, root};
use ed25519_dalek::{Signature, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::windows::io::AsRawHandle,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use tauri::{ipc::Channel, AppHandle, Emitter, Listener, Manager, WebviewWindow};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const ORIGIN: &str = "https://brain-ui.rulith.invalid";
const MANIFEST: &str = "rulith-brain-runtime.json";
const MEMORY_LIMIT: usize = 64 << 20;
const CAPS: [&str; 4] = [
    "ui.frame.v1",
    "manager.python.v1",
    "memory.stream.v1",
    "ui.store.v1",
];
type Reply = Result<Value, String>;

/// Tauri resolves this ACL against the sending document URL before unpacking command arguments.
pub struct LocalDocument;
impl<'de, R: tauri::Runtime> tauri::ipc::CommandArg<'de, R> for LocalDocument {
    fn from_command(
        item: tauri::ipc::CommandItem<'de, R>,
    ) -> Result<Self, tauri::ipc::InvokeError> {
        if item.acl.as_ref().is_some_and(|entries| {
            entries
                .iter()
                .any(|e| matches!(e.context, tauri::utils::acl::ExecutionContext::Local))
        }) && item.message.webview_ref().label() == "main"
        {
            Ok(Self)
        } else {
            Err(tauri::ipc::InvokeError::from(error("op_denied")))
        }
    }
}

fn error(code: &str) -> String {
    json!({"code":code,"error":code.replace('_'," "),"params":{}}).to_string()
}

fn home() -> Result<PathBuf, String> {
    dirs::data_local_dir()
        .map(|p| p.join("Rulith/Brain"))
        .ok_or_else(|| error("runtime_missing"))
}

fn lock(name: &str, shared: bool) -> Result<File, String> {
    #[link(name = "kernel32")]
    extern "system" {
        fn LockFileEx(
            file: *mut std::ffi::c_void,
            flags: u32,
            reserved: u32,
            low: u32,
            high: u32,
            offset: *mut usize,
        ) -> i32;
    }
    let dir = home()?.join("config/brain");
    fs::create_dir_all(&dir).map_err(|_| error("owner_unknown"))?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(name))
        .map_err(|_| error("owner_unknown"))?;
    let mut offset = [0usize; 5];
    if unsafe {
        LockFileEx(
            file.as_raw_handle(),
            if shared { 1 } else { 3 },
            0,
            1,
            0,
            offset.as_mut_ptr(),
        )
    } == 0
    {
        return Err(error(if shared { "runtime_busy" } else { "owner_busy" }));
    }
    Ok(file)
}

fn bounded(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let mut raw = Vec::new();
    File::open(path)
        .and_then(|f| f.take(max + 1).read_to_end(&mut raw))
        .map_err(|_| error("runtime_untrusted"))?;
    if raw.len() as u64 > max {
        return Err(error("runtime_integrity"));
    }
    Ok(raw)
}

fn signed_manifest() -> Result<(String, Value), String> {
    let live = home()?.join("bin/brain");
    let raw = bounded(&live.join(MANIFEST), 1 << 20)?;
    let sig = bounded(&live.join(format!("{MANIFEST}.sig")), 64)?;
    let pin: [u8; 32] = bounded(&root().join("tools/brain_runtime_trust.bin"), 32)?
        .try_into()
        .map_err(|_| error("runtime_untrusted"))?;
    let key = VerifyingKey::from_bytes(&pin).map_err(|_| error("runtime_untrusted"))?;
    let digest = Sha256::digest(&raw);
    let mut msg = b"rulith-brain-runtime-manifest-v2\n".to_vec();
    msg.extend_from_slice(&digest);
    key.verify_strict(
        &msg,
        &Signature::from_slice(&sig).map_err(|_| error("runtime_untrusted"))?,
    )
    .map_err(|_| error("runtime_untrusted"))?;
    Ok((
        hex::encode(digest),
        serde_json::from_slice(&raw).map_err(|_| error("runtime_schema"))?,
    ))
}

#[derive(Default)]
struct Flow {
    sent: u64,
    ack: u64,
    outstanding: std::collections::VecDeque<usize>,
    bytes: usize,
}
#[derive(Default)]
struct Active {
    cancel: CancellationToken,
    flow: Mutex<Flow>,
    stream: bool,
}
struct Mount {
    id: String,
    digest: String,
    _lease: File,
    assets: HashMap<String, Vec<u8>>,
    seen: HashSet<String>,
    calls: HashMap<String, Arc<Active>>,
    picked: HashSet<PathBuf>,
    offered: HashSet<PathBuf>,
}
impl Drop for Mount {
    fn drop(&mut self) {
        for active in self.calls.values() {
            active.cancel.cancel();
        }
    }
}
#[derive(Default)]
struct Desktop {
    owner: Option<File>,
    engine: Option<(File, String)>,
    mount: Option<Mount>,
    handler: bool,
}
fn desktop() -> &'static Mutex<Desktop> {
    static S: OnceLock<Mutex<Desktop>> = OnceLock::new();
    S.get_or_init(Default::default)
}
fn serial() -> &'static Mutex<()> {
    static S: Mutex<()> = Mutex::new(());
    &S
}

fn host_kind() -> &'static str {
    if env!("CARGO_PKG_NAME") == "rulith-brain" {
        "standalone"
    } else {
        "inference"
    }
}

fn owner() -> Result<Value, String> {
    static INSTALL: OnceLock<String> = OnceLock::new();
    static INSTANCE: OnceLock<String> = OnceLock::new();
    let install = if let Some(id) = INSTALL.get() {
        id.clone()
    } else {
        let dir = root().join("config/jan");
        fs::create_dir_all(&dir).map_err(|_| error("owner_unknown"))?;
        let path = dir.join("install-id");
        if let Ok(mut f) = OpenOptions::new().write(true).create_new(true).open(&path) {
            f.write_all(Uuid::new_v4().to_string().as_bytes())
                .and_then(|_| f.sync_all())
                .map_err(|_| error("owner_unknown"))?;
        }
        let id = fs::read_to_string(path).map_err(|_| error("owner_unknown"))?;
        Uuid::parse_str(&id).map_err(|_| error("owner_unknown"))?;
        INSTALL.get_or_init(|| id).clone()
    };
    Ok(
        json!({"host":host_kind(),"install_id":install,"instance_id":INSTANCE.get_or_init(||Uuid::new_v4().to_string())}),
    )
}

fn child(op: &str, data: Value, ctx: &Value) -> Reply {
    helper(&json!({"op":op,"data":data}), Some(ctx))
}
fn context(permit: bool) -> Reply {
    Ok(json!({"owner":owner()?,"permit":permit,"host_kind":host_kind()}))
}
fn claim() -> Result<(), String> {
    let mut s = desktop().lock().unwrap();
    if s.owner.is_none() {
        s.owner = Some(lock("engine-owner.lock", false)?);
    }
    Ok(())
}
fn pin_engine(ctx: &mut Value) -> Result<(), String> {
    if let Some((_, digest)) = &desktop().lock().unwrap().engine {
        ctx["digest"] = json!(digest);
        return Ok(());
    }
    let lease = lock("runtime.lock", true)?;
    let (hash, _) = signed_manifest()?;
    let mut check = ctx.clone();
    check["digest"] = json!(hash);
    child("host_verify", json!({}), &check)?;
    desktop().lock().unwrap().engine = Some((lease, hash));
    *ctx = check;
    Ok(())
}
fn release_idle(ctx: &Value) {
    if let Ok(status) = child("mode_status", json!({}), ctx) {
        let mut s = desktop().lock().unwrap();
        if status["active"] != "brain"
            && (status["target"] != "brain" || !status["error"].is_null())
        {
            s.engine = None;
        }
        if status["active"] == "none"
            && (status["state"] == "idle"
                || status["state"] == "launch_failed"
                || status["state"] == "blocked" && status["error"]["code"] != "owner_unknown")
        {
            s.owner = None;
            s.engine = None;
        }
    }
}

fn trusted(window: &WebviewWindow) -> Result<(), String> {
    // The app command ACL checks the actual sending frame's URL before command dispatch.
    if window.label() != "main"
        || window
            .url()
            .ok()
            .is_none_or(|u| u.origin().ascii_serialization() != "http://tauri.localhost")
    {
        return Err(error("op_denied"));
    }
    Ok(())
}

fn mode(op: &str, data: Value) -> Reply {
    if ![
        "mode_status",
        "mode_switch_brain",
        "mode_switch_inference",
        "mode_retry",
    ]
    .contains(&op)
    {
        return Err(error("op_denied"));
    }
    if op == "mode_status" {
        let mut held = desktop().lock().unwrap().owner.is_some();
        if !held {
            // Preview never claims an idle installation. Reattach only a journal from our installation.
            let journal = fs::read(home()?.join("config/brain/mode.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            if let Some(j) = journal {
                if j["owner"]["install_id"] == owner()?["install_id"] && j["state"] != "idle" {
                    held = claim().is_ok();
                }
            }
        }
        let mut ctx = context(held)?;
        if held {
            let record = fs::read(home()?.join("config/process.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            if record.is_some_and(|r| {
                r["identity"].is_object() && r["owner"]["install_id"] == ctx["owner"]["install_id"]
            }) {
                let _ = pin_engine(&mut ctx);
            }
        }
        return child(op, data, &ctx);
    }
    let Ok(_serial) = serial().try_lock() else {
        return child(
            "host_replay",
            json!({"op":op,"data":data}),
            &context(false)?,
        );
    };
    claim()?;
    let mut ctx = context(true)?;
    let accepted = match child("host_accept", json!({"op":op,"data":data}), &ctx) {
        Ok(value) => value,
        Err(e) => {
            release_idle(&ctx);
            return Err(e);
        }
    };
    if accepted["fresh"] == true {
        let status = accepted["status"].clone();
        std::thread::spawn(move || {
            let _serial = serial().lock().unwrap();
            if status["target"] == "brain" || status["active"] == "brain" {
                let _ = pin_engine(&mut ctx);
            }
            let _ = child("host_run", json!({}), &ctx);
            release_idle(&ctx);
        });
    }
    Ok(accepted["status"].clone())
}

async fn blocking<F>(f: F) -> Reply
where
    F: FnOnce() -> Reply + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|_| error("manager_failed"))?
}
#[tauri::command]
pub async fn brain_host_mode(
    _sender: LocalDocument,
    window: WebviewWindow,
    op: String,
    data: Value,
) -> Reply {
    trusted(&window)?;
    blocking(move || mode(&op, data)).await
}
#[tauri::command]
pub async fn brain_host_setup(
    _sender: LocalDocument,
    window: WebviewWindow,
    op: String,
    data: Value,
) -> Reply {
    trusted(&window)?;
    if ![
        "brain_runtime",
        "brain_runtime_install",
        "brain_runtime_cancel",
        "brain_runtime_check",
    ]
    .contains(&op.as_str())
    {
        return Err(error("op_denied"));
    }
    blocking(move || helper(&json!({"op":op,"data":data}), None)).await
}

pub async fn inference_request(request: Value) -> Reply {
    blocking(move || {
        let op = request["op"]
            .as_str()
            .ok_or_else(|| error("request_invalid"))?;
        if op == "status" {
            let result = helper(&request, None)?;
            if let Some(reason) = result["replacement_needed"].as_str() {
                if let Ok(_serial) = serial().try_lock() {
                    if desktop().lock().unwrap().owner.is_some() {
                        child("host_replace", json!({"reason":reason}), &context(true)?)?;
                    }
                }
            }
            return Ok(result);
        }
        if op == "save" {
            let _serial = serial().try_lock().map_err(|_| error("switch_busy"))?;
            return helper(
                &request,
                Some(&context(desktop().lock().unwrap().owner.is_some())?),
            );
        }
        let _serial = serial().lock().unwrap();
        claim()?;
        let ctx = context(true)?;
        let result = helper(&request, Some(&ctx));
        if op == "start" && result.is_ok() {
            let worker = ctx.clone();
            std::thread::spawn(move || {
                let _ = child("host_finish_start", json!({}), &worker);
                release_idle(&worker);
            });
        }
        release_idle(&ctx);
        result
    })
    .await
}

fn asset_map() -> Result<Mount, String> {
    let lease = lock("runtime.lock", true)?;
    let (digest, manifest) = signed_manifest()?;
    let mut ctx = context(false)?;
    ctx["digest"] = json!(digest);
    child("host_verify", json!({}), &ctx)?;
    let mut assets = HashMap::new();
    for (name, entry) in manifest["files"]
        .as_object()
        .ok_or_else(|| error("runtime_schema"))?
    {
        if let Some(path) = name.strip_prefix("ui/") {
            let raw = bounded(
                &home()?.join("bin/brain").join(name),
                entry["size"]
                    .as_u64()
                    .ok_or_else(|| error("runtime_schema"))?,
            )?;
            if entry["sha256"] != hex::encode(Sha256::digest(&raw)) {
                return Err(error("runtime_integrity"));
            }
            assets.insert(format!("/{path}"), raw);
        }
    }
    Ok(Mount {
        id: Uuid::new_v4().to_string(),
        digest,
        _lease: lease,
        assets,
        seen: HashSet::new(),
        calls: HashMap::new(),
        picked: HashSet::new(),
        offered: HashSet::new(),
    })
}

fn register_origin(window: &WebviewWindow) -> Result<(), String> {
    use webview2_com::{
        CoTaskMemPWSTR, FrameCreatedEventHandler, FrameDestroyedEventHandler,
        FrameNavigationStartingEventHandler, Microsoft::Web::WebView2::Win32::*,
        WebResourceRequestedEventHandler,
    };
    use windows::core::{Interface, HSTRING, PWSTR};
    use windows::Win32::UI::Shell::SHCreateMemStream;
    let (tx, rx) = std::sync::mpsc::channel();
    window.with_webview(move |wv| unsafe {
        let install=||->windows::core::Result<()> {
            let core=wv.controller().CoreWebView2()?;
            let env=wv.environment();
            core.cast::<ICoreWebView2_22>()?.AddWebResourceRequestedFilterWithRequestSourceKinds(
                &HSTRING::from(format!("{ORIGIN}/*")),COREWEBVIEW2_WEB_RESOURCE_CONTEXT_ALL,COREWEBVIEW2_WEB_RESOURCE_REQUEST_SOURCE_KINDS_ALL)?;
            let mut token=0;
            core.cast::<ICoreWebView2_4>()?.add_FrameCreated(&FrameCreatedEventHandler::create(Box::new(|_,args|{
                if let Some(args)=args {
                    let frame=args.Frame()?.cast::<ICoreWebView2Frame2>()?;
                    let bound=std::rc::Rc::new(std::cell::RefCell::new(None::<String>));
                    let destroyed=bound.clone();
                    let mut destroy_token=0;
                    frame.add_Destroyed(&FrameDestroyedEventHandler::create(Box::new(move|_,_|{
                        if let Some(id)=destroyed.borrow().as_ref(){
                            let mut s=desktop().lock().unwrap();
                            if s.mount.as_ref().is_some_and(|m|&m.id==id){s.mount=None;}
                        }
                        Ok(())
                    })),&mut destroy_token)?;
                    let mut navigation_token=0;
                    frame.add_NavigationStarting(&FrameNavigationStartingEventHandler::create(Box::new(move|_,args|{
                        if let Some(args)=args {
                            let mut uri=PWSTR::null();args.Uri(&mut uri)?;
                            let uri=CoTaskMemPWSTR::from(uri).to_string();
                            if bound.borrow().is_some(){args.SetCancel(true)?;}
                            else if uri==format!("{ORIGIN}/index.html"){
                                let generation=desktop().lock().unwrap().mount.as_ref().map(|m|m.id.clone());
                                if generation.is_none(){args.SetCancel(true)?;}
                                *bound.borrow_mut()=generation;
                            }
                            else if uri.starts_with(ORIGIN){args.SetCancel(true)?;}
                        }
                        Ok(())
                    })),&mut navigation_token)?;
                }
                Ok(())
            })),&mut token)?;
            core.add_WebResourceRequested(&WebResourceRequestedEventHandler::create(Box::new(move|_,args|{
                let Some(args)=args else{return Ok(())};
                let req=args.Request()?;
                let mut uri=PWSTR::null();
                req.Uri(&mut uri)?;
                let uri=CoTaskMemPWSTR::from(uri).to_string();
                let Ok(url)=url::Url::parse(&uri) else{return Ok(())};
                if url.origin().ascii_serialization()!=ORIGIN{return Ok(());}
                let s=desktop().lock().unwrap();
                let path=url.path();
                let bytes=s.mount.as_ref().and_then(|m|m.assets.get(path));
                let typ=match path.rsplit('.').next().unwrap_or("") {
                    "html"=>"text/html; charset=utf-8","js"|"mjs"=>"text/javascript","css"=>"text/css","svg"=>"image/svg+xml",
                    "png"=>"image/png","woff2"=>"font/woff2","json"=>"application/json",_=>"application/octet-stream"};
                let headers=format!("Content-Type: {typ}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'none'; worker-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors http://tauri.localhost; sandbox allow-same-origin allow-scripts");
                let stream=SHCreateMemStream(bytes.map(|v|v.as_slice()));
                let response=env.CreateWebResourceResponse(stream.as_ref(),if bytes.is_some(){200}else{404},
                    &HSTRING::from(if bytes.is_some(){"OK"}else{"Not Found"}),&HSTRING::from(headers))?;
                args.SetResponse(&response)
            })),&mut token)?;
            Ok(())
        };
        let _=tx.send(install().is_ok());
    }).map_err(|_|error("bridge_closed"))?;
    if rx.recv_timeout(Duration::from_secs(10)) != Ok(true) {
        return Err(error("bridge_closed"));
    }
    Ok(())
}
#[tauri::command]
pub async fn brain_host_mount(_sender: LocalDocument, window: WebviewWindow) -> Reply {
    trusted(&window)?;
    blocking(move || {
        let _serial = serial().lock().unwrap();
        let mount = asset_map()?;
        if !desktop().lock().unwrap().handler {
            register_origin(&window)?;
            desktop().lock().unwrap().handler = true;
        }
        let id = mount.id.clone();
        desktop().lock().unwrap().mount = Some(mount);
        Ok(json!({"url":format!("{ORIGIN}/index.html"),"generation":id}))
    })
    .await
}
#[tauri::command]
pub fn brain_host_unmount(
    _sender: LocalDocument,
    window: WebviewWindow,
    generation: String,
) -> Reply {
    trusted(&window)?;
    let mut s = desktop().lock().unwrap();
    if s.mount.as_ref().is_some_and(|m| m.id == generation) {
        s.mount = None;
    }
    Ok(Value::Null)
}

fn fields(v: &Value, allowed: &[&str], required: &[&str]) -> Result<(), String> {
    let o = v.as_object().ok_or_else(|| error("request_invalid"))?;
    if !o.keys().all(|k| allowed.contains(&k.as_str()))
        || !required.iter().all(|k| o.contains_key(*k))
    {
        return Err(error("request_invalid"));
    }
    Ok(())
}
fn valid_id(id: &str) -> bool {
    Uuid::parse_str(id).is_ok_and(|v| {
        v.get_version_num() == 4 && v.to_string() == id && v.get_variant() == uuid::Variant::RFC4122
    })
}
fn unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"-._~".contains(&b)
}
fn segment(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut decoded = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'%' && i + 2 < bytes.len() && bytes[i + 1..i + 3].iter().all(u8::is_ascii_hexdigit)
        {
            let c = u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap();
            if c.is_ascii_control() || b"/\\%?#:".contains(&c) {
                return false;
            }
            decoded.push(c);
            i += 3;
        } else if unreserved(b) {
            decoded.push(b);
            i += 1;
        } else {
            return false;
        }
    }
    !decoded.is_empty()
        && decoded != b"."
        && decoded != b".."
        && std::str::from_utf8(&decoded).is_ok()
}
fn memory_path(method: &str, path: &str) -> bool {
    if path == "/slots" {
        return method == "GET";
    }
    let (name, query) = path
        .split_once('?')
        .map_or((path, None), |(n, q)| (n, Some(q)));
    name.strip_prefix("/v1/memory/")
        .is_some_and(|s| s.split('/').all(segment))
        && query.is_none_or(|q| {
            !q.is_empty()
                && q.split('&').all(|p| {
                    let (k, v) = p.split_once('=').unwrap_or((p, ""));
                    !k.is_empty() && k.bytes().chain(v.bytes()).all(unreserved)
                })
        })
}
fn private_memory_path(path: &str) -> bool {
    let raw = path.split('?').next().unwrap_or("");
    if raw == "/slots" {
        return false;
    }
    // Compare the same decoded operation name the listener will dispatch.
    let leaf = raw.rsplit('/').next().unwrap_or("").as_bytes();
    let mut name = Vec::new();
    let mut i = 0;
    while i < leaf.len() {
        if leaf[i] == b'%' && i + 2 < leaf.len() {
            let Ok(pair) = std::str::from_utf8(&leaf[i + 1..i + 3]) else {
                return false;
            };
            let Ok(b) = u8::from_str_radix(pair, 16) else {
                return false;
            };
            name.push(b);
            i += 3;
        } else {
            name.push(leaf[i]);
            i += 1;
        }
    }
    name != b"input" && name != b"chat"
}
fn memory_size(bytes: usize, additional: usize, code: &str) -> Result<(), String> {
    if additional > MEMORY_LIMIT || bytes > MEMORY_LIMIT - additional {
        Err(error(code))
    } else {
        Ok(())
    }
}
fn utf8_prefix(bytes: &[u8]) -> Result<usize, String> {
    match std::str::from_utf8(bytes) {
        Ok(_) => Ok(bytes.len()),
        Err(e) if e.error_len().is_none() => Ok(e.valid_up_to()),
        Err(_) => Err(error("bridge_protocol")),
    }
}
async fn cancellable<F: std::future::Future<Output = Reply>>(active: &Active, work: F) -> Reply {
    tokio::select! {biased;_=active.cancel.cancelled()=>Err(error("cancelled")),r=work=>r}
}
fn external_url(s: &str) -> bool {
    let Some(after_scheme) = s.strip_prefix("https://") else {
        return false;
    };
    if after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .contains('@')
    {
        return false;
    }
    if s.contains(['\\', '%']) && s.split('?').next().unwrap_or("").contains(['\\', '%'])
        || s.chars().any(|c| c.is_control() || c.is_whitespace())
        || s.split('?')
            .next()
            .unwrap_or("")
            .split('/')
            .any(|p| p == "." || p == "..")
    {
        return false;
    }
    let Ok(u) = url::Url::parse(s) else {
        return false;
    };
    if u.scheme() != "https"
        || u.port().is_some()
        || !u.username().is_empty()
        || u.password().is_some()
        || u.fragment().is_some()
    {
        return false;
    }
    match u.host_str() {
        Some("rulith.ai") => true,
        Some("github.com") => u.path().starts_with("/rulith-dev/"),
        Some("huggingface.co") => u.path().starts_with("/Rulith/"),
        Some("creem.io" | "www.creem.io") => u.path().starts_with("/checkout/"),
        _ => false,
    }
}
fn send(channel: &Channel<Value>, generation: &str, v: Value) -> Result<(), String> {
    let s = desktop().lock().unwrap();
    if !s.mount.as_ref().is_some_and(|m| m.id == generation) {
        return Err(error("bridge_closed"));
    }
    channel.send(v).map_err(|_| error("bridge_closed"))
}

async fn memory(
    op: &str,
    body: &Value,
    id: &str,
    generation: &str,
    active: &Active,
    ch: &Channel<Value>,
) -> Reply {
    fields(body, &["method", "path", "body"], &["method", "path"])?;
    let method = body["method"].as_str().unwrap_or("");
    let path = body["path"].as_str().unwrap_or("");
    if !["GET", "POST", "DELETE"].contains(&method)
        || !memory_path(method, path)
        || path.len() + id.len() + op.len() + 64 > 8192
        || body.get("body").is_some_and(|b| {
            !(op == "brain_memory" && b.is_null()) && (method != "POST" || !b.is_object())
        })
    {
        return Err(error("request_invalid"));
    }
    if op == "brain_memory" {
        if !private_memory_path(path) {
            return Err(error("op_denied"));
        }
        if serde_json::to_vec(&json!({"op":op,"data":body}))
            .map_err(|_| error("request_invalid"))?
            .len()
            > 65536
        {
            return Err(error("request_too_large"));
        }
    }
    let payload = if method == "POST" {
        Some(
            serde_json::to_vec(
                body.get("body")
                    .filter(|v| !v.is_null())
                    .unwrap_or(&json!({})),
            )
            .map_err(|_| error("request_invalid"))?,
        )
    } else {
        None
    };
    if let Some(b) = &payload {
        memory_size(0, b.len(), "request_too_large")?;
    }
    let ep = blocking(|| child("host_endpoint", json!({}), &context(false)?)).await?;
    let base = url::Url::parse(
        ep["url"]
            .as_str()
            .ok_or_else(|| error("brain_not_running"))?,
    )
    .map_err(|_| error("owner_unknown"))?;
    if base.scheme() != "http" || base.host_str() != Some("127.0.0.1") {
        return Err(error("owner_unknown"));
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| error("brain_memory_unreachable"))?;
    let mut req = client.request(
        reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
        format!("{}{path}", base.as_str().trim_end_matches('/')),
    );
    for k in ["Authorization", "X-Rulith-Admin", "Content-Type"] {
        if let Some(v) = ep["headers"][k].as_str() {
            req = req.header(k, v);
        }
    }
    if let Some(bytes) = payload {
        req = req.body(bytes);
    }
    let mut res = req
        .send()
        .await
        .map_err(|_| error("brain_memory_unreachable"))?;
    let status = res.status().as_u16();
    let mut headers = serde_json::Map::new();
    for (k, v) in res.headers() {
        let name = k.as_str();
        if (name == "content-type" || name.starts_with("x-memory-") || name.starts_with("x-brain-"))
            && !["auth", "key", "secret", "admin", "token"]
                .iter()
                .any(|s| name.contains(s))
        {
            if let Ok(text) = v.to_str() {
                headers.insert(name.to_owned(), json!(text));
            }
        }
    }
    let stream = op == "memory_stream";
    if !stream
        && res
            .content_length()
            .is_some_and(|n| n > MEMORY_LIMIT as u64)
    {
        return Err(error("response_too_large"));
    }
    if stream {
        send(
            ch,
            generation,
            json!({"v":1,"id":id,"type":"head","status":status,"headers":headers}),
        )?;
    }
    let mut body = Vec::new();
    let mut seq = 0;
    while let Some(chunk) = res
        .chunk()
        .await
        .map_err(|_| error("brain_memory_unreachable"))?
    {
        if !stream {
            memory_size(body.len(), chunk.len(), "response_too_large")?;
            body.extend_from_slice(&chunk);
            continue;
        }
        for part in chunk.chunks(65532) {
            body.extend_from_slice(part);
            let n = utf8_prefix(&body)?;
            if n == 0 {
                continue;
            }
            let text = std::str::from_utf8(&body[..n]).unwrap();
            {
                let mut flow = active.flow.lock().unwrap();
                if flow.bytes + n > 1 << 20 {
                    return Err(error("slow_consumer"));
                }
                flow.sent += 1;
                flow.outstanding.push_back(n);
                flow.bytes += n;
            }
            if active.cancel.is_cancelled() {
                return Err(error("cancelled"));
            }
            send(
                ch,
                generation,
                json!({"v":1,"type":"chunk","id":id,"seq":seq,"text":text}),
            )?;
            body.drain(..n);
            seq += 1;
        }
    }
    if stream {
        if !body.is_empty() {
            return Err(error("bridge_protocol"));
        }
        Ok(json!({"v":1,"id":id,"type":"end","next_seq":seq}))
    } else {
        let mut out = json!({"status":status,"headers":headers,"data":null});
        match serde_json::from_slice(&body) {
            Ok(j) => out["data"] = j,
            Err(_) => out["text"] = json!(String::from_utf8_lossy(&body)),
        };
        if op == "brain_memory" {
            out.as_object_mut().unwrap().remove("headers");
        }
        Ok(out)
    }
}

fn mounted_child(op: &str, data: Value, generation: &str) -> Reply {
    let mut ctx = context(false)?;
    {
        let s = desktop().lock().unwrap();
        let m = s
            .mount
            .as_ref()
            .filter(|m| m.id == generation)
            .ok_or_else(|| error("bridge_closed"))?;
        ctx["digest"] = json!(m.digest);
        ctx["picked"] = json!(m.picked);
        ctx["offered"] = json!(m.offered);
    }
    ctx["frame"] = json!(true);
    ctx["scope"] = json!("brain");
    if op.starts_with("ui_store_") {
        ctx["store"] = json!(true);
    }
    if op == "stop" {
        fields(&data, &[], &[])?;
        let _serial = serial().lock().unwrap();
        claim()?;
        let _lease = lock("runtime.lock", true)?;
        ctx["permit"] = json!(true);
        let res = child(op, data, &ctx);
        release_idle(&ctx);
        if let Err(ref e) = res {
            let fault: Value = serde_json::from_str(e).unwrap_or_default();
            if fault["code"] == "save_failed" {
                return Ok(
                    json!({"status":"running","memory_saved":false,"save_failed":true,"reason":"save_failed"}),
                );
            }
        }
        return res;
    }
    if op == "brain_app_start" {
        fields(&data, &[], &[])?;
        let mut state = mode(
            "mode_switch_brain",
            json!({"request_id":Uuid::new_v4().to_string()}),
        )?;
        let tid = state["transition_id"].clone();
        while state["transition_id"] == tid
            && state["error"].is_null()
            && state["state"] != "brain_ready"
        {
            std::thread::sleep(Duration::from_millis(200));
            state = mode("mode_status", json!({}))?;
        }
        if !state["error"].is_null() {
            return Err(state["error"].to_string());
        }
        return child("host_launch_view", json!({}), &ctx);
    }
    let _lease = lock("runtime.lock", true)?;
    let v = child(op, data, &ctx)?;
    if op == "brain_app" {
        if let Some(m) = desktop()
            .lock()
            .unwrap()
            .mount
            .as_mut()
            .filter(|m| m.id == generation)
        {
            if let Some(p) = v["memory_root"].as_str() {
                m.offered.insert(PathBuf::from(p));
            }
            for row in v["memories"].as_array().into_iter().flatten() {
                if let Some(p) = row["dir"].as_str() {
                    m.offered.insert(PathBuf::from(p));
                }
            }
        }
    }
    Ok(v)
}

const FRAME_OPS:&str="brain_memory brain_licence brain_trial brain_activate brain_transfer brain_upgrade brain_upgrade_finish brain_create brain_runtime brain_runtime_install brain_runtime_cancel brain_runtime_check brain_app brain_app_select brain_app_readonly brain_room brain_app_start brain_self brain_self_set brain_capacity brain_capacity_set brain_folders brain_folder_add brain_folder_remove brain_folder_set brain_folder_scan brain_web_settings brain_web_set brain_web_search brain_web_fetch status logs stop roots dl_start dl_status dl_pause dl_resume dl_cancel memory memory_stream open_link pick_folder open_folder app_info set_theme ui_store_read ui_store_write";

async fn operation(
    win: &WebviewWindow,
    op: &str,
    b: Value,
    id: &str,
    g: &str,
    a: &Active,
    ch: &Channel<Value>,
) -> Reply {
    if !FRAME_OPS.split_whitespace().any(|o| o == op) {
        return Err(error("op_denied"));
    }
    match op {
        "memory" | "memory_stream" | "brain_memory" => memory(op, &b, id, g, a, ch).await,
        "app_info" => {
            fields(&b, &[], &[])?;
            Ok(
                json!({"host":host_kind(),"name":if host_kind()=="standalone"{"Rulith Brain"}else{"Rulith Inference"},"version":win.app_handle().package_info().version.to_string(),
                "host_abi":1,"home":home()?,"root":root(),"licence_file":home()?.join("licence.txt"),"capabilities":CAPS}),
            )
        }
        "set_theme" => {
            fields(&b, &["theme"], &["theme"])?;
            let theme = match b["theme"].as_str() {
                Some("dark") => Some(tauri::Theme::Dark),
                Some("light") => Some(tauri::Theme::Light),
                Some("system") => None,
                _ => return Err(error("request_invalid")),
            };
            win.set_theme(theme).map_err(|_| error("bridge_closed"))?;
            Ok(Value::Null)
        }
        "pick_folder" => {
            fields(&b, &["start", "title"], &[])?;
            let mut picker = rfd::AsyncFileDialog::new();
            if let Some(t) = b["title"].as_str() {
                picker = picker.set_title(t);
            }
            if let Some(p) = b["start"].as_str() {
                let path = PathBuf::from(p);
                let s = desktop().lock().unwrap();
                let m = s
                    .mount
                    .as_ref()
                    .filter(|m| m.id == g)
                    .ok_or_else(|| error("bridge_closed"))?;
                if path != home()? && !m.picked.contains(&path) && !m.offered.contains(&path) {
                    return Err(error("folder_denied"));
                }
                picker = picker.set_directory(path);
            }
            let Some(folder) = picker.pick_folder().await else {
                return Ok(Value::Null);
            };
            let path = folder.path().to_owned();
            if !path.is_dir()
                || !matches!(path.components().next(),Some(std::path::Component::Prefix(p)) if matches!(p.kind(), std::path::Prefix::Disk(_)))
            {
                return Err(error("folder_denied"));
            }
            let mut s = desktop().lock().unwrap();
            let m = s
                .mount
                .as_mut()
                .filter(|m| m.id == g)
                .ok_or_else(|| error("bridge_closed"))?;
            m.picked.insert(path.clone());
            Ok(json!(path))
        }
        "open_link" | "open_folder" => {
            let arg = if op == "open_link" { "url" } else { "path" };
            fields(&b, &[arg], &[arg])?;
            let value = b[arg].as_str().ok_or_else(|| error("request_invalid"))?;
            if op == "open_link" {
                if !external_url(value) {
                    return Err(error("op_denied"));
                }
                tauri_plugin_opener::open_url(value, None::<&str>)
                    .map_err(|_| error("op_denied"))?;
            } else {
                let path = PathBuf::from(value);
                let s = desktop().lock().unwrap();
                let m = s
                    .mount
                    .as_ref()
                    .filter(|m| m.id == g)
                    .ok_or_else(|| error("bridge_closed"))?;
                if !path.is_dir()
                    || path != home()? && !m.picked.contains(&path) && !m.offered.contains(&path)
                {
                    return Err(error("folder_denied"));
                }
                tauri_plugin_opener::open_path(value, None::<&str>)
                    .map_err(|_| error("folder_denied"))?;
            }
            Ok(Value::Null)
        }
        _ => {
            let op = op.to_owned();
            let g = g.to_owned();
            blocking(move || mounted_child(&op, b, &g)).await
        }
    }
}

#[tauri::command]
pub async fn brain_host_call(
    _sender: LocalDocument,
    window: WebviewWindow,
    generation: String,
    request: Value,
    on_event: Channel<Value>,
) -> Result<(), String> {
    trusted(&window)?;
    let id = request["id"]
        .as_str()
        .filter(|s| valid_id(s))
        .ok_or_else(|| error("bridge_protocol"))?;
    if request["v"].as_u64() != Some(1) {
        return Err(error("bridge_protocol"));
    }
    let active = Arc::new(Active {
        stream: request["op"] == "memory_stream",
        ..Active::default()
    });
    {
        let mut s = desktop().lock().unwrap();
        let m = s
            .mount
            .as_mut()
            .filter(|m| m.id == generation)
            .ok_or_else(|| error("bridge_closed"))?;
        match request["type"].as_str() {
            Some("cancel") => {
                fields(&request, &["v", "type", "id"], &["id"])?;
                if let Some(a) = m.calls.get(id).filter(|a| a.stream) {
                    a.cancel.cancel();
                }
                return Ok(());
            }
            Some("ack") => {
                fields(&request, &["v", "type", "id", "next_seq"], &["next_seq"])?;
                let n = request["next_seq"]
                    .as_u64()
                    .ok_or_else(|| error("bridge_protocol"))?;
                if let Some(a) = m.calls.get(id) {
                    let mut f = a.flow.lock().unwrap();
                    if n < f.ack || n > f.sent {
                        return Err(error("bridge_protocol"));
                    }
                    while f.ack < n {
                        let len = f.outstanding.pop_front().unwrap();
                        f.bytes -= len;
                        f.ack += 1;
                    }
                }
                return Ok(());
            }
            Some("request") => {
                fields(
                    &request,
                    &["v", "type", "id", "op", "body"],
                    &["op", "body"],
                )?;
                let op = request["op"]
                    .as_str()
                    .ok_or_else(|| error("request_invalid"))?;
                if !request["body"].is_object() {
                    return Err(error("request_invalid"));
                }
                if op.len() + id.len() + 64 > 8192 {
                    return Err(error("request_too_large"));
                }
                if !m.seen.insert(id.to_owned()) {
                    return Err(error("bridge_protocol"));
                }
                m.calls.insert(id.to_owned(), active.clone());
            }
            _ => return Err(error("bridge_protocol")),
        }
    }
    tauri::async_runtime::spawn(async move {
        let id = request["id"].as_str().unwrap();
        let op = request["op"].as_str().unwrap_or("");
        let result = cancellable(
            &active,
            operation(
                &window,
                op,
                request["body"].clone(),
                id,
                &generation,
                &active,
                &on_event,
            ),
        )
        .await;
        let event = match result {
            Ok(v) if op == "memory_stream" => v,
            Ok(v) => json!({"v":1,"type":"result","id":id,"body":v}),
            Err(e) => {
                let err: Value =
                    serde_json::from_str(&e).unwrap_or(json!({"code":"manager_failed"}));
                json!({"v":1,"type":"error","id":id,"code":err["code"],"params":err.get("params").cloned().unwrap_or(json!({})),
                   "message":err["error"].as_str().unwrap_or("The request could not be completed.")})
            }
        };
        let _ = send(&on_event, &generation, event);
        if let Some(m) = desktop()
            .lock()
            .unwrap()
            .mount
            .as_mut()
            .filter(|m| m.id == generation)
        {
            m.calls.remove(id);
        }
    });
    Ok(())
}

fn close_engine() -> Result<(), String> {
    let _serial = serial().lock().unwrap();
    let held = desktop().lock().unwrap().owner.is_some();
    if !held && claim().is_err() {
        let journal: Value =
            serde_json::from_slice(&bounded(&home()?.join("config/brain/mode.json"), 4 << 20)?)
                .map_err(|_| error("owner_unknown"))?;
        let other = &journal["owner"];
        if !other["install_id"].is_string() {
            return Err(error("owner_unknown"));
        }
        if other["install_id"] == owner()?["install_id"] && other["host"] == host_kind() {
            return Err(error("owner_busy"));
        }
        return Ok(());
    }
    let mut ctx = context(true)?;
    let status = child("mode_status", json!({}), &ctx)?;
    if status["error"]["code"] == "owner_busy" {
        desktop().lock().unwrap().owner = None;
        return Ok(());
    }
    if status["active"] == "brain" {
        pin_engine(&mut ctx)?;
    }
    child("host_close", json!({}), &ctx)?;
    Ok(())
}
pub fn prepare_uninstall() -> Result<(), String> {
    close_engine()
}

pub async fn prepare_exit(app: &AppHandle, reason: &str) -> Result<(), String> {
    let generation = desktop()
        .lock()
        .unwrap()
        .mount
        .as_ref()
        .map(|m| m.id.clone());
    let id = Uuid::new_v4().to_string();
    let result = async {
        if let Some(g) = generation.clone() {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let check = id.clone();
            let listen = app.listen("brain-host-flushed", move |e| {
                if let Ok(v) = serde_json::from_str::<Value>(e.payload()) {
                    if v["generation"] == g
                        && v["id"] == check
                        && v["v"] == 1
                        && v["type"] == "flushed"
                    {
                        let _ = tx.try_send(v["ok"] == true);
                    }
                }
            });
            let sent = app.emit_to(
                "main",
                "brain-host-lifecycle",
                json!({"generation":generation,"event":{
                "v":1,"type":"event","op":"closing","id":id,"body":{"reason":reason}}}),
            );
            let flushed = sent.is_ok()
                && matches!(
                    tokio::time::timeout(Duration::from_secs(10), rx.recv()).await,
                    Ok(Some(true))
                );
            app.unlisten(listen);
            if !flushed {
                return Err(error("store_flush_failed"));
            }
        }
        blocking(|| {
            close_engine()?;
            Ok(Value::Null)
        })
        .await?;
        Ok(())
    }
    .await;
    if let Err(ref e) = result {
        let code = serde_json::from_str::<Value>(e).unwrap_or_default();
        let _=app.emit_to("main","brain-host-lifecycle",json!({"generation":generation,"event":{
            "v":1,"type":"event","op":"closeFailed","id":id,"body":{"reason":reason,"code":code["code"]}}}));
    } else {
        desktop().lock().unwrap().mount = None;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_memory_paths_and_sizes() {
        assert!(memory_path("GET", "/slots"));
        assert!(memory_path("POST", "/v1/memory/items/%E6%96%87?count=2"));
        for path in [
            "/slots?debug=1",
            "//host",
            "/v1/memory/%2e%2e",
            "/v1/memory/a%2Fb",
            "/v1/memory/%252f",
            "/v1/memory//x",
            "/v1/memory/x#y",
            "/v1/memory/x?q=%0d",
        ] {
            assert!(!memory_path("GET", path));
        }
        assert!(!private_memory_path("/v1/memory/%69nput"));
        for code in ["request_too_large", "response_too_large"] {
            assert!(memory_size(MEMORY_LIMIT - 2, 1, code).is_ok());
            assert!(memory_size(MEMORY_LIMIT - 2, 2, code).is_ok());
            assert!(memory_size(MEMORY_LIMIT - 2, 3, code).is_err());
        }
    }
    #[test]
    fn split_utf8_and_external_links() {
        let raw = "a文\r\n\r\n".as_bytes();
        assert_eq!(utf8_prefix(&raw[..2]).unwrap(), 1);
        assert_eq!(utf8_prefix(&raw[..3]).unwrap(), 1);
        assert_eq!(utf8_prefix(raw).unwrap(), raw.len());
        assert!(utf8_prefix(&[255]).is_err());
        assert!(external_url(
            "https://github.com/rulith-dev/rulith-inference"
        ));
        assert!(!external_url("https://github.com/rulith-dev/../other"));
        assert!(!external_url("https://creem.io@another.invalid/checkout/a"));
    }
    #[tokio::test]
    async fn cancelling_drops_a_silent_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        for headers in [false, true] {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let url = format!("http://{}/", listener.local_addr().unwrap());
            let (reached, wait) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let (mut peer, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    request.push(peer.read_u8().await.unwrap());
                }
                if headers {
                    peer.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\n")
                        .await
                        .unwrap();
                }
                reached.send(()).unwrap();
                tokio::time::timeout(Duration::from_secs(3), peer.read_u8())
                    .await
                    .unwrap()
                    .is_err()
            });
            let active = Arc::new(Active::default());
            let cancel = active.cancel.clone();
            let request = tokio::spawn(async move {
                cancellable(&active, async move {
                    reqwest::Client::builder()
                        .no_proxy()
                        .build()
                        .unwrap()
                        .get(url)
                        .send()
                        .await
                        .unwrap()
                        .bytes()
                        .await
                        .unwrap();
                    Ok(Value::Null)
                })
                .await
            });
            wait.await.unwrap();
            cancel.cancel();
            assert!(request.await.unwrap().is_err());
            assert!(server.await.unwrap());
        }
    }
}
