"""Apply the strixllama management pages to a Jan v0.8.4 source tree.

    python integrations/jan/apply.py [path to a Jan checkout] [--keep-data-dir]

Every edit is anchored against upstream Jan text and fails loudly rather than guessing if an anchor
has moved, so a Jan version this was not written for is a clear error and not a half-applied tree.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The name people see, and the one machines use. The package, the binary, the provider id and the
# data directory are the slug; the window, the sidebar, the credits and the installer show the name.
# Up to 0.3.5 the app was Strix Llama (LEGACY_NAME, the name an existing install is found by); it was
# renamed in 0.3.6 so as not to be confused with halo-box/strix-llama.cpp, and the slug stayed, so the
# data, the settings and the app's identity carry over unchanged.
NAME = 'Rulith Inference'
LEGACY_NAME = 'Strix Llama'
SLUG = 'strixllama'
# The repository on GitHub, renamed from rulith-dev/strixllama with the app: GitHub redirects the old
# name, so the old updater endpoint and posted links keep working.
REPO = 'rulith-dev/rulith-inference'
# The publisher the installer registers the app under (Jan's configuration names Menlo Research, which
# made Jan, not this build); the uninstall entry and the install-location key go under it.
PUBLISHER = 'Rulith'
# Ours, not Jan's: the installer's file name, the uninstall entry and Settings › General show it.
VERSION = '0.4.8'
ARGS = [a for a in sys.argv[1:] if not a.startswith('-')]
KEEP_DATA_DIR = '--keep-data-dir' in sys.argv
JAN = Path(ARGS[0]).resolve() if ARGS else ROOT / 'src/jan'
# The model pages (StrixLlamaPage lays out the three views), the welcome screen that replaces Jan's
# setup screen, the chat header's model state, the app-wide status poll, and the canned answers the
# browser preview uses (fixtures.ts, imported only by development builds).
UI_FILES = ('StrixLlamaPage.tsx', 'ModelsView.tsx', 'ConfigurationView.tsx', 'LogsView.tsx', 'Welcome.tsx',
            'ModelState.tsx', 'Sidebar.tsx', 'StrixLlamaSync.tsx', 'parts.tsx', 'store.ts', 'status.ts',
            'attachments.ts', 'compact.ts', 'downloads.ts', 'fixtures.ts', 'strixllama.css', 'rulith-theme.css')
# NSIS setup hooks (Tauri's bundle.windows.nsis.installerHooks), run before the files are laid down
NSIS_HOOKS = r"""Var LegacyDir

; An install made before the rename (up to 0.3.5, as Strix Llama). The installer names the folder, the
; uninstall entry, the install-location key and the shortcuts after the product, so the new name's setup
; would install a second copy beside the old one - and the model server's settings and prompt cache live
; under runtime\config in the install folder, tens of GB. So the new version goes into the old folder, as
; an update would, and the old name's entries give way to the new name's (NSIS_HOOK_POSTINSTALL).
!macro STRIX_FIND_LEGACY
  ReadRegStr $LegacyDir HKCU "Software\Menlo Research Pte. Ltd.\@LEGACY@" ""
  ${If} $LegacyDir == ""
    ; the uninstall entry's copy is quoted
    ReadRegStr $LegacyDir HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@LEGACY@" "InstallLocation"
    StrCpy $0 $LegacyDir 1
    ${If} $0 == '"'
      StrCpy $LegacyDir $LegacyDir "" 1
      StrCpy $LegacyDir $LegacyDir -1
    ${EndIf}
  ${EndIf}
  ${IfNot} ${FileExists} "$LegacyDir\@SLUG@.exe"
    StrCpy $LegacyDir ""
  ${EndIf}
!macroend

; a shortcut under the old name, renamed (it points at the same exe, which has not moved)
!macro STRIX_RENAME_SHORTCUT DIR
  ${If} ${FileExists} "${DIR}\@LEGACY@.lnk"
    ${If} ${FileExists} "${DIR}\${PRODUCTNAME}.lnk"
      Delete "${DIR}\@LEGACY@.lnk"
    ${Else}
      Rename "${DIR}\@LEGACY@.lnk" "${DIR}\${PRODUCTNAME}.lnk"
    ${EndIf}
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro STRIX_FIND_LEGACY
  ; only in place of the default folder: a folder chosen on the directory page is kept
  ${If} $LegacyDir != ""
  ${AndIf} $INSTDIR == "$LOCALAPPDATA\${PRODUCTNAME}"
    ; Section Install made the default folder already; it is empty
    SetOutPath "$LOCALAPPDATA"
    RMDir "$INSTDIR"
    StrCpy $INSTDIR $LegacyDir
    SetOutPath $INSTDIR
  ${EndIf}
  ; a model server still running from this install holds runtime\bin\hip open. Stop it as the app does -
  ; through the manager, which writes its conversations to the disk tier first - then stop whatever of this
  ; install is still running: only processes whose file lies under $INSTDIR.
  IfFileExists "$INSTDIR\runtime\tools\manager.py" 0 +2
    nsExec::Exec 'cmd /c echo {"op":"stop"}| "$INSTDIR\runtime\python\python.exe" "$INSTDIR\runtime\tools\manager.py"'
  nsExec::Exec `powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
  ; up to 0.2.4 the runtime lived in runtime\bin\hip-rocm101, named after the ROCm 10.1 it was first built
  ; with; it is runtime\bin\hip now, and the old copy would otherwise stay behind after an upgrade
  RMDir /r "$INSTDIR\runtime\bin\hip-rocm101"
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; the old name's uninstall entry and install-location key, now that the new name's cover the same folder
  ; (two entries would uninstall one folder), and its shortcuts, renamed. A taskbar pin is left as it is: it
  ; points at the same exe and keeps working.
  ${If} $LegacyDir != ""
  ${AndIf} $LegacyDir == $INSTDIR
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@LEGACY@"
    DeleteRegKey HKCU "Software\Menlo Research Pte. Ltd.\@LEGACY@"
    DeleteRegKey /ifempty HKCU "Software\Menlo Research Pte. Ltd."
    !insertmacro STRIX_RENAME_SHORTCUT "$SMPROGRAMS"
    !insertmacro STRIX_RENAME_SHORTCUT "$DESKTOP"
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; The uninstaller deletes the files the installer laid down, one by one, and leaves what the runtime wrote
  ; beside them: its logs and bytecode, and runtime\config - the manager's settings, the model catalog and
  ; the prompt cache, which grows to tens of GB. Those go with "Delete the application data", like the app's
  ; other data; an update (the updater runs the old uninstaller) keeps everything.
  ${If} $UpdateMode <> 1
    RMDir /r "$INSTDIR\runtime\logs"
    RMDir /r "$INSTDIR\runtime\tools\__pycache__"
    ${If} $DeleteAppDataCheckboxState = 1
      RMDir /r "$INSTDIR\runtime\config"
    ${EndIf}
    RMDir "$INSTDIR\runtime\tools"
    RMDir "$INSTDIR\runtime"
    RMDir "$INSTDIR"
  ${EndIf}
!macroend
""".replace('@LEGACY@', LEGACY_NAME).replace('@SLUG@', SLUG)

def replace_once(path, old, new):
    text = path.read_text(encoding='utf-8')
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f'Upstream source changed: {path}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8', newline='\n')

def main():
    shutil.copyfile(HERE / 'strixllama.rs', JAN / 'src-tauri/src/strixllama.rs')
    ui = JAN / 'web-app/src/components/strixllama'
    ui.mkdir(parents=True, exist_ok=True)
    for name in UI_FILES:
        shutil.copyfile(HERE / name, ui / name)
    # StrixLlamaSync polls the manager for the whole app and registers the one provider Jan's chat
    # needs, so a fresh install can chat without first naming an endpoint in a dialog.
    root = JAN / 'web-app/src/routes/__root.tsx'
    replace_once(root, "import { DataProvider } from '@/providers/DataProvider'\n",
                 "import { DataProvider } from '@/providers/DataProvider'\n"
                 "import { StrixLlamaSync } from '@/components/strixllama/StrixLlamaSync'\n")
    replace_once(root, "            <DataProvider />\n", "            <DataProvider />\n            <StrixLlamaSync />\n")
    # Jan's i18n discovers namespaces with import.meta.glob over locales/**/*.json, so dropping the
    # files in is enough - no registration to patch. A language Jan has but we do not falls back to
    # its own fallbackLng, which is en.
    for locale in sorted(p.name for p in (HERE / 'locales').iterdir() if p.is_dir()):
        target = JAN / 'web-app/src/locales' / locale
        if not target.is_dir():
            raise RuntimeError(f'Jan has no locale {locale}; update integrations/jan/locales')
        shutil.copyfile(HERE / 'locales' / locale / 'strixllama.json', target / 'strixllama.json')
    routes = JAN / 'web-app/src/routes/strixllama'
    routes.mkdir(parents=True, exist_ok=True)
    for view in ('models', 'configuration', 'logs'):
        (routes / f'{view}.tsx').write_text(
            "import { createFileRoute } from '@tanstack/react-router'\n"
            "import StrixLlamaPage from '@/components/strixllama/StrixLlamaPage'\n"
            f"export const Route = createFileRoute('/strixllama/{view}')({{\n"
            f"  component: () => <StrixLlamaPage view=\"{view}\" />,\n}})\n", encoding='utf-8')
    # the sidebar links to /strixllama, so its one entry is active on all three views
    (routes / 'index.tsx').write_text(
        "import { createFileRoute, redirect } from '@tanstack/react-router'\n"
        "export const Route = createFileRoute('/strixllama/')({\n"
        "  beforeLoad: () => { throw redirect({ to: '/strixllama/models' }) },\n})\n", encoding='utf-8')
    # the log view was /strixllama/developer up to 0.2.4; the file is ours, written by this script
    stale = routes / 'developer.tsx'
    if stale.is_file() and "createFileRoute('/strixllama/developer')" in stale.read_text(encoding='utf-8'):
        stale.unlink()
    lib = JAN / 'src-tauri/src/lib.rs'
    replace_once(lib, 'pub mod core;', 'pub mod core;\nmod strixllama;')
    replace_once(lib, 'tauri::generate_handler![', 'tauri::generate_handler![\n            strixllama::strixllama_request,')
    replace_once(lib, '            strixllama::strixllama_request,\n',
                 '            strixllama::strixllama_request,\n            strixllama::strixllama_parse_dropped,\n')
    replace_once(lib, '            strixllama::strixllama_parse_dropped,\n',
                 '            strixllama::strixllama_parse_dropped,\n            strixllama::strixllama_save_download,\n')
    replace_once(JAN / 'src-tauri/src/main.rs', '#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]', '#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]')
    # The chat goes through the app's Rust HTTP client (reqwest), which picks up the Windows
    # system proxy but not its "bypass for 127.*" list - so with Clash or similar on, every
    # request to the local server was handed to the proxy, and when the proxy could not reach it
    # the chat failed with "Bad Gateway" (seen in app.log: proxy(http://127.0.0.1:7897/) intercepts
    # 'http://127.0.0.1:8080/'). reqwest does honour NO_PROXY, for the registry proxy as well.
    replace_once(JAN / 'src-tauri/src/main.rs', "    app_lib::run();\n", """    // strixllama: the model server is on loopback and must never be routed through a proxy
    let mut no_proxy = String::from("127.0.0.1,localhost,::1");
    if let Ok(existing) = std::env::var("NO_PROXY") {
        if !existing.is_empty() {
            no_proxy.push(',');
            no_proxy.push_str(&existing);
        }
    }
    std::env::set_var("NO_PROXY", &no_proxy);

    app_lib::run();
""")
    nav = JAN / 'web-app/src/components/left-sidebar/NavMain.tsx'
    # a tree this script patched before has entries of its own in the main list - the three model pages up
    # to 0.2.4, one model entry in the first 0.2.5 builds - and their imports: take them out first. The model
    # pages are a sidebar group of their own now (Sidebar.tsx), and Settings moved to the sidebar's foot.
    text = nav.read_text(encoding='utf-8')
    for old_line in ("  { title: 'strixllama:tabs.models', url: '/strixllama/models', icon: Database },\n",
                     "  { title: 'strixllama:tabs.configuration', url: '/strixllama/configuration', icon: SlidersHorizontal },\n",
                     "  { title: 'strixllama:tabs.developer', url: '/strixllama/developer', icon: Terminal },\n",
                     "  { title: 'strixllama:title', url: '/strixllama', icon: Cpu, shortcut: <NavStatus /> },\n"):
        text = text.replace(old_line, '')
    text = text.replace("import { LucideIcon, Database, SlidersHorizontal, Terminal } from 'lucide-react'",
                        "import { LucideIcon } from 'lucide-react'")
    text = text.replace("import { LucideIcon, Cpu } from 'lucide-react'\n"
                        "import { NavStatus } from '@/components/strixllama/ModelState'", "import { LucideIcon } from 'lucide-react'")
    nav.write_text(text, encoding='utf-8', newline='\n')
    # Jan never marks the section you are in: an entry with a URL is active on every page under it
    # (Settings on all of /settings, the model entry on all of /strixllama)
    replace_once(nav, "import { Link, useNavigate } from '@tanstack/react-router'",
                 "import { Link, useNavigate, useRouterState } from '@tanstack/react-router'")
    replace_once(nav, "  ).filter((item) => item.title !== 'common:newAgentChat')\n",
                 "  ).filter((item) => item.title !== 'common:newAgentChat')\n"
                 "    .map((item) => ({ ...item, isActive: item.isActive ?? (!!item.url && pathname.startsWith(item.url.replace(/\\/general$/, ''))) }))\n")
    replace_once(nav, "  const navigate = useNavigate()\n",
                 "  const navigate = useNavigate()\n"
                 "  const pathname = useRouterState({ select: (s) => s.location.pathname })\n")
    # Jan's setup screen offers to download Jan's own model; ours finds the model files and loads one
    home = JAN / 'web-app/src/routes/index.tsx'
    thread = JAN / 'web-app/src/routes/threads/$threadId.tsx'
    for page in (home, thread):   # the first 0.2.5 builds put a pill beside the model picker
        text = page.read_text(encoding='utf-8')
        text = (text.replace("import { ModelState } from '@/components/strixllama/ModelState'\n", '')
                    .replace("          <ModelState />\n", ''))
        page.write_text(text, encoding='utf-8', newline='\n')
    replace_once(home, "import SetupScreen from '@/containers/SetupScreen'\n",
                 "import Welcome from '@/components/strixllama/Welcome'\n"
                 "import { ModelBanner } from '@/components/strixllama/ModelState'\n")
    replace_once(home, "    return <SetupScreen />\n", "    return <Welcome />\n")
    # above a new chat and a thread, as Rulith's desktop app has it: why the model cannot answer, and the fix
    replace_once(home, '    <div className="flex h-full flex-col justify-center">\n      <HeaderPage>',
                 '    <div className="flex h-full flex-col justify-center">\n      <ModelBanner />\n      <HeaderPage>')
    replace_once(thread, "import DropdownModelProvider from '@/containers/DropdownModelProvider'\n",
                 "import DropdownModelProvider from '@/containers/DropdownModelProvider'\n"
                 "import { ModelBanner } from '@/components/strixllama/ModelState'\n")
    replace_once(thread, '      <HeaderPage>\n        <div className="flex items-center justify-between w-full pr-2">',
                 '      <ModelBanner />\n      <HeaderPage>\n        <div className="flex items-center justify-between w-full pr-2">')
    # the sidebar in Rulith's shape: flush with a hairline instead of a floating card, the model pages as a
    # group under the chat entries, the loaded model and Settings at its foot (brand() puts the brand block in)
    side = JAN / 'web-app/src/components/left-sidebar/index.tsx'
    replace_once(side, "import { NavProjects } from './NavProjects'\n",
                 "import { NavProjects } from './NavProjects'\n"
                 "import { ModelNav, ModelPanel, SidebarBrand } from '@/components/strixllama/Sidebar'\n")
    replace_once(side, "  SidebarContent,\n", "  SidebarContent,\n  SidebarFooter,\n")
    replace_once(side, '<Sidebar variant="floating" collapsible="offcanvas">', '<Sidebar variant="sidebar" collapsible="offcanvas">')
    replace_once(side, "          <NavMain />\n        </SidebarHeader>", "          <NavMain />\n          <ModelNav />\n        </SidebarHeader>")
    replace_once(side, "        <SidebarRail />", '        <SidebarFooter className="p-0">\n          <ModelPanel />\n        </SidebarFooter>\n        <SidebarRail />')
    drop_nav_entries(nav)
    # Rulith's palette is fixed (rulith-theme.css overrides --primary and --sidebar), so Jan's accent colour
    # picker would change nothing: it goes, kept in the source behind a constant
    interface = JAN / 'web-app/src/routes/settings/interface.tsx'
    replace_once(interface, """              <CardItem
                title="Accent color"
                description="Customize the accent color of the application."
                className="flex-col sm:flex-row items-start sm:items-center sm:justify-between gap-y-2"
                actions={<AccentColorPicker />}
              />""", """              {/* strixllama: Rulith's palette is fixed, see rulith-theme.css */}
              {false && (
              <CardItem
                title="Accent color"
                description="Customize the accent color of the application."
                className="flex-col sm:flex-row items-start sm:items-center sm:justify-between gap-y-2"
                actions={<AccentColorPicker />}
              />
              )}""")
    # Settings in the model pages' layout: the card's name as a small caps label above a bordered card,
    # 13 px titles over 12 px descriptions (Jan's are 16 and 14 px with the name inside the card)
    card = JAN / 'web-app/src/containers/Card.tsx'
    replace_once(card, """        <div className="space-y-1.5">
          <h1 className="font-medium text-foreground">{title}</h1>
          {description && (
            <span className="text-muted-foreground leading-normal">""", """        <div className="space-y-1">
          <h1 className="text-[13px] font-medium text-foreground">{title}</h1>
          {description && (
            <span className="text-xs text-muted-foreground leading-normal">""")
    replace_once(card, """    <div className="bg-card p-4 rounded-lg text-muted-foreground w-full">
      {title && (
        <h1 className="text-foreground font-studio font-medium text-base mb-4">
          {title}
        </h1>
      )}
      {header && header}
      {children}
    </div>""", """    <div className="w-full not-first:mt-2">
      {title && (
        <h1 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-[var(--rl-label)]">
          {title}
        </h1>
      )}
      <div className="bg-card border px-4 py-3 rounded-lg text-[13px] text-muted-foreground w-full">
        {header && header}
        {children}
      </div>
    </div>""")
    # "Colored user message bubble" filled your messages with the accent, which is white on dark in Rulith's
    # palette: a blue tint instead (Rulith's link colour; green means a state), the plain grey one when off
    replace_once(JAN / 'web-app/src/containers/MessageItem.tsx', "? 'bg-primary text-primary-foreground'",
                 "? 'bg-[var(--rl-bubble)] text-foreground'")
    # Our provider is llama-server, but Jan builds its requests as for any OpenAI-compatible endpoint, which drops
    # the llama.cpp-only sampling keys: Repeat Penalty, its window and Min P were shown in the chat's parameter panel
    # and never sent. Send them, with the rest of what Jan asserts for its own llama-server (cache_prompt, progress).
    replace_once(JAN / 'web-app/src/lib/model-factory.ts',
                 ": createCustomFetch(getRuntimeFetch(), parameters, false, undefined, true)",
                 ": createCustomFetch(getRuntimeFetch(), parameters, provider.provider === 'strixllama', undefined, true)")
    # A reply the server ended with an error - every answer in flight when an allocation failed - was saved as an
    # ordinary finished message, cut off mid-sentence and without a Continue button; later turns then copied the
    # cut-off text. It is saved as a stopped turn now, like one the user stopped: marked, and continued in place.
    thread_page = JAN / 'web-app/src/routes/threads/$threadId.tsx'
    # A thread keeps its own copy of the assistant it was started with (thread.json), and that copy is what the chat
    # sends - so the date line brand_text() removes from the default assistant stayed in every existing conversation.
    # It is dropped when the prompt is built: an existing conversation's prefix changes once, then never again.
    replace_once(thread_page, """  const systemMessage = threadAssistant?.instructions
    ? renderInstructions(threadAssistant.instructions)
    : undefined""", """  // strixllama: without the date line Jan's default instructions ended with (see brand_text in apply.py)
  const threadInstructions =
    threadAssistant?.id === 'jan' && threadAssistant.instructions
      ? threadAssistant.instructions.replace(/\\s*Current date: \\{\\{\\s*current_date\\s*\\}\\}\\s*$/, '')
      : threadAssistant?.instructions
  const systemMessage = threadInstructions
    ? renderInstructions(threadInstructions)
    : undefined""")
    replace_once(thread_page, "    onFinish: ({ message, isAbort }) => {\n",
                 "    onFinish: ({ message, isAbort, isError, isDisconnect }) => {\n")
    replace_once(thread_page, "      const isStoppedTurn = isAbort || finishReason === 'length'\n",
                 "      // strixllama: an error or a dropped connection mid-reply leaves a partial too\n"
                 "      const isStoppedTurn = isAbort || isError || isDisconnect || finishReason === 'length'\n")
    converge_settings()
    drop_integrations()
    documents()
    web_tools()
    chat_width()
    compaction()
    concurrent_tools()
    brand(KEEP_DATA_DIR)
    brand_text()
    # Jan's own llama.cpp engine is not loaded at all. Left in, its extension downloads a Vulkan
    # backend on first start (llamacpp-b9967-win-vulkan..., seen in the download tray), starts an
    # embedding server, and registers the provider the picker then has to filter out. Everything
    # that looks it up does so by name with a fallback, as on the platforms where it is absent.
    replace_once(JAN / 'web-app/src/services/core/bundled-extensions.ts', """  {
    load: () => import('@janhq/llamacpp-extension'),
    name: '@janhq/llamacpp-extension',
    productName: 'llama.cpp Inference Engine',
    version: '1.0.1',
    description: 'This extension enables llama.cpp chat completion API calls',
  },
""", """  // strixllama: no llama.cpp engine of Jan's - inference is the server tools/manager.py starts
""")
    print('strixllama: native command, pages, sidebar, settings and branding applied to %s' % JAN)


def cut_lines(path, first_line, last_line, expect_first, expect_last):
    """Delete an inclusive 1-based line range, refusing unless both anchors still match.

    apply.py is run again by every build, so this has to be idempotent: once the range is gone the
    file is shorter and the anchors no longer line up, and the call becomes a no-op.
    """
    lines = path.read_text(encoding='utf-8').split('\n')
    if last_line > len(lines):
        return False                       # already cut
    if expect_first not in lines[first_line - 1] or expect_last not in lines[last_line - 1]:
        return False                       # already cut, or upstream moved
    del lines[first_line - 1:last_line]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return True


def cut_block(path, start_marker):
    """Delete the statement beginning at start_marker, through its matching closing brace.

    Anchor-based rather than by line number, because each cut shifts everything after it.
    """
    text = path.read_text(encoding='utf-8')
    if start_marker not in text:
        return False                       # already cut
    start = text.index(start_marker)
    depth, i, seen = 0, start, False
    while i < len(text):
        if text[i] == '{':
            depth += 1
            seen = True
        elif text[i] == '}':
            depth -= 1
            if seen and depth == 0:
                break
        i += 1
    end = text.index('\n', i) + 1          # take the rest of the closing line, e.g. "}, [])"
    path.write_text(text[:start] + text[end:], encoding='utf-8', newline='\n')
    return True


def drop_declarations(path, names):
    """Delete named import specifiers and whole single-line declarations.

    The web app builds with noUnusedLocals, so every card and menu entry removed here takes its
    icon import - and sometimes a constant - down with it.
    """
    lines = path.read_text(encoding='utf-8').split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        # a specifier on its own line inside a multi-line import
        if any(stripped in (f'{n},', n) for n in names):
            continue
        # a whole single-line import or const
        if any(stripped.startswith(f'import {{ {n} }}') or stripped.startswith(f'const {n} =')
               for n in names):
            continue
        out.append(line)
    path.write_text('\n'.join(out), encoding='utf-8', newline='\n')


def brand(keep_data_dir=False):
    """Make the built app strixllama's rather than Jan's: name, window title, icon, data directory.

    Jan keeps its data in %APPDATA%/<Cargo package name>/data - the threads, the settings, the
    providers - and names the binary after the package too, so the package is renamed along with
    the identifier. That gives this build its own directory instead of sharing an installed Jan's,
    which is what you want for a separate product: two apps writing one settings directory is how
    you lose a conversation history. It does mean an existing Jan install's data is not carried
    over. --keep-data-dir leaves both the package name and the identifier alone.

    This renames a *build* of Jan, which Apache-2.0 allows. NOTICE.md states what it is; do not
    imply that Jan endorses it.
    """
    icons = HERE / 'icons'
    if not (icons / 'icon.ico').is_file():
        subprocess.run([sys.executable, str(HERE / 'make_icons.py')], check=True)
    # A clean Jan checkout ships only icon.png; the other sizes are generated at build time. We
    # write the whole set, because tauri.conf.json's bundle.icon names several of them explicitly
    # and a missing one fails the bundle rather than falling back.
    dst = JAN / 'src-tauri/icons'
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(icons.iterdir()):
        if f.suffix in ('.png', '.ico', '.icns'):
            shutil.copyfile(f, dst / f.name)
            copied += 1

    conf = JAN / 'src-tauri/tauri.conf.json'
    data = json.loads(conf.read_text(encoding='utf-8'))
    data['productName'] = NAME
    data['mainBinaryName'] = SLUG      # strixllama.exe, whatever the product is called
    data.setdefault('bundle', {})['publisher'] = PUBLISHER
    data['version'] = VERSION
    # Settings › General reads the web app's package version
    web_pkg = JAN / 'web-app/package.json'
    web = json.loads(web_pkg.read_text(encoding='utf-8'))
    # ...and its build type-checks before Vite runs, against the committed routeTree.gen.ts, which does
    # not list the /strixllama routes: a fresh checkout failed there. Vite's router plugin writes the
    # tree as the build starts, so type-check after it.
    build = web.get('scripts', {}).get('build')
    if web.get('version') != VERSION or build == 'tsc -b && vite build':
        web['version'] = VERSION
        if build == 'tsc -b && vite build':
            web['scripts']['build'] = 'vite build && tsc -b'
        web_pkg.write_text(json.dumps(web, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    if not keep_data_dir:
        data['identifier'] = 'dev.rulith.strixllama'
        cargo = JAN / 'src-tauri/Cargo.toml'
        replace_once(cargo, '[package]\nname = "Jan"\n', '[package]\nname = "strixllama"\n')
        replace_once(cargo, 'default-run = "Jan"\n', 'default-run = "strixllama"\n')
        # No jan-cli: it serves Jan's engine, brand() already stops installing it, and Tauri
        # bundles every [[bin]] of the package - so an unbuilt one fails the bundle outright.
        cli_bin = '[[bin]]\nname = "jan-cli"\npath = "src/bin/jan-cli.rs"\nrequired-features = ["cli"]\n'
        text = cargo.read_text(encoding='utf-8')
        if cli_bin in text:
            cargo.write_text(text.replace(cli_bin, '', 1), encoding='utf-8', newline='\n')
        # ...and without the explicit target cargo would auto-discover src/bin/jan-cli.rs and try
        # to compile it without its feature. Discovery off covers src/main.rs too, so the one
        # binary is declared explicitly.
        replace_once(cargo, '[package]\nname = "strixllama"\n', '[package]\nname = "strixllama"\nautobins = false\n')
        replace_once(cargo, '[lib]\nname = "app_lib"\n',
                     '[[bin]]\nname = "strixllama"\npath = "src/main.rs"\n\n[lib]\nname = "app_lib"\n')
        # The Tauri CLI has its own discovery too: it bundles every file under src/bin whatever
        # Cargo.toml says, and fails when the binary was never built. The source goes.
        cli_src = JAN / 'src-tauri/src/bin/jan-cli.rs'
        if cli_src.is_file():
            cli_src.unlink()
            if not any(cli_src.parent.iterdir()):
                cli_src.parent.rmdir()
        # The bundle-identifier constant is only used to look for a legacy settings file to
        # migrate, and the migration deletes the file it copies. Pointed at Jan's directory, a
        # first run would carry off - and remove - an installed Jan's settings.
        constants = JAN / 'src-tauri/src/core/app/constants.rs'
        replace_once(constants, 'pub const TAURI_BUNDLE_IDENTIFIER: &str = "jan.ai.app";',
                     'pub const TAURI_BUNDLE_IDENTIFIER: &str = "dev.rulith.strixllama";')
        replace_once(constants, 'assert_eq!(TAURI_BUNDLE_IDENTIFIER, "jan.ai.app");',
                     'assert_eq!(TAURI_BUNDLE_IDENTIFIER, "dev.rulith.strixllama");')
    # The updater, pointed at this project's releases and this project's key. Jan's configuration named
    # Jan's endpoints and carried Jan's signing key, so an upstream release would have verified and installed
    # over this build - runtime and model pages gone - and up to 0.2.4 the updater was simply off. Since 0.2.5
    # every release carries latest.json and a signature made with the key whose public half is updater.pub;
    # an update is installed only if it verifies against that. The private key never enters the repository:
    # a build signs only when TAURI_SIGNING_PRIVATE_KEY is set, and one without it still checks for, verifies
    # and installs official releases (tools/make_update_manifest.py writes latest.json at release time).
    plugins = data.get('plugins') or {}
    plugins['updater'] = {
        'pubkey': (HERE / 'updater.pub').read_text(encoding='utf-8').strip(),
        'endpoints': [f'https://github.com/{REPO}/releases/latest/download/latest.json'],
        'windows': {'installMode': 'passive'},
    }
    data['plugins'] = plugins
    bundle = data.get('bundle') or {}
    bundle['createUpdaterArtifacts'] = bool(os.environ.get('TAURI_SIGNING_PRIVATE_KEY'))
    data['bundle'] = bundle
    # The runtime bundle, when one has been made (tools/make_runtime_bundle.py): the server, the
    # ROCm DLLs it needs, the manager and an embedded Python, installed under <app>/runtime so a
    # user needs nothing but the model files. Without one, the build is a development build that
    # runs the repository it was compiled in.
    runtime = ROOT / 'dist' / 'runtime'
    resources = bundle.get('resources') or []
    if isinstance(resources, list):
        resources = {r: r for r in resources}
    resources = {k: v for k, v in resources.items() if not v.startswith('runtime/') and 'jan-cli' not in k}
    staged = JAN / 'src-tauri' / 'runtime'
    if (runtime / 'BUNDLE.json').is_file():
        # Copied into the Tauri project and mapped as a directory: Tauri walks a directory into
        # the target preserving its structure (a glob flattens every match by file name), and a
        # path that climbs out of the project was silently left out of the bundle.
        if staged.is_dir():
            shutil.rmtree(staged)
        shutil.copytree(runtime, staged)
        resources['runtime'] = 'runtime/'
        print('  brand: runtime bundle from %s' % runtime)
    bundle['resources'] = resources
    data['bundle'] = bundle
    conf.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

    # No `jan` command on the user's PATH. Jan copies its CLI into resources/bin at every launch
    # and appends that directory to the Windows user PATH; the CLI serves Jan's engine, which this
    # build never loads, and an application editing the user's environment on startup is not
    # something to inherit. The settings card that offered it went in converge_settings().
    replace_once(JAN / 'src-tauri/src/lib.rs',
                 "            setup::setup_jan_cli(app.handle().clone(), stored_version != app_version);\n",
                 "            // strixllama: no `jan` CLI install - it serves Jan's engine and edits the user's PATH\n"
                 "            let _ = (&stored_version, &app_version);\n")

    # The tray icon's menu and tooltip are Rust string literals: "Open Jan" was still there in 0.2.5's first builds
    tray = JAN / 'src-tauri/src/core/setup.rs'
    replace_once(tray, 'MenuItem::with_id(app.handle(), "open", "Open Jan", true, None::<&str>)?',
                 f'MenuItem::with_id(app.handle(), "open", "Open {NAME}", true, None::<&str>)?')
    replace_once(tray, '        .icon(app.default_window_icon().unwrap().clone())\n        .menu(&menu)\n',
                 f'        .icon(app.default_window_icon().unwrap().clone())\n        .tooltip("{NAME}")\n        .menu(&menu)\n')

    # The window title lives in the per-platform config, not in index.html and not in the main one -
    # this is the name in the title bar, which is the first thing anyone sees.
    for name in ('tauri.windows.conf.json', 'tauri.macos.conf.json', 'tauri.linux.conf.json'):
        platform = JAN / 'src-tauri' / name
        if not platform.is_file():
            continue
        pdata = json.loads(platform.read_text(encoding='utf-8'))
        windows = pdata.get('app', {}).get('windows') or []
        changed = False
        for w in windows:
            if w.get('title') == 'Jan':
                w['title'] = NAME
                changed = True
        # The runtime's folder was runtime\bin\hip-rocm101 up to 0.2.4 and is runtime\bin\hip since: a hook
        # removes the old one before the files are laid down, so an upgrade cannot leave 340 MB behind.
        if name == 'tauri.windows.conf.json':
            hooks = JAN / 'src-tauri' / 'windows' / 'strixllama-hooks.nsh'
            hooks.parent.mkdir(exist_ok=True)
            hooks.write_text(NSIS_HOOKS, encoding='utf-8', newline='\r\n')
            nsis = pdata.setdefault('bundle', {}).setdefault('windows', {}).setdefault('nsis', {})
            if nsis.get('installerHooks') != './windows/strixllama-hooks.nsh':
                nsis['installerHooks'] = './windows/strixllama-hooks.nsh'
                changed = True
        # One installer. Jan also builds an MSI, which needs the WiX toolset fetched from GitHub
        # at bundle time and adds nothing the NSIS setup does not already do.
        targets = pdata.get('bundle', {}).get('targets')
        if isinstance(targets, list) and 'msi' in targets:
            pdata['bundle']['targets'] = [t for t in targets if t != 'msi']
            changed = True
        # The platform file's own bundle.resources replaces the main one wholesale, so the runtime
        # bundle has to be declared here too or the installer quietly ships without it.
        presources = pdata.get('bundle', {}).get('resources')
        if presources is not None:
            if isinstance(presources, list):
                presources = {r: r for r in presources}
            presources = {k: v for k, v in presources.items() if v != 'runtime/' and 'jan-cli' not in k}
            if 'runtime' in resources:
                presources['runtime'] = 'runtime/'
            if presources != pdata['bundle'].get('resources'):
                pdata['bundle']['resources'] = presources
                changed = True
        if not changed:
            continue
        platform.write_text(json.dumps(pdata, indent=2, ensure_ascii=False) + '\n',
                            encoding='utf-8', newline='\n')

    # build:tauri starts with `tauri icon`, which regenerates every size from icon.png by plain
    # downscaling — including the 16 px one, where make_icons.py deliberately drops the ring and
    # grows the eyes. We ship the whole set already, so drop that step and keep the hinted sizes.
    pkg = JAN / 'package.json'
    scripts = json.loads(pkg.read_text(encoding='utf-8'))
    tauri = scripts['scripts'].get('build:tauri', '')
    if 'build:icon &&' in tauri:
        scripts['scripts']['build:tauri'] = tauri.replace('yarn build:icon && ', '', 1)
        pkg.write_text(json.dumps(scripts, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

    html = JAN / 'web-app/index.html'
    replace_once(html, '<title>Jan</title>', f'<title>{NAME}</title>')
    # The splash: index.html shows /images/jan-logo.png (the waving hand) until the app mounts,
    # and two other places use the same file. One image replaces all three.
    shutil.copyfile(icons / 'icon.png', JAN / 'web-app/public/images/jan-logo.png')
    replace_once(html, '<img src="/images/jan-logo.png" alt="Jan Logo" data-tauri-drag-region />',
                 f'<img src="/images/jan-logo.png" alt="{NAME}" data-tauri-drag-region />')
    replace_once(html, 'Booting up Jan…', f'Starting {NAME}…')
    replace_once(html, '        animation: wave 2s ease-in-out 2.5s infinite;\n',
                 '        animation: none;   /* strixllama: an owl does not wave */\n')
    # The name at the top of the sidebar - the native title bar is hidden behind Jan's own window
    # chrome, so this is the name people actually see.
    sidebar = JAN / 'web-app/src/components/left-sidebar/index.tsx'
    replace_once(sidebar, '<span className="ml-2 font-medium font-studio">Jan</span>', '<SidebarBrand />')
    replace_once(sidebar, '<span className="mr-2 font-medium font-studio">Jan</span>', '<SidebarBrand />')
    # The download tray beside it managed Hub models and engine backends, neither of which this
    # build fetches; models come from the catalog on disk.
    replace_once(sidebar, "              {isLeftPanelOpen && <DownloadManagement />}\n",
                 "              {/* strixllama: no download tray - nothing here is downloaded */}\n")
    text = sidebar.read_text(encoding='utf-8')
    text = text.replace("import { DownloadManagement } from '@/containers/DownloadManegement'\n", "", 1)
    if text.count('isLeftPanelOpen') == 1:   # only its declaration is left, and the app builds with noUnusedLocals
        text = (text.replace("  const { open: isLeftPanelOpen } = useLeftPanel()\n", "", 1)
                    .replace("import { useLeftPanel } from '@/hooks/useLeftPanel'\n", "", 1))
    sidebar.write_text(text, encoding='utf-8', newline='\n')
    # ...and its twin in the page header, shown beside the sidebar toggle while the sidebar is closed
    header = JAN / 'web-app/src/containers/HeaderPage.tsx'
    replace_once(header, "            <DownloadManagement />\n", "            {/* strixllama: no download tray */}\n")
    replace_once(header, "import { DownloadManagement } from '@/containers/DownloadManegement'\n",
                 "// strixllama: no download tray, so no DownloadManagement\n")
    # Jan capitalises a provider it has no title for: give ours its name.
    replace_once(JAN / 'web-app/src/lib/utils.ts', "    case 'llamacpp':\n      return 'Llama.cpp'\n",
                 f"    case '{SLUG}':\n      return '{NAME}'\n    case 'llamacpp':\n      return 'Llama.cpp'\n")
    # Two cards Jan shows a fresh install: "download Jan V3.5 for your device" fetches a model for
    # Jan's engine, which this build never loads, and the analytics consent asks about telemetry
    # that is not configured (no PostHog key) and would go to Jan's project if it were.
    root = JAN / 'web-app/src/routes/__root.tsx'
    for line in ("import { useAnalytic } from '@/hooks/useAnalytic'\n",
                 "import { PromptAnalytic } from '@/containers/analytics/PromptAnalytic'\n",
                 "import { useJanModelPrompt } from '@/hooks/useJanModelPrompt'\n",
                 "import { PromptJanModel } from '@/containers/PromptJanModel'\n",
                 "  const { productAnalyticPrompt } = useAnalytic()\n",
                 "  const { showJanModelPrompt } = useJanModelPrompt()\n",
                 "        {productAnalyticPrompt && <PromptAnalytic />}\n",
                 "        {showJanModelPrompt && <PromptJanModel />}\n"):
        text = root.read_text(encoding='utf-8')
        if line in text:
            root.write_text(text.replace(line, '', 1), encoding='utf-8', newline='\n')
    print('  brand: %d icons, productName=%s, binary=%s, identifier=%s' % (copied, NAME, SLUG, data['identifier']))


import re

# The product name wherever a locale string names the product. Not \b: Japanese and Chinese run
# straight into the word, and \w counts their characters as word characters.
PRODUCT_WORD = re.compile(r'(?<![A-Za-z])Jan(?![A-Za-z])')
CREDITS = {
    'en': (f"{NAME} is a build of Jan by Menlo Research (Apache-2.0), with its own inference "
           "runtime and management pages in place of Jan's engines and providers.",
           "It runs on a pwilkin branch of llama.cpp, TheRock ROCm and Tauri. NOTICE.md in the "
           "repository lists every licence."),
    'zh-CN': (f"{NAME} 基于 Menlo Research 的 Jan（Apache-2.0）构建，用自己的推理运行时和管理页面"
              "取代了 Jan 的引擎与模型提供商。",
              "底层依赖 llama.cpp 的 pwilkin 分支、TheRock ROCm 与 Tauri。完整许可见仓库中的 NOTICE.md。"),
}


def brand_text():
    """The name where the app says it: the default assistant, the credits, every locale string.

    Attribution is the one thing not renamed. The credits say what this is built on rather than
    claiming Jan's sentence about its own team, and the other strings that describe Jan itself
    (documentation, release notes, GitHub) belong to cards converge_settings() already removed.
    """
    # The default assistant: seeded by the assistant extension on first run, and the web app's
    # own fallback when no extension answers. Both carry the name and a sentence about Jan.
    sentence = re.compile(r"Jan is a helpful desktop assistant that can reason through complex tasks "
                          r"and use tools to complete them on the user.s behalf\.")
    # a curly apostrophe survives both the single- and the double-quoted string it lands in
    description = ("A local assistant that reasons through complex tasks and uses tools to "
                   "complete them on the user’s behalf.")
    # The default instructions end with "Current date: {{current_date}}", which Jan renders into the system prompt -
    # the first tokens of every conversation. At local midnight the rendered date changes, and with it every cached
    # prefix: a 51K-token conversation continued after midnight shared 226 tokens with its stored state and was
    # processed again from the start (44 s). The date line goes; an assistant a user wrote keeps whatever it asks for.
    date_line = "\n\nCurrent date: {{current_date}}`"
    # Jan's default parameters also set a repeat penalty of 1.12. Jan dropped it for our provider (an OpenAI-compatible
    # endpoint gets no llama.cpp-only keys), so the model always ran without one; now that the panel's settings are
    # sent, it would apply - and on this model it costs MTP a tenth of its acceptance and a tenth of the speed
    # (Chinese prose, 400 tokens, two seeds: acceptance 0.49/0.54 -> 0.40/0.44, 32.3/33.9 -> 28.9/30.3 tok/s). The
    # default goes, here and in saved assistants and conversations; a penalty someone sets is sent as set.
    for path in (JAN / 'extensions/assistant-extension/src/index.ts',
                 JAN / 'web-app/src/hooks/useAssistant.ts'):
        text = path.read_text(encoding='utf-8')
        new = sentence.sub(description, text.replace("name: 'Jan',", f"name: '{NAME}',", 1)
                           .replace("avatar: '👋',", "avatar: '🦉',", 1)
                           .replace(date_line, "`")
                           .replace("          repeat_penalty: 1.12,\n", "")
                           .replace("      repeat_penalty: 1.12,\n", ""))
        if new != text:
            path.write_text(new, encoding='utf-8', newline='\n')
    # conversations keep their own copy of the assistant's parameters (thread.json), which the chat sends
    replace_once(JAN / 'web-app/src/hooks/useThreads.ts', "  setThreads: (threads) => {\n",
                 "  setThreads: (threads) => {\n"
                 "    // strixllama: Jan's default repeat penalty, which the model never ran with (see brand_text in apply.py)\n"
                 "    for (const t of threads) {\n"
                 "      const a = t.assistants?.[0] as { id?: string; parameters?: Record<string, unknown> } | undefined\n"
                 "      if (a?.id === 'jan' && a.parameters?.repeat_penalty === 1.12) {\n"
                 "        delete a.parameters.repeat_penalty\n"
                 "      }\n"
                 "    }\n")
    # ...and an assistant.json a Jan build wrote before the rename still says Jan: rename it as
    # it is read, in the store, so an existing data directory shows the same name as a new one.
    store = JAN / 'web-app/src/hooks/useAssistant.ts'
    replace_once(store, """  setAssistants: (assistants) => {
    if (assistants) {
      assistants.forEach((a) => (a.id = a.id?.toString())) // new String("id") !== "id"
""", """  setAssistants: (assistants) => {
    if (assistants) {
      assistants.forEach((a) => (a.id = a.id?.toString())) // new String("id") !== "id"
      // strixllama: the default assistant as a Jan build wrote it keeps Jan's name and wave on disk
      assistants.forEach((a) => {
        if (a.id === 'jan' && a.name === 'Jan') {
          a.name = '""" + NAME + """'
          a.description = '""" + description + """'
        }
        if (a.id === 'jan' && a.avatar === '👋') {
          a.avatar = '🦉'
        }
        // and the date line Jan's default instructions ended with (see brand_text: it changed the prompt's first
        // tokens at every midnight, so no conversation's cache survived the day)
        if (a.id === 'jan' && typeof a.instructions === 'string') {
          a.instructions = a.instructions.replace(/\\s*Current date: \\{\\{\\s*current_date\\s*\\}\\}\\s*$/, '')
        }
        // and Jan's default repeat penalty (see brand_text)
        if (a.id === 'jan' && a.parameters?.repeat_penalty === 1.12) {
          delete a.parameters.repeat_penalty
        }
      })
""")

    # The update prompt's release notes come from GitHub's list of releases: this project's, not Jan's
    replace_once(JAN / 'web-app/src/hooks/useReleaseNotes.ts', "'https://api.github.com/repos/janhq/jan/releases'",
                 f"'https://api.github.com/repos/rulith-dev/{SLUG}/releases'")
    # Installing an update replaces runtime\bin\hip, which the model server holds open: stop it first, through
    # the manager, so the conversations in memory are written to the disk tier as on any unload. The setup's
    # pre-install hook stops one that is still running (a manual install over a running app).
    updater = JAN / 'web-app/src/hooks/useAppUpdater.ts'
    replace_once(updater, "  const downloadAndInstallUpdate = useCallback(async () => {\n",
                 "  const downloadAndInstallUpdate = useCallback(async () => {\n"
                 "    // strixllama: the update replaces the runtime the model server runs from\n"
                 "    await stopModelServer()\n")
    replace_once(updater, "import { getServiceHub } from '@/hooks/useServiceHub'\n",
                 "import { getServiceHub } from '@/hooks/useServiceHub'\n"
                 "import { stopModelServer } from '@/components/strixllama/status'\n")
    # The download's state lived in the instance of the hook that started it (the update prompt), so the sidebar's
    # instance never saw it: the panel stayed on "New version", and the prompt said "Downloading..." with no number
    # while a 122 MB installer came in at 30 KB/s. Every change now reaches the other instances too (Jan's own
    # onAppUpdateStateSync), the progress at most four times a second, and the prompt shows the percentage.
    replace_once(updater, "      let downloaded = 0\n",
                 "      let downloaded = 0\n"
                 "      let lastSync = 0\n")
    replace_once(updater, "        isDownloading: true,\n      }))\n",
                 "        isDownloading: true,\n      }))\n"
                 "      syncStateToOtherInstances({ isDownloading: true, downloadProgress: 0, downloadedBytes: 0 })\n")
    replace_once(updater, "            console.log(`Started downloading ${contentLength} bytes`)\n",
                 "            console.log(`Started downloading ${contentLength} bytes`)\n"
                 "            syncStateToOtherInstances({ totalBytes: contentLength })\n")
    replace_once(updater, "            console.log(`Downloaded ${downloaded} from ${contentLength}`)\n",
                 "            if (Date.now() - lastSync > 250) {\n"
                 "              lastSync = Date.now()\n"
                 "              syncStateToOtherInstances({ isDownloading: true, downloadProgress: progress, downloadedBytes: downloaded })\n"
                 "            }\n")
    replace_once(updater, "              isDownloading: false,\n              downloadProgress: 1,\n            }))\n",
                 "              isDownloading: false,\n              downloadProgress: 1,\n            }))\n"
                 "            syncStateToOtherInstances({ isDownloading: false, downloadProgress: 1 })\n")
    replace_once(updater, "        isDownloading: false,\n      }))\n\n      // Emit app update download error event\n",
                 "        isDownloading: false,\n      }))\n"
                 "      syncStateToOtherInstances({ isDownloading: false })\n\n      // Emit app update download error event\n")
    replace_once(JAN / 'web-app/src/containers/dialogs/AppUpdater.tsx',
                 "                    {updateState.isDownloading\n                      ? t('updater:downloading')\n",
                 "                    {updateState.isDownloading\n"
                 "                      ? `${t('updater:downloading')} ${Math.round(updateState.downloadProgress * 100)}%`\n")
    # Telemetry: there is no key, so nothing is collected, and the consent card would be asking
    # on Jan's behalf. Gated on a constant rather than cut, for the same reason as SHOW_PROVIDERS.
    privacy = JAN / 'web-app/src/routes/settings/privacy.tsx'
    replace_once(privacy, "  return (\n", "  const SHOW_ANALYTICS = false   // strixllama: no telemetry is configured\n  return (\n")
    card = """            <Card
              header={
                <div className="flex items-center justify-between mb-4">
                  <h1 className="font-medium text-foreground text-base">
                    {t('settings:privacy.analytics')}"""
    text = privacy.read_text(encoding='utf-8')
    if 'SHOW_ANALYTICS && (' not in text:
        if text.count(card) != 1:
            raise RuntimeError(f'Upstream source changed: {privacy}')
        start = text.index(card)
        end = text.index('            </Card>\n', start) + len('            </Card>\n')
        text = text[:start] + '            {SHOW_ANALYTICS && (\n' + text[start:end] + '            )}\n' + text[end:]
        privacy.write_text(text, encoding='utf-8', newline='\n')

    # Every locale: the product's name in strings, the credits replaced (English and Chinese
    # written here; the others drop the keys and fall back to English, which is Jan's own
    # fallback rule) rather than reworded into a claim about who built Jan.
    def walk(node, locale):
        if isinstance(node, dict):
            for key in list(node):
                if key in ('creditsDesc1', 'creditsDesc2'):
                    if locale in CREDITS:
                        node[key] = CREDITS[locale][int(key[-1]) - 1]
                    else:
                        del node[key]
                else:
                    node[key] = walk(node[key], locale)
            return node
        if isinstance(node, list):
            return [walk(x, locale) for x in node]
        if isinstance(node, str):
            return PRODUCT_WORD.sub(NAME, node)
        return node
    for path in sorted((JAN / 'web-app/src/locales').glob('*/*.json')):
        if path.name == 'strixllama.json':
            continue
        original = path.read_text(encoding='utf-8')
        data = walk(json.loads(original), path.parent.name)
        text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        if json.loads(text) != json.loads(original):
            path.write_text(text, encoding='utf-8', newline='\n')


def converge_settings():
    """Strip the settings surfaces this build has no path through.

    Everything here is either dead (it drives Jan's own llama.cpp engine or the Hub, neither of
    which this build ever loads) or actively wrong (the updater would replace a custom binary, and
    the Resources/Community links point at upstream Jan rather than at this fork).
    """
    general = JAN / 'web-app/src/routes/settings/general.tsx'
    # Resources + Community cards: upstream Jan's docs, release notes, GitHub and Discord
    cut_lines(general, 554, 643, '{/* Resources */}', '</Card>')
    # HuggingFace token: only ever used by the Hub downloads that the sidebar no longer exposes
    cut_lines(general, 473, 551, '<CardItem', '/>')
    # Jan CLI install/uninstall: serves models through Jan's engine, which is never loaded here
    cut_lines(general, 409, 439, '{IS_TAURI && (', ')}')
    # and the state, effects and handlers the two removed cards were the only readers of.
    # These are anchor-based: line numbers shift as soon as the first cut lands.
    cut_block(general, "  useEffect(() => {\n    if (!IS_TAURI) return")   # CLI status probe
    cut_block(general, "  const handleInstallCli = async () => {")
    cut_block(general, "  const handleUninstallCli = async () => {")
    drop_declarations(general, ('IconBrandDiscord', 'IconBrandGithub', 'IconExternalLink',
                                'Input', 'TOKEN_VALIDATION_TIMEOUT_MS',
                                'huggingfaceToken', 'setHuggingfaceToken', 'invoke'))
    for decl in ('  const [isValidatingToken, setIsValidatingToken] = useState(false)\n',
                 '  const [cliInstalled, setCliInstalled] = useState<boolean | null>(null)\n',
                 '  const [cliPath, setCliPath] = useState<string | null>(null)\n',
                 '  const [isCliLoading, setIsCliLoading] = useState(false)\n'):
        general.write_text(general.read_text(encoding='utf-8').replace(decl, '', 1),
                           encoding='utf-8', newline='\n')

    # The model picker in the chat header lists every active provider, so Jan's own llama.cpp
    # engine and the remote APIs (Anthropic, Azure, Gemini, ...) show up there even after the
    # settings section is gone. Filter at the source, so all five uses in that file follow.
    # Exclusions rather than a name allow-list: a provider you configured yourself keeps working
    # whatever you called it.
    picker = JAN / 'web-app/src/containers/DropdownModelProvider.tsx'
    replace_once(picker, """    providers,
    getProviderByName,""", """    providers: allProviders,
    getProviderByName,""")
    unmemoized = """  // strixllama: this build serves one local endpoint from tools/manager.py. Jan's bundled
  // engines never load a model here, and the remote APIs are not what it is for.
  const providers = allProviders.filter(
    (p) =>
      p.provider !== 'llamacpp' &&
      p.provider !== 'mlx' &&
      !predefinedProviders.some((e) => e.provider.includes(p.provider))
  )
"""
    # Memoised, and it matters: the list is a dependency of the effect that selects a thread's
    # model. A fresh array every render re-ran that effect, which set state, which rendered again -
    # React error #185 the moment any existing thread was opened.
    memoized = """  // strixllama: this build serves one local endpoint from tools/manager.py. Jan's bundled
  // engines never load a model here, and the remote APIs are not what it is for. Memoised because
  // the list feeds the effect that selects a thread's model; a new array per render loops it.
  const providers = useMemo(
    () =>
      allProviders.filter(
        (p) =>
          p.provider !== 'llamacpp' &&
          p.provider !== 'mlx' &&
          !predefinedProviders.some((e) => e.provider.includes(p.provider))
      ),
    [allProviders]
  )
"""
    text = picker.read_text(encoding='utf-8')
    if memoized not in text:
        if unmemoized in text:             # a tree an earlier apply.py left with the looping array
            text = text.replace(unmemoized, memoized, 1)
        else:
            anchor = "  const [displayModel, setDisplayModel] = useState<string>('')"
            if text.count(anchor) != 1:
                raise RuntimeError(f'Upstream source changed: {picker}')
            text = text.replace(anchor, memoized + anchor, 1)
        picker.write_text(text, encoding='utf-8', newline='\n')
    # With Jan's engine filtered out, its "first llamacpp model" fallback for a new chat never
    # fires and the picker opens on "select a model". Fall back to the first provider that has any.
    replace_once(picker, """          const llamacppProvider = providers.find(
            (p) => p.provider === 'llamacpp' && p.active && p.models.length > 0
          )""", """          const llamacppProvider = providers.find(
            (p) => p.active && p.models.length > 0 // strixllama: the local provider, not Jan's engine
          )""")
    replace_once(picker, """            selectModelProvider('llamacpp', firstModel.id)
            setLastUsedModel('llamacpp', firstModel.id)""",
                 """            selectModelProvider(llamacppProvider.provider, firstModel.id)
            setLastUsedModel(llamacppProvider.provider, firstModel.id)""")
    # A thread keeps the id of the model that answered it, and Jan clears the choice when that model is
    # gone - but the local server runs one model at a time, so after a switch every older thread opened
    # on "select a model". A thread of ours goes on with whichever model is loaded now.
    replace_once(picker, """        if (!checkModelExists(model.provider, model.id)) {
          selectModelProvider('', '')
        }""", """        if (!checkModelExists(model.provider, model.id)) {
          // strixllama: the local server's current model, for a thread it answered with another
          const loaded = model.provider === 'strixllama' ? getProviderByName('strixllama')?.models[0] : undefined
          selectModelProvider(loaded ? 'strixllama' : '', loaded?.id ?? '')
        }""")
    # The gear beside the provider opens Jan's provider page: base URL, API keys, a Delete button
    # that would take the only provider with it. For ours it opens the Configuration page instead.
    replace_once(picker, """                            navigate({
                              to: route.settings.providers,
                              params: { providerName: providerInfo.provider },
                            })""", """                            if (providerInfo.provider === 'strixllama') {
                              navigate({ to: '/strixllama/configuration' as '/strixllama/models' })
                            } else {
                              navigate({
                                to: route.settings.providers,
                                params: { providerName: providerInfo.provider },
                              })
                            }""")

    # The provider page is still reachable by URL: keep it from deleting the one provider.
    replace_once(JAN / 'web-app/src/routes/settings/providers/$providerName.tsx',
                 "                <DeleteProvider provider={provider} />\n",
                 "                {provider?.provider !== 'strixllama' && <DeleteProvider provider={provider} />}\n")

    menu = JAN / 'web-app/src/containers/SettingsMenu.tsx'
    text = menu.read_text(encoding='utf-8')
    for entry in ('local_api_server',   # we serve on 8080 from tools/manager.py, not from Jan
                  'https_proxy',        # only mattered for the remote providers dropped below
                  'hardware'):          # GPU detection for Jan's engine
        block = f"""    {{
      title: 'common:{entry}',
      route: route.settings.{entry},"""
        if block not in text:
            continue
        start = text.index(block)
        end = text.index('    },\n', start) + len('    },\n')
        text = text[:start] + text[end:]
    # only the strixllama provider is real here: Jan's llama.cpp never loads a model in this build and
    # the remote providers are not what this fork is for
    old_filter = """  const activeProviders = providers.filter((provider) => {
    if (!provider.active) return false
    if (!IS_MACOS && provider.provider === 'mlx') return false
    return true
  })"""
    new_filter = """  const activeProviders = providers.filter((provider) => {
    if (!provider.active) return false
    // strixllama: this build serves every model from tools/manager.py, so Jan's own engines and the
    // remote APIs have nothing to configure. Their routes still resolve if visited directly.
    return provider.provider.toLowerCase().includes('strixllama')
  })"""
    if old_filter in text:
        text = text.replace(old_filter, new_filter, 1)
        text = text.replace("""  const hiddenProviders = providers.filter((provider) => {
    if (provider.active) return false
    if (!IS_MACOS && provider.provider === 'mlx') return false
    return true
  })""", """  const hiddenProviders: typeof providers = []""", 1)

    # "Model providers" is a concept this build does not have: one local server, started by
    # tools/manager.py, and nothing to choose between. The whole section goes.
    #
    # It is gated on a constant rather than deleted because web-app/tsconfig.app.json sets
    # noUnusedLocals: deleting the only use of createProvider, AddProviderDialog and IconPlus turns
    # three unused declarations into build errors, and chasing those down removes things that are
    # genuinely upstream Jan's. The bundler drops the dead branch anyway.
    gate = """  const SHOW_PROVIDERS = false   // strixllama: one local backend, nothing to choose between
  const activeProviders"""
    if 'const SHOW_PROVIDERS' not in text:
        text = text.replace("  const activeProviders", gate, 1)
    section_open = """          {/* Model Providers section */}
          <div className="mt-4">"""
    section_close = """            </div>
            <div className="m-3" />
          </div>"""
    if section_open in text and text.count(section_close) == 1:
        text = text.replace(section_open, """          {/* Model Providers section - see converge_settings() in integrations/jan/apply.py */}
          {SHOW_PROVIDERS && (
          <div className="mt-4">""", 1)
        text = text.replace(section_close, section_close + """
          )}""", 1)
    elif 'SHOW_PROVIDERS && (' not in text:
        raise SystemExit('apply.py: the Model Providers section in SettingsMenu.tsx has moved')
    menu.write_text(text, encoding='utf-8', newline='\n')
    drop_declarations(menu, ('IconCircles', 'IconCpu', 'IconWorld'))


def drop_nav_entries(nav):
    """Hide the Jan features this build has no path through.

    The sidebar should describe what the app can actually do. Hub downloads models for Jan's own
    engine, which this build never loads - every model comes from the strixllama manager instead - and
    agent chats and projects are Jan surfaces none of the strixllama work touches. The routes stay
    reachable by URL; only the entry points go, leaving chat, search, the three strixllama pages and
    settings.
    """
    text = nav.read_text(encoding='utf-8')
    for title in ("common:newAgentChat", "common:projects.new", "common:hub", "common:settings"):
        marker = f"    title: '{title}',"
        if marker not in text:
            continue                       # already dropped
        start = text.rindex("  {\n", 0, text.index(marker))
        depth, i = 0, start
        while i < len(text):               # walk to the entry's matching brace
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = text.index('\n', text.index(',', i)) + 1
        text = text[:start] + text[end:]

    # the entries took their icons and callbacks with them, and the web app builds with
    # noUnusedLocals. The *Handle types are still referenced by the NavMainIcon union, so drop only
    # the icon values from each import.
    for old, new in (
            ("import {\n  FolderPlusIcon,\n  type FolderPlusIconHandle,\n} from '@/components/animated-icon/folder-plus'",
             "import { type FolderPlusIconHandle } from '@/components/animated-icon/folder-plus'"),
            ("import { BlocksIcon, type BlocksIconHandle } from '../animated-icon/blocks'",
             "import { type BlocksIconHandle } from '../animated-icon/blocks'"),
            ("import {\n  BotIcon,\n  type BotIconHandle,\n} from '@/components/animated-icon/bot'",
             "import { type BotIconHandle } from '@/components/animated-icon/bot'"),
            ("import {\n  SettingsIcon,\n  type SettingsIconHandle,\n} from '@/components/animated-icon/settings'",
             "import { type SettingsIconHandle } from '@/components/animated-icon/settings'")):
        text = text.replace(old, new, 1)
    for param in ("onNewProject", "onJanClaw"):
        text = text.replace(f"  {param}: () => void,\n", f"  _{param}: () => void,\n", 1)
        text = text.replace(f"  {param}: () => void\n", f"  _{param}: () => void\n", 1)
    nav.write_text(text, encoding='utf-8', newline='\n')


def cut_span(path, start_marker, end_marker, keep_end):
    """Delete from start_marker up to end_marker (kept when keep_end), idempotent."""
    text = path.read_text(encoding='utf-8')
    if start_marker not in text:
        return False                       # already cut
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    if not keep_end:
        end += len(end_marker)
    path.write_text(text[:start] + text[end:], encoding='utf-8', newline='\n')
    return True


def drop_integrations():
    """Remove the Integrations section (MCP servers, Claude Code) from the settings menu.

    This build only ever talks to the local strixllama server; the agent integrations are Jan
    surfaces none of the strixllama work touches. The routes stay reachable by URL.
    """
    menu = JAN / 'web-app/src/containers/SettingsMenu.tsx'
    cut_span(menu, "  const integrationSettings = [\n", "  ]\n", keep_end=False)
    # prefix only: converge_settings() rewrites the rest of that comment line, and runs before this
    cut_span(menu, "          {/* Integrations section */}\n", "          {/* Model Providers section", keep_end=True)
    drop_declarations(menu, ('IconTopologyStar3',))


def documents():
    """Documents in the chat (issue #4): attached with "Add documents or files" or dropped on the chat box.

    Jan reads a document either into the message or into a vector store through an embedding model its own llama.cpp
    engine serves, and this build leaves that engine out - so a document goes into the message whole, from wherever it
    came (a project's conversations too). So it needs nothing of the model: Jan offers the menu item only to a model
    with the 'tools' capability, and 0.3.6 relied on StrixLlamaSync giving it - which a model listed by 0.3.5 got only
    once a model had loaded, so after the update the item stayed grey. It is offered always now, and drops the same.
    components/strixllama/attachments.ts reads the text and refuses, with the reason, a
    file that gives none or that the model's context cannot hold. Jan's chat box takes only images, audio and video
    when dropped: an HTML5 drop carries a file's bytes but no path, so a dropped document goes to
    strixllama_parse_dropped (strixllama.rs), which reads it with the parser the file picker reaches - made public here
    for that.
    """
    replace_once(JAN / 'src-tauri/plugins/tauri-plugin-rag/src/lib.rs', 'mod parser;\n', 'pub mod parser;\n')

    chat = JAN / 'web-app/src/containers/ChatInput.tsx'
    replace_once(chat, "import { toast } from 'sonner'\n",
                 "import { toast } from 'sonner'\n"
                 "import { fileExtension, parseDroppedDocument, readDocument } from '@/components/strixllama/attachments'\n")
    replace_once(chat, """            size,
            parseMode: parsePreference,
          })""", """            size,
            // strixllama: into the message; there is no embedding engine to index it
            parseMode: 'inline',
          })""")
    replace_once(chat, """                    <DropdownMenuItem
                      onClick={handleAttachDocsIngest}
                      disabled={!selectedModel?.capabilities?.includes('tools')}
                    >""", """                    <DropdownMenuItem
                      onClick={handleAttachDocsIngest}
                    >""")
    replace_once(chat, "  const dropAcceptsAnything = hasMmproj || audioSupported || videoSupported\n", """\
  // strixllama: a document dropped on the chat box is read as it lands and goes into the message, like one the file
  // picker attached (Jan's chat box takes only images, audio and video). Text in the message needs nothing of the
  // model, so both are always offered
  const docsSupported = true
  const attachDroppedDocuments = async (files: File[]) => {
    if (!attachmentsEnabled) {
      toast.info(t('strixllama:attach.disabled'))
      return
    }
    const limit = typeof maxFileSizeMB === 'number' && maxFileSizeMB > 0 ? maxFileSizeMB : undefined
    for (const file of files) {
      if (limit !== undefined && file.size > limit * 1024 * 1024) {
        toast.error(t('strixllama:attach.tooLarge', { name: file.name, limit }))
        continue
      }
      // a dropped file has no path: this finds its chip again, and tells a second drop of the same file apart
      const key = `drop:${file.name}:${file.size}:${file.lastModified}`
      let duplicate = false
      setAttachmentsForThread(attachmentsKey, (prev) => {
        duplicate = prev.some((a) => a.contentHash === key)
        return duplicate
          ? prev
          : [...prev, { type: 'document' as const, name: file.name, fileType: fileExtension(file.name), size: file.size,
              contentHash: key, processing: true }]
      })
      if (duplicate) continue
      try {
        const text = await readDocument(file.name, () => parseDroppedDocument(file))
        setAttachmentsForThread(attachmentsKey, (prev) =>
          prev.map((a) => a.contentHash === key
            ? { ...a, processing: false, processed: true, injectionMode: 'inline' as const, inlineContent: text }
            : a))
      } catch (err) {
        setAttachmentsForThread(attachmentsKey, (prev) => prev.filter((a) => a.contentHash !== key))
        toast.error(err instanceof Error ? err.message : String(err))
      }
    }
  }

  const dropAcceptsAnything = hasMmproj || audioSupported || videoSupported || docsSupported
""")
    replace_once(chat, """    if (otherOnes.length > 0 && hasMmproj) {
      const dt = new DataTransfer()
      otherOnes.forEach((f) => dt.items.add(f))""", """    // strixllama: pictures go to the image path, every other file is a document (attachDroppedDocuments)
    const imageOnes = otherOnes.filter((f) => f.type.startsWith('image/'))
    const docOnes = otherOnes.filter((f) => !f.type.startsWith('image/'))
    if (imageOnes.length > 0 && !hasMmproj) {
      toast.info(t('strixllama:attach.imagesOff'))
    }
    if (docOnes.length > 0 && docsSupported) {
      void attachDroppedDocuments(docOnes)
    }
    if (imageOnes.length > 0 && hasMmproj) {
      const dt = new DataTransfer()
      imageOnes.forEach((f) => dt.items.add(f))""")
    processing = JAN / 'web-app/src/lib/attachmentProcessing.ts'
    replace_once(processing, "import { toast } from 'sonner'\n",
                 "import { toast } from 'sonner'\n"
                 "import { readDocument } from '@/components/strixllama/attachments'\n")
    replace_once(processing, """      // Project files always use embeddings, never inline
      if (projectId) {
        targetMode = 'embeddings'
      }

      const canInline = !projectId && targetPreference !== 'embeddings' && !!doc.path

      if (canInline) {
        try {
          parsedContent = await serviceHub
            .rag()
            .parseDocument?.(doc.path!, doc.fileType)
        } catch (err) {
          console.warn(`Failed to parse ${doc.name} for inline use`, err)
        }
      }""", """      // strixllama: into the message, in a project's conversations too - there is no embedding engine to index it
      // with. A file that gives no text, or that the context cannot hold, is refused with the reason.
      const canInline = targetPreference !== 'embeddings' && !!doc.path

      if (canInline) {
        parsedContent = await readDocument(doc.name, () =>
          serviceHub.rag().parseDocument?.(doc.path!, doc.fileType) ?? Promise.resolve(undefined))
      }""")

    transport = JAN / 'web-app/src/lib/custom-chat-transport.ts'
    # with attachments, Jan's system prompt points the model at retrieval tools for them; here their text is in the message
    replace_once(transport, """      'attached to that turn (file_id, name, type, size, chunk count, mode).',
      'Use the available retrieval tools with those file_ids when their',
      'contents are relevant to the request.',""", """      'attached to that turn (file_id, name, type, size, chunk count, mode).',
      // strixllama: every document is in the message (mode: inline); there are no retrieval tools in this build
      'The text of each file follows in the same message, after a line',
      '"File: <name>".',""")
    # the rest of the attachment settings are the embedding engine's: chunking, retrieval, search
    replace_once(JAN / 'web-app/src/routes/settings/attachments.tsx', "              {defs.map((d) => {\n",
                 "              {/* strixllama: the others (chunks, retrieval, search) are the embedding engine's, not in this build */}\n"
                 "              {defs.filter((d) => d.key === 'enabled' || d.key === 'max_file_size_mb').map((d) => {\n")



def chat_width():
    """The chat's column as wide as the model pages' (StrixLlamaPage.tsx: max-w-[1280px], px-6). Jan sizes it as a
    share of the window instead - 4/5, then 4/6 from the xl breakpoint - so on a wide window the conversation ran wider
    than Configuration and Logs beside it, and on a narrower one it was the narrower of the two. The conversation, the
    chat box under it, the new-chat page and a project's page take the same column now.
    """
    col = 'mx-auto w-full max-w-[1280px] px-6'
    thread = JAN / 'web-app/src/routes/threads/$threadId.tsx'
    replace_once(thread, "              className={cn('mx-auto w-full md:w-4/5 xl:w-4/6')}\n",
                 "              className={cn('" + col + "')}\n")
    replace_once(thread, '        <div className="py-4 mx-auto w-full md:w-4/5 xl:w-4/6">\n',
                 '        <div className="py-4 ' + col + '">\n')
    replace_once(JAN / 'web-app/src/routes/index.tsx', "            'mx-auto w-full md:w-4/5 xl:w-4/6 -mt-20',\n",
                 "            '" + col + " -mt-20',\n")
    replace_once(JAN / 'web-app/src/routes/project/$projectId.tsx', '        <div className="mx-auto w-full md:w-4/5 xl:w-4/6">\n',
                 '        <div className="' + col + '">\n')


def compaction():
    """A conversation that outgrows the model's context is folded into a summary instead of failing
    (components/strixllama/compact.ts says how and why Jan's own trimmer is not used). The transport calls it for the
    local provider once the system prompt is built: it gets the window to send - the messages after the last fold,
    with the summary in the system prompt - and, when a fold is due, asks the model for the summary with the request as
    it went last time (same system prompt, same conversion, same tool definitions without their execute, so the server
    finds it cached) plus the instruction, thinking off.
    """
    t = JAN / 'web-app/src/lib/custom-chat-transport.ts'
    replace_once(t, "  convertToModelMessages,\n  streamText,\n", "  convertToModelMessages,\n  generateText,\n  streamText,\n")
    replace_once(t, "import { ModelFactory } from './model-factory'\n",
                 "import { ModelFactory } from './model-factory'\n"
                 "import { compactConversation, INSTRUCTION } from '@/components/strixllama/compact'\n")
    replace_once(t, "    const effectiveSystem =\n      typeof rawSystem === 'string' && rawSystem.trim().length > 0\n",
                 "    let effectiveSystem =\n      typeof rawSystem === 'string' && rawSystem.trim().length > 0\n")
    replace_once(t, """    // Auto-trim or auto-compact conversation history when max_context_tokens is configured
    let effectiveMessages = messagesToConvert
""", """    // strixllama: a conversation past the context is folded into a summary first (components/strixllama/compact.ts)
    let strixWindow = messagesToConvert
    if (providerId === 'strixllama' && this.threadId) {
      const summaryTools =
        Object.keys(this.tools).length > 0 &&
        (selectedModel?.capabilities?.includes('tools') ?? this.modelSupportsTools)
          ? Object.fromEntries(Object.entries(this.tools).map(([k, tool]) => [k, { ...tool, execute: undefined }]))
          : undefined
      const compacted = await compactConversation({
        threadId: this.threadId,
        messages: messagesToConvert,
        system: effectiveSystem,
        summarize: async (msgs, system) => {
          const vision = selectedModel?.capabilities?.includes('vision') ?? false
          const summaryModel = await ModelFactory.createModel(
            modelId,
            useModelProvider.getState().getProviderByName(providerId) ?? provider,
            {
              ...extractModelSamplingDefaults(selectedModel),
              ...inferenceParams,
              chat_template_kwargs: { enable_thinking: false },
            }
          )
          const asked: UIMessage[] = [
            ...msgs,
            { id: 'strix-compact', role: 'user', parts: [{ type: 'text', text: INSTRUCTION }] },
          ]
          const summaryMessages = await convertToModelMessages(
            coalesceMessagesForAlternation(
              resolveOrphanToolCalls(
                this.encodeVideoAttachments(
                  this.encodeAudioAttachments(
                    stripUnsupportedImageParts(this.mapUserInlineAttachments(asked), vision)
                  )
                )
              )
            )
          )
          const r = await generateText({
            model: summaryModel,
            system,
            messages: summaryMessages,
            tools: summaryTools,
            toolChoice: summaryTools ? 'auto' : undefined,
            maxOutputTokens: 2500,
            abortSignal: options.abortSignal,
          })
          return r.text
        },
      })
      strixWindow = compacted.messages
      effectiveSystem = compacted.system
    }

    // Auto-trim or auto-compact conversation history when max_context_tokens is configured
    let effectiveMessages = strixWindow
""")
    replace_once(t, "        const compactResult = await compactMessages(\n          messagesToConvert,\n",
                 "        const compactResult = await compactMessages(\n          strixWindow,\n")
    replace_once(t, "        const trimResult = trimMessages(\n          messagesToConvert,\n",
                 "        const trimResult = trimMessages(\n          strixWindow,\n")


def concurrent_tools():
    """Tool calls in several conversations at once. The thread route is one component reused across threads, and each
    thread's Chat keeps the callbacks it was created with, so the tool loop's AbortController - one useRef - was shared by
    every conversation: two answers that called tools at the same time overwrote and cleared each other's, and the one
    whose results came back last was never sent back to the model (it showed its sources and stopped; seen with two web
    searches 0.7 s apart). Leaving a thread also aborted its loop, so a search stopped when you switched chats. Now each
    thread has its own controller (toolLoops), a thread left while its tools run keeps going, and only a loop waiting for
    the user's approval is stopped on leaving, as before.
    """
    r = JAN / 'web-app/src/routes/threads/$threadId.tsx'
    replace_once(r, "import { useAutoScroll } from '@/hooks/useAutoScroll'\n",
                 "import { useAutoScroll } from '@/hooks/useAutoScroll'\n\n"
                 "// strixllama: the running tool loop of each thread (see concurrent_tools() in apply.py)\n"
                 "const toolLoops = new Map<string, AbortController>()\n")
    replace_once(r, """  // AbortController for cancelling tool calls
  const toolCallAbortController = useRef<AbortController | null>(null)
""", """  // strixllama: the tool loop's AbortController is per thread, in toolLoops above
""")
    replace_once(r, """      if (
        !toolCallAbortController.current ||
        toolCallAbortController.current?.signal.aborted
      ) {
        return false
      }
      return lastAssistantMessageIsCompleteWithToolCalls({ messages })
    },
    []
  )""", """      const toolLoop = toolLoops.get(threadId)
      if (!toolLoop || toolLoop.signal.aborted) {
        return false
      }
      return lastAssistantMessageIsCompleteWithToolCalls({ messages })
    },
    [threadId]
  )""")
    replace_once(r, """      toolCallAbortController.current = new AbortController()
      const signal = toolCallAbortController.current.signal
""", """      const toolLoop = new AbortController()
      toolLoops.set(threadId, toolLoop)
      const signal = toolLoop.signal
""")
    replace_once(r, """        sessionData.tools = []
        toolApprovalPromises.current.clear()
        toolCallAbortController.current = null
        useAppState.getState().setThreadBusy(threadId, false)
      })().catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Tool call error:', error)
        }
        sessionData.tools = []
        toolApprovalPromises.current.clear()
        toolCallAbortController.current = null
        useAppState.getState().setThreadBusy(threadId, false)
      })""", """        for (const t of sessionData.tools) toolApprovalPromises.current.delete(t.toolCallId)
        sessionData.tools = []
        if (toolLoops.get(threadId) === toolLoop) toolLoops.delete(threadId)
        useAppState.getState().setThreadBusy(threadId, false)
      })().catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Tool call error:', error)
        }
        for (const t of sessionData.tools) toolApprovalPromises.current.delete(t.toolCallId)
        sessionData.tools = []
        if (toolLoops.get(threadId) === toolLoop) toolLoops.delete(threadId)
        useAppState.getState().setThreadBusy(threadId, false)
      })""")
    replace_once(r, """  useEffect(() => {
    // Stable ref object (never reassigned) — capture for the cleanup closure.
    const approvalPromises = toolApprovalPromises.current
    return () => {
      toolCallAbortController.current?.abort()
      toolCallAbortController.current = null
      approvalPromises.clear()
      useToolApprovalRequests.getState().clearPendingForThread(threadId)
    }
  }, [threadId])""", """  useEffect(() => {
    return () => {
      // strixllama: a thread left while its tools run (a web search) goes on and answers in the background; one
      // waiting for the user's approval stops, as before - no one is there to give it
      const pending = useToolApprovalRequests.getState().pending ?? {}
      if (Object.values(pending).some((q) => q.threadId === threadId)) {
        toolLoops.get(threadId)?.abort()
        toolLoops.delete(threadId)
      }
      useToolApprovalRequests.getState().clearPendingForThread(threadId)
    }
  }, [threadId])""")


def web_tools():
    """Jan's web search, which this build's model never had: Jan offers web_search and web_fetch (tauri-plugin-websearch:
    Exa's keyless endpoint by default, Tavily or SearXNG in Settings > Web Search) only to a model with the 'tools'
    capability, and StrixLlamaSync now gives it. Jan has web search on unless it is turned off, so every chat would send
    the two tools and every search would go to a third party: here it starts off, and the globe in the chat box turns it
    on. It was already on (Jan's default) in every system prompt, as an instruction about tools the model was never
    given; while off, that instruction is not sent either.
    """
    config = JAN / 'web-app/src/hooks/useWebSearchConfig.ts'
    replace_once(config, "      webSearchEnabled: true,\n",
                 "      webSearchEnabled: false, // strixllama: off until the globe in the chat box turns it on\n")
    # a stored setting from before says true because Jan's default did: off once, too
    replace_once(config, """      name: localStorageKey.settingWebSearch,
      storage: createJSONStorage(() => backendStorage),""", """      name: localStorageKey.settingWebSearch,
      // strixllama: version 1 - web search starts off in this build (see its default above), stored settings included
      version: 1,
      migrate: (stored) => ({ ...(stored as WebSearchConfigState), webSearchEnabled: false }),
      storage: createJSONStorage(() => backendStorage),""")
    # 'tools' with 'vision' also brings the button of Jan's browser extension, which runs through MCP servers this build
    # does not set up (drop_integrations)
    replace_once(JAN / 'web-app/src/containers/ChatInput.tsx',
                 "                {!effectiveAgentMode && hasJanBrowserMCPConfig && modelSupportsBrowser && (\n",
                 "                {/* strixllama: Jan's browser extension runs through MCP, which this build does not set up */}\n"
                 "                {false && !effectiveAgentMode && hasJanBrowserMCPConfig && modelSupportsBrowser && (\n")


if __name__ == '__main__':
    main()
