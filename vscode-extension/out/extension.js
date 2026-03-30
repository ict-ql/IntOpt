"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
// ── Config helpers ───────────────────────────────────────
function cfg(key, fallback) {
    return vscode.workspace.getConfiguration("intopt").get(key, fallback);
}
function condaWrap(cmd) {
    const condaEnv = cfg("condaEnv", "");
    if (!condaEnv) {
        return cmd;
    }
    return `conda run --no-capture-output -n ${condaEnv} ${cmd}`;
}
function spawnWithConda(cmd, cwd, extraEnv) {
    return (0, child_process_1.spawn)("bash", ["--login", "-c", condaWrap(cmd)], {
        cwd, env: { ...process.env, ...extraEnv },
    });
}
function extractFuncNames(text) {
    const re = /define\s+.*?@(?:"([^"]+)"|([\w.$-]+))/g;
    const names = [];
    let m;
    while ((m = re.exec(text)) !== null) {
        names.push(m[1] || m[2]);
    }
    return names;
}
function runCmd(cmd, args) {
    return new Promise((resolve) => {
        const fullCmd = `${cmd} ${args.map(a => `'${a}'`).join(" ")}`;
        const proc = spawnWithConda(fullCmd);
        proc.on("close", (code) => resolve(code === 0));
        proc.on("error", () => resolve(false));
    });
}
function escHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function runStep(inputFile, outputDir, step, outputChannel) {
    const pythonPath = cfg("pythonPath", "python");
    const projectRoot = cfg("projectRoot", "/home/amax/yangz/intop");
    const configFile = cfg("configFile", path.join(projectRoot, "config", "config.yaml"));
    const args = [
        pythonPath,
        path.join(projectRoot, "src", "main.py"),
        "--mode", "step",
        "--step", String(step),
        "--input", inputFile,
        "--output", outputDir,
        "--config", configFile,
    ];
    const cmdStr = args.join(" ");
    return new Promise((resolve) => {
        const proc = spawnWithConda(cmdStr, path.join(projectRoot, "src"), { PYTHONUNBUFFERED: "1" });
        let allOutput = "";
        const logs = [];
        const handle = (data) => {
            const text = data.toString();
            allOutput += text;
            for (const line of text.split("\n")) {
                const t = line.trim();
                if (t) {
                    logs.push(t);
                    outputChannel.appendLine(t);
                }
            }
        };
        proc.stdout?.on("data", handle);
        proc.stderr?.on("data", handle);
        proc.on("close", () => {
            // Parse STEP_RESULT and STEP_RESULT_ANALYSIS from output
            const marker = "STEP_RESULT:";
            const analysisMarker = "STEP_RESULT_ANALYSIS:";
            let result = null;
            let analysisResult = null;
            for (const line of allOutput.split("\n")) {
                const idx = line.indexOf(marker);
                if (idx >= 0 && !line.includes(analysisMarker)) {
                    try {
                        result = JSON.parse(line.substring(idx + marker.length));
                    }
                    catch { /* ignore */ }
                }
                const aIdx = line.indexOf(analysisMarker);
                if (aIdx >= 0) {
                    try {
                        analysisResult = JSON.parse(line.substring(aIdx + analysisMarker.length));
                    }
                    catch { /* ignore */ }
                }
            }
            resolve({ result, analysisResult, logs });
        });
        proc.on("error", () => resolve({ result: null, analysisResult: null, logs }));
    });
}
// ── Chat-like Webview Panel ──────────────────────────────
const STEP_LABELS = [
    "Preparing input",
    "Step 1: Strategy Generation",
    "Step 2: Strategy Mapping",
    "Step 3: Analysis & Refinement Prompt",
    "Step 3b: LLM Strategy Refinement",
    "Step 4: LLM Realization",
    "Step 5: Post-processing",
    "Step 6: Verification",
    "Step 7: Performance Testing",
];
// Steps where we pause for user review/edit
const PAUSE_STEPS = new Set([1, 2, 3, 4, 5]);
class ChatPanel {
    constructor() {
        this.messages = [];
        this.resolveUserAction = null;
        this.currentStep = -1;
        this.panel = vscode.window.createWebviewPanel("intoptChat", "IntOpt", vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
        this.panel.webview.onDidReceiveMessage((msg) => {
            if (msg.type === "userAction" && this.resolveUserAction) {
                this.resolveUserAction(msg.action);
                this.resolveUserAction = null;
            }
        });
    }
    addSystem(html) {
        this.messages.push({ role: "system", html });
        this.render();
    }
    addAssistant(html) {
        this.messages.push({ role: "assistant", html });
        this.render();
    }
    addUser(html) {
        this.messages.push({ role: "user", html });
        this.render();
    }
    setStep(step) {
        this.currentStep = step;
    }
    /** Show content and wait for user to click Continue or Edit */
    waitForUser(title, content, filePath) {
        const contentHtml = content.length > 5000
            ? escHtml(content.substring(0, 5000)) + "\n... (truncated, see full file)"
            : escHtml(content);
        this.addAssistant(`<div class="step-title">${escHtml(title)}</div>` +
            `<pre class="code-block">${contentHtml}</pre>` +
            `<div class="file-path">📄 ${escHtml(filePath)}</div>` +
            `<div class="actions" id="actions-${this.messages.length}">` +
            `<button onclick="send('continue')">▶ Continue</button>` +
            `<button onclick="send('edit')">✏️ Edit in Editor</button>` +
            `</div>`);
        return new Promise((resolve) => {
            this.resolveUserAction = resolve;
        });
    }
    /** Show a yes/no choice and wait for user response */
    waitForChoice(title, message) {
        this.addAssistant(`<div class="step-title">${escHtml(title)}</div>` +
            `<div>${escHtml(message)}</div>` +
            `<div class="actions">` +
            `<button onclick="send('yes')">✅ Yes</button>` +
            `<button onclick="send('no')">⏭ Skip</button>` +
            `</div>`);
        return new Promise((resolve) => {
            this.resolveUserAction = resolve;
        });
    }
    showFinal(title, content, filePath) {
        const contentHtml = content.length > 8000
            ? escHtml(content.substring(0, 8000)) + "\n... (truncated)"
            : escHtml(content);
        this.addAssistant(`<div class="step-title">✅ ${escHtml(title)}</div>` +
            `<pre class="code-block">${contentHtml}</pre>` +
            `<div class="file-path">📄 ${escHtml(filePath)}</div>`);
    }
    showError(msg) {
        this.addAssistant(`<div class="error">❌ ${escHtml(msg)}</div>`);
    }
    render() {
        const msgsHtml = this.messages.map((m) => {
            const cls = m.role === "user" ? "msg-user" :
                m.role === "system" ? "msg-system" : "msg-assistant";
            return `<div class="msg ${cls}">${m.html}</div>`;
        }).join("\n");
        // Step progress bar
        const stepsHtml = STEP_LABELS.map((label, i) => {
            const cls = i < this.currentStep ? "done" :
                i === this.currentStep ? "active" : "pending";
            const icon = i < this.currentStep ? "✅" :
                i === this.currentStep ? "⏳" : "⬜";
            return `<span class="step-badge ${cls}">${icon} ${escHtml(label)}</span>`;
        }).join("");
        this.panel.webview.html = `<!DOCTYPE html>
<html><head><style>
  * { box-sizing: border-box; }
  body { font-family: var(--vscode-font-family); margin: 0; padding: 0;
         color: var(--vscode-foreground); background: var(--vscode-editor-background);
         display: flex; flex-direction: column; height: 100vh; }
  .progress { padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 6px;
              border-bottom: 1px solid var(--vscode-panel-border); font-size: 12px; }
  .step-badge { padding: 2px 6px; border-radius: 3px; opacity: 0.5; }
  .step-badge.done { opacity: 0.7; }
  .step-badge.active { opacity: 1; font-weight: bold;
    background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); }
  .chat { flex: 1; overflow-y: auto; padding: 12px; }
  .msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 95%; }
  .msg-system { background: var(--vscode-textBlockQuote-background); font-size: 12px; opacity: 0.7; }
  .msg-assistant { background: var(--vscode-editor-inactiveSelectionBackground); }
  .msg-user { background: var(--vscode-button-background); color: var(--vscode-button-foreground);
              margin-left: auto; max-width: 60%; text-align: right; }
  .step-title { font-weight: bold; margin-bottom: 8px; font-size: 14px; }
  .code-block { background: var(--vscode-terminal-background); padding: 10px;
                border-radius: 4px; font-family: monospace; font-size: 12px;
                white-space: pre-wrap; word-break: break-all;
                max-height: 400px; overflow: auto; margin: 8px 0; }
  .file-path { font-size: 11px; opacity: 0.6; margin-bottom: 8px; }
  .actions { display: flex; gap: 8px; margin-top: 8px; }
  .actions button { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
    background: var(--vscode-button-background); color: var(--vscode-button-foreground);
    font-size: 13px; }
  .actions button:hover { background: var(--vscode-button-hoverBackground); }
  .error { color: var(--vscode-errorForeground); font-weight: bold; }
</style></head><body>
  <div class="progress">${stepsHtml}</div>
  <div class="chat" id="chat">${msgsHtml}</div>
  <script>
    const vscode = acquireVsCodeApi();
    function send(action) {
      if (action === 'continue' || action === 'yes' || action === 'no' || action === 'edit') {
        document.querySelectorAll('.actions button').forEach(b => b.disabled = true);
      }
      vscode.postMessage({ type: 'userAction', action });
    }
    // Auto-scroll
    const chat = document.getElementById('chat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  </script>
</body></html>`;
    }
    dispose() { this.panel.dispose(); }
}
// ── Interactive pipeline orchestrator ────────────────────
async function runInteractivePipeline(inputFile, outputDir, chat, outputChannel) {
    const stem = path.basename(inputFile, ".ll");
    // Steps 0-6: optimization pipeline
    for (let step = 0; step <= 6; step++) {
        chat.setStep(step);
        chat.addSystem(`Running ${STEP_LABELS[step]} ...`);
        const { result, analysisResult, logs } = await runStep(inputFile, outputDir, step, outputChannel);
        if (!result || result.status === "fail") {
            chat.showError(`${STEP_LABELS[step]} failed.\n` +
                (logs.length > 0 ? logs.slice(-5).join("\n") : "No output"));
            return;
        }
        // Steps 1-5: show content and let user review/edit
        if (PAUSE_STEPS.has(step) && result.content) {
            const action = await chat.waitForUser(result.title || STEP_LABELS[step], result.content, result.file || "");
            if (action === "edit" && result.file && fs.existsSync(result.file)) {
                chat.addUser("Opening file for editing ...");
                const doc = await vscode.workspace.openTextDocument(result.file);
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
                await chat.waitForUser("Edit the file, save, then Continue", "", result.file);
                chat.addUser("Edits done, continuing");
            }
            else {
                chat.addUser("Continue");
            }
        }
        else if (step === 6) {
            // Show final result with action buttons in webview
            if (result.content && result.output_file) {
                const contentHtml = result.content.length > 8000
                    ? escHtml(result.content.substring(0, 8000)) + "\n... (truncated)"
                    : escHtml(result.content);
                chat.addAssistant(`<div class="step-title">✅ Final Optimized IR</div>` +
                    `<pre class="code-block">${contentHtml}</pre>` +
                    `<div class="file-path">📄 ${escHtml(result.output_file)}</div>` +
                    `<div class="actions">` +
                    `<button onclick="send('continue')">▶ Continue</button>` +
                    `<button onclick="send('open')">📄 Open Result</button>` +
                    `<button onclick="send('diff')">🔀 Open Diff</button>` +
                    `</div>`);
                // Loop: open/diff don't advance, only Continue proceeds to verify
                let finalAction = "";
                while (finalAction !== "continue") {
                    finalAction = await new Promise((resolve) => {
                        chat.resolveUserAction = resolve;
                    });
                    if (finalAction === "open" && result.output_file) {
                        const doc = await vscode.workspace.openTextDocument(result.output_file);
                        await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
                    }
                    else if (finalAction === "diff" && result.output_file) {
                        await vscode.commands.executeCommand("vscode.diff", vscode.Uri.file(inputFile), vscode.Uri.file(result.output_file), `${stem}.ll ↔ ${stem}.optimized.ll`);
                    }
                }
            }
            else {
                chat.showError("No optimized IR produced");
                return;
            }
        }
    }
    // Ask user: verify correctness?
    const verifyChoice = await chat.waitForChoice("Verify Correctness?", "Run alive2 formal verification on the optimized IR?");
    if (verifyChoice === "yes") {
        chat.addUser("Yes, verify");
        chat.setStep(7);
        chat.addSystem(`Running ${STEP_LABELS[7]} ...`);
        const { result: vResult, logs: vLogs } = await runStep(inputFile, outputDir, 7, outputChannel);
        if (!vResult || vResult.status === "fail") {
            chat.showError(`Verification failed.\n` +
                (vLogs.length > 0 ? vLogs.slice(-5).join("\n") : "No output"));
            return;
        }
        // Show verification result
        const verifyStatus = vResult.verify_status || "UNKNOWN";
        chat.addAssistant(`<div class="step-title">${escHtml(vResult.title || "Verification")}</div>` +
            `<pre class="code-block">${escHtml(vResult.content || "")}</pre>`);
        if (verifyStatus === "PASS") {
            // Ask user: performance test?
            const perfChoice = await chat.waitForChoice("Performance Test?", "Verification passed! Run performance benchmarks?");
            if (perfChoice === "yes") {
                chat.addUser("Yes, test performance");
                chat.setStep(8);
                chat.addSystem(`Running ${STEP_LABELS[8]} ...`);
                const { result: pResult, logs: pLogs } = await runStep(inputFile, outputDir, 8, outputChannel);
                if (!pResult || pResult.status === "fail") {
                    chat.showError(`Performance testing failed.\n` +
                        (pLogs.length > 0 ? pLogs.slice(-5).join("\n") : "No output"));
                    return;
                }
                chat.addAssistant(`<div class="step-title">📊 ${escHtml(pResult.title || "Performance")}</div>` +
                    `<pre class="code-block">${escHtml(pResult.content || "")}</pre>`);
            }
            else {
                chat.addUser("Skip performance test");
            }
        }
        else {
            chat.addAssistant(`<div class="error">Verification did not pass (${escHtml(verifyStatus)}). ` +
                `Performance testing requires verification to pass.</div>`);
        }
    }
    else {
        chat.addUser("Skip verification");
    }
    chat.addSystem("Pipeline complete.");
}
// ── llvm-extract ─────────────────────────────────────────
async function extractFunctions(sourceFile, funcNames, outputFile, outputChannel) {
    const llvmExtract = cfg("llvmExtract", "/home/amax/yangz/Env/llvm-project/build/bin/llvm-extract");
    const llvmAs = llvmExtract.replace("llvm-extract", "llvm-as");
    const llvmDis = llvmExtract.replace("llvm-extract", "llvm-dis");
    const bcFile = outputFile.replace(/\.ll$/, ".bc");
    const extractedBc = outputFile.replace(/\.ll$/, ".extracted.bc");
    outputChannel.appendLine(`[IntOpt] llvm-as ${sourceFile} -o ${bcFile}`);
    const asOk = await runCmd(llvmAs, [sourceFile, "-o", bcFile]);
    const funcArgs = [];
    for (const fn of funcNames) {
        funcArgs.push("--func", fn);
    }
    if (!asOk) {
        outputChannel.appendLine("[IntOpt] llvm-as failed, trying llvm-extract on .ll directly");
        const ok = await runCmd(llvmExtract, [...funcArgs, sourceFile, "-o", extractedBc]);
        if (!ok) {
            return false;
        }
        return await runCmd(llvmDis, [extractedBc, "-o", outputFile]);
    }
    const ok = await runCmd(llvmExtract, [...funcArgs, bcFile, "-o", extractedBc]);
    if (!ok) {
        return false;
    }
    return await runCmd(llvmDis, [extractedBc, "-o", outputFile]);
}
// ── Commands ─────────────────────────────────────────────
function activate(context) {
    const outputChannel = vscode.window.createOutputChannel("IntOpt");
    context.subscriptions.push(vscode.commands.registerCommand("intopt.optimizeSelection", async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return vscode.window.showErrorMessage("No active editor");
        }
        const selection = editor.selection;
        if (selection.isEmpty) {
            return vscode.window.showErrorMessage("Please select the IR region to optimize");
        }
        const selectedText = editor.document.getText(selection);
        const funcNames = extractFuncNames(selectedText);
        if (funcNames.length === 0) {
            return vscode.window.showErrorMessage("No 'define' blocks found in selection");
        }
        const sourceFile = editor.document.uri.fsPath;
        const stem = path.basename(sourceFile, ".ll");
        let outputDir = cfg("outputDir", "");
        if (!outputDir) {
            outputDir = path.join(path.dirname(sourceFile), `${stem}_intopt`);
        }
        fs.mkdirSync(outputDir, { recursive: true });
        const chat = new ChatPanel();
        chat.addSystem(`Source: ${sourceFile}`);
        chat.addSystem(`Functions: ${funcNames.join(", ")}`);
        // Extract
        chat.addSystem("Extracting selected functions with llvm-extract ...");
        const extractedFile = path.join(outputDir, `${stem}.ll`);
        const ok = await extractFunctions(sourceFile, funcNames, extractedFile, outputChannel);
        if (!ok) {
            chat.addSystem("llvm-extract failed, using raw selection as input");
            fs.writeFileSync(extractedFile, selectedText, "utf-8");
        }
        await runInteractivePipeline(extractedFile, outputDir, chat, outputChannel);
    }));
    context.subscriptions.push(vscode.commands.registerCommand("intopt.optimizeFile", async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return vscode.window.showErrorMessage("No active editor");
        }
        const sourceFile = editor.document.uri.fsPath;
        const stem = path.basename(sourceFile, ".ll");
        let outputDir = cfg("outputDir", "");
        if (!outputDir) {
            outputDir = path.join(path.dirname(sourceFile), `${stem}_intopt`);
        }
        fs.mkdirSync(outputDir, { recursive: true });
        const chat = new ChatPanel();
        chat.addSystem(`Source: ${sourceFile}`);
        await runInteractivePipeline(sourceFile, outputDir, chat, outputChannel);
    }));
    outputChannel.appendLine("[IntOpt] Extension activated");
}
function deactivate() { }
//# sourceMappingURL=extension.js.map