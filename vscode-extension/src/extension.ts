import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import { ChildProcess, spawn } from "child_process";

// ── Helpers ──────────────────────────────────────────────

function cfg<T>(key: string, fallback: T): T {
  return vscode.workspace.getConfiguration("intopt").get<T>(key, fallback);
}

/** Build a shell command string that activates conda first if configured. */
function condaWrap(cmd: string): string {
  const condaEnv = cfg("condaEnv", "");
  if (!condaEnv) { return cmd; }
  // Use conda run to execute in the env without permanently activating
  return `conda run --no-capture-output -n ${condaEnv} ${cmd}`;
}

/** Spawn a command through bash with conda activation. */
function spawnWithConda(
  cmd: string,
  cwd?: string,
  extraEnv?: Record<string, string>
): ChildProcess {
  const wrapped = condaWrap(cmd);
  return spawn("bash", ["--login", "-c", wrapped], {
    cwd,
    env: { ...process.env, ...extraEnv },
  });
}

/** Extract function names from selected IR text (lines starting with "define"). */
function extractFuncNames(text: string): string[] {
  const re = /define\s+.*?@(?:"([^"]+)"|([\w.$-]+))/g;
  const names: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    names.push(m[1] || m[2]);
  }
  return names;
}

// ── Result panel ─────────────────────────────────────────

class ResultPanel {
  private panel: vscode.WebviewPanel;
  private logs: string[] = [];
  private steps: { name: string; status: string; detail: string }[] = [
    { name: "Step 1: Strategy Generation", status: "pending", detail: "" },
    { name: "Step 2: Strategy Mapping", status: "pending", detail: "" },
    { name: "Step 3: Strategy Refinement", status: "pending", detail: "" },
    { name: "Step 4: LLM Realization", status: "pending", detail: "" },
    { name: "Step 5: Post-processing", status: "pending", detail: "" },
  ];

  constructor() {
    this.panel = vscode.window.createWebviewPanel(
      "intoptResult",
      "IntOpt: Optimization Progress",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    this.render();
  }

  appendLog(line: string) {
    this.logs.push(line);
    // Detect step transitions
    for (let i = 0; i < this.steps.length; i++) {
      const stepNum = `Step ${i + 1}`;
      if (line.includes(stepNum)) {
        // Mark previous steps as done
        for (let j = 0; j < i; j++) {
          if (this.steps[j].status !== "done") {
            this.steps[j].status = "done";
          }
        }
        this.steps[i].status = "running";
      }
    }
    this.render();
  }

  setStepDetail(stepIdx: number, detail: string) {
    if (stepIdx >= 0 && stepIdx < this.steps.length) {
      this.steps[stepIdx].detail = detail;
      this.render();
    }
  }

  finish(success: boolean, outputFile?: string) {
    for (const s of this.steps) {
      if (s.status === "running") { s.status = "done"; }
    }
    if (success && outputFile) {
      this.appendLog(`\n✓ Optimization complete: ${outputFile}`);
    } else {
      this.appendLog("\n✗ Optimization failed.");
    }
    this.render();
  }

  private render() {
    const stepsHtml = this.steps
      .map((s) => {
        const icon =
          s.status === "done" ? "✅" :
          s.status === "running" ? "⏳" : "⬜";
        const detail = s.detail
          ? `<pre class="detail">${escHtml(s.detail)}</pre>` : "";
        return `<div class="step ${s.status}">${icon} ${escHtml(s.name)}${detail}</div>`;
      })
      .join("\n");

    const logsHtml = this.logs.map((l) => escHtml(l)).join("\n");

    this.panel.webview.html = `<!DOCTYPE html>
<html><head><style>
  body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); background: var(--vscode-editor-background); }
  .step { padding: 6px 0; font-size: 14px; }
  .step.running { font-weight: bold; }
  .detail { margin: 4px 0 4px 28px; font-size: 12px; opacity: 0.8; max-height: 200px; overflow: auto; }
  .log-box { margin-top: 16px; padding: 8px; background: var(--vscode-terminal-background); border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 400px; overflow: auto; }
  h3 { margin: 0 0 8px 0; }
</style></head><body>
  <h3>IntOpt Optimization Pipeline</h3>
  ${stepsHtml}
  <div class="log-box">${logsHtml}</div>
  <script>
    const box = document.querySelector('.log-box');
    if (box) box.scrollTop = box.scrollHeight;
  </script>
</body></html>`;
  }

  dispose() { this.panel.dispose(); }
}

function escHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Core optimization logic ──────────────────────────────

async function runOptimization(
  inputFile: string,
  outputDir: string,
  panel: ResultPanel,
  outputChannel: vscode.OutputChannel
): Promise<string | undefined> {
  const pythonPath = cfg("pythonPath", "python");
  const projectRoot = cfg("projectRoot", "/home/amax/yangz/intop");
  const configFile = cfg("configFile", path.join(projectRoot, "config", "config.yaml"));

  const args = [
    path.join(projectRoot, "src", "main.py"),
    "--mode", "single",
    "--input", inputFile,
    "--output", outputDir,
    "--config", configFile,
  ];

  const cmdStr = `${pythonPath} ${args.join(" ")}`;
  panel.appendLog(`$ ${cmdStr}`);
  outputChannel.appendLine(`[IntOpt] Running: ${cmdStr}`);

  return new Promise<string | undefined>((resolve) => {
    const proc = spawnWithConda(
      cmdStr,
      path.join(projectRoot, "src"),
      { PYTHONUNBUFFERED: "1" }
    );

    let lastOutput = "";

    const handleData = (data: Buffer) => {
      const text = data.toString();
      lastOutput += text;
      for (const line of text.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed) { continue; }
        panel.appendLog(trimmed);
        outputChannel.appendLine(trimmed);
      }
    };

    proc.stdout?.on("data", handleData);
    proc.stderr?.on("data", handleData);

    proc.on("close", (code: number | null) => {
      if (code === 0) {
        // Find the .optimized.ll file
        const stem = path.basename(inputFile, ".ll");
        const optimized = path.join(outputDir, `${stem}.optimized.ll`);
        if (fs.existsSync(optimized)) {
          panel.finish(true, optimized);
          resolve(optimized);
        } else {
          panel.finish(false);
          resolve(undefined);
        }
      } else {
        panel.finish(false);
        resolve(undefined);
      }
    });

    proc.on("error", (err: Error) => {
      panel.appendLog(`Process error: ${err.message}`);
      panel.finish(false);
      resolve(undefined);
    });
  });
}


// ── llvm-extract ─────────────────────────────────────────

async function extractFunctions(
  sourceFile: string,
  funcNames: string[],
  outputFile: string,
  outputChannel: vscode.OutputChannel
): Promise<boolean> {
  const llvmExtract = cfg("llvmExtract",
    "/home/amax/yangz/Env/llvm-project/build/bin/llvm-extract");

  // First assemble the .ll to .bc, then extract, then disassemble back
  const projectRoot = cfg("projectRoot", "/home/amax/yangz/intop");
  const llvmAs = llvmExtract.replace("llvm-extract", "llvm-as");
  const llvmDis = llvmExtract.replace("llvm-extract", "llvm-dis");

  const bcFile = outputFile.replace(/\.ll$/, ".bc");
  const extractedBc = outputFile.replace(/\.ll$/, ".extracted.bc");

  // Step 1: llvm-as source.ll -o source.bc
  outputChannel.appendLine(`[IntOpt] llvm-as ${sourceFile} -o ${bcFile}`);
  const asOk = await runCmd(llvmAs, [sourceFile, "-o", bcFile]);
  if (!asOk) {
    // If llvm-as fails, the file might already be bitcode or has errors.
    // Try using the .ll directly with llvm-extract (it can handle .ll too)
    outputChannel.appendLine("[IntOpt] llvm-as failed, trying llvm-extract on .ll directly");

    const funcArgs: string[] = [];
    for (const fn of funcNames) {
      funcArgs.push("--func", fn);
    }
    const extractOk = await runCmd(llvmExtract, [
      ...funcArgs, sourceFile, "-o", extractedBc,
    ]);
    if (!extractOk) {
      outputChannel.appendLine("[IntOpt] llvm-extract failed");
      return false;
    }
    const disOk = await runCmd(llvmDis, [extractedBc, "-o", outputFile]);
    return disOk;
  }

  // Step 2: llvm-extract --func=name1 --func=name2 source.bc -o extracted.bc
  const funcArgs: string[] = [];
  for (const fn of funcNames) {
    funcArgs.push("--func", fn);
  }
  outputChannel.appendLine(`[IntOpt] llvm-extract ${funcArgs.join(" ")} ${bcFile}`);
  const extractOk = await runCmd(llvmExtract, [
    ...funcArgs, bcFile, "-o", extractedBc,
  ]);
  if (!extractOk) {
    outputChannel.appendLine("[IntOpt] llvm-extract failed");
    return false;
  }

  // Step 3: llvm-dis extracted.bc -o output.ll
  const disOk = await runCmd(llvmDis, [extractedBc, "-o", outputFile]);
  return disOk;
}

function runCmd(cmd: string, args: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    const fullCmd = `${cmd} ${args.map(a => `'${a}'`).join(" ")}`;
    const proc = spawnWithConda(fullCmd);
    proc.on("close", (code: number | null) => resolve(code === 0));
    proc.on("error", () => resolve(false));
  });
}

// ── Commands ─────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel("IntOpt");

  // Command: optimize selected region
  context.subscriptions.push(
    vscode.commands.registerCommand("intopt.optimizeSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor");
        return;
      }

      const selection = editor.selection;
      if (selection.isEmpty) {
        vscode.window.showErrorMessage(
          "Please select the IR region you want to optimize"
        );
        return;
      }

      const selectedText = editor.document.getText(selection);
      const funcNames = extractFuncNames(selectedText);

      if (funcNames.length === 0) {
        vscode.window.showErrorMessage(
          "No function definitions found in selection. " +
          "Please select a region containing 'define' blocks."
        );
        return;
      }

      const sourceFile = editor.document.uri.fsPath;
      const sourceDir = path.dirname(sourceFile);
      const stem = path.basename(sourceFile, ".ll");

      // Determine output directory
      let outputDir = cfg("outputDir", "");
      if (!outputDir) {
        outputDir = path.join(sourceDir, `${stem}_intopt`);
      }
      fs.mkdirSync(outputDir, { recursive: true });

      const panel = new ResultPanel();
      panel.appendLog(`Source: ${sourceFile}`);
      panel.appendLog(`Functions: ${funcNames.join(", ")}`);
      panel.appendLog(`Output: ${outputDir}`);
      panel.appendLog("");

      // Extract functions using llvm-extract
      panel.appendLog("Extracting selected functions with llvm-extract ...");
      const extractedFile = path.join(outputDir, `${stem}.ll`);

      const ok = await extractFunctions(
        sourceFile, funcNames, extractedFile, outputChannel
      );

      if (!ok) {
        // Fallback: write the selected text directly as a .ll file
        panel.appendLog(
          "llvm-extract failed, using raw selection as input"
        );
        fs.writeFileSync(extractedFile, selectedText, "utf-8");
      }

      panel.appendLog(`Extracted IR: ${extractedFile}`);
      panel.appendLog("");

      // Run optimization pipeline
      const result = await runOptimization(
        extractedFile, outputDir, panel, outputChannel
      );

      if (result) {
        const action = await vscode.window.showInformationMessage(
          `Optimization complete: ${path.basename(result)}`,
          "Open Result", "Open Diff"
        );
        if (action === "Open Result") {
          const doc = await vscode.workspace.openTextDocument(result);
          await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
        } else if (action === "Open Diff") {
          const leftUri = vscode.Uri.file(extractedFile);
          const rightUri = vscode.Uri.file(result);
          await vscode.commands.executeCommand(
            "vscode.diff", leftUri, rightUri,
            `${stem}.ll ↔ ${stem}.optimized.ll`
          );
        }
      }
    })
  );

  // Command: optimize entire file
  context.subscriptions.push(
    vscode.commands.registerCommand("intopt.optimizeFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor");
        return;
      }

      const sourceFile = editor.document.uri.fsPath;
      const sourceDir = path.dirname(sourceFile);
      const stem = path.basename(sourceFile, ".ll");

      let outputDir = cfg("outputDir", "");
      if (!outputDir) {
        outputDir = path.join(sourceDir, `${stem}_intopt`);
      }
      fs.mkdirSync(outputDir, { recursive: true });

      const panel = new ResultPanel();
      panel.appendLog(`Source: ${sourceFile}`);
      panel.appendLog(`Output: ${outputDir}`);
      panel.appendLog("");

      const result = await runOptimization(
        sourceFile, outputDir, panel, outputChannel
      );

      if (result) {
        const action = await vscode.window.showInformationMessage(
          `Optimization complete: ${path.basename(result)}`,
          "Open Result", "Open Diff"
        );
        if (action === "Open Result") {
          const doc = await vscode.workspace.openTextDocument(result);
          await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
        } else if (action === "Open Diff") {
          const leftUri = vscode.Uri.file(sourceFile);
          const rightUri = vscode.Uri.file(result);
          await vscode.commands.executeCommand(
            "vscode.diff", leftUri, rightUri,
            `${stem}.ll ↔ ${stem}.optimized.ll`
          );
        }
      }
    })
  );

  outputChannel.appendLine("[IntOpt] Extension activated");
}

export function deactivate() {}
