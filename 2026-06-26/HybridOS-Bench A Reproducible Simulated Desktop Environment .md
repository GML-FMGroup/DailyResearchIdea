# HybridOS-Bench: A Reproducible Simulated Desktop Environment Integrating a Full Browser for Agent Evaluation

## Motivation

Existing agent benchmarks either focus solely on web browsing (e.g., WebVoyager, REAL) or on OS-level desktop tasks (e.g., TheAgentCompany), but none provide a unified, reproducible environment where agents can seamlessly interact with both a real browser and OS applications (file system, notifications). This gap prevents evaluation of cross-application tasks like 'download a file from a web page and open it in a text editor', which are common in real-world workflows. The root cause is that prior environments sacrifice either browser fidelity (by using simplified web containers) or OS reproducibility (by relying on live systems).

## Key Insight

Enclosing a full browser inside a lightweight, snapshotted desktop environment makes the entire system state deterministic, so that any performance difference across runs can be causally attributed to the agent's actions rather than to environment variability.

## Method

**A) What it is** – HybridOS-Bench is a benchmark environment that runs a real Chromium browser inside a simulated Linux desktop (Xvfb + lightweight window manager) with a snapshotted file system, notification service, and helper applications (text editor, file manager, email client). The agent receives screenshots and a textual description of the current OS state (e.g., open windows, file tree) and outputs keyboard/mouse actions. Each task starts from the same snapshot, ensuring reproducibility. Web content is controlled via a local proxy that serves static archives of websites (using `wget --mirror` to pre-download pages and assets, then served via a lightweight HTTP server). Network access is disabled except to the proxy.

**B) How it works**
```pseudocode
class HybridOSEnv:
  def __init__(snapshot_path, web_archive_path):
    self.desktop = DesktopVM(snapshot_path)         # QEMU + overlay disk
    self.browser = BrowserController(self.desktop)   # drives Chromium via Xdotool
    self.proxy = LocalProxy(web_archive_path)        # HTTP server serving static pages
    self.browser.set_proxy('127.0.0.1', self.proxy.port)
    self.snapshot = snapshot_path

  def reset():
    restore_vm_snapshot(self.snapshot)               # reverts disk & memory
    start_vm()
    self.proxy.reset_logs()                          # clear request logs for deterministic state
    wait_until_ready()                               # desktop idle, browser at homepage
    return get_observation()                         # screenshot + state text

  def step(action):
    # action: mouse_click(x,y), type(text), key_combo(...)
    execute_action(self.desktop, action)
    wait_for_stable_state(timeout=2.0)                # no screen change for 0.5s
    obs = get_observation()
    done = check_task_completion()                   # programmatic or LLM-judged
    reward = 1.0 if done else 0.0
    return obs, reward, done

  def get_observation():
    screenshot = capture_screen(self.desktop)
    window_tree = parse_open_windows()               # list of window titles & positions
    file_tree = list_files('/home/user/Desktop') if task involves files
    notifications = get_recent_notifications()
    return {'screenshot': screenshot, 'state_text': f"Windows: {window_tree}\nFiles: {file_tree}\nNotifications: {notifications}"}
```
Hyperparameters: snapshot frequency (every 10 steps optionally for mid-task checkpoint), stable-state timeout 2.0s, task timeout 60 steps, web archive path: static HTML pages generated from original websites.

**C) Why this design** – We chose to embed a real browser (Chromium) rather than a simulated web view because real browser behavior (JavaScript rendering, network requests, cookies) must be captured for tasks like form submission or shopping; the cost is higher resource usage and slower step times (~2s per step). We selected an overlay disk snapshot over full VM clone because it restores in <1s versus >10s for clone, while still guaranteeing bit-identical filesystem state; the trade-off is that memory state (e.g., browser cache) is also discarded, which can affect tasks relying on session persistence, so we optionally provide mid-task snapshots. We adopted a combined visual+textual observation (screenshot plus parsed window tree) to align with typical LLM agent inputs; this is more expensive than pure text but necessary for web content rendered as images. Unlike TheAgentCompany which uses Dockerized services, our VM-level snapshot gives stronger determinism (even clock and device state are reset), accepting higher startup overhead. To ensure web content determinism, we serve static archives of websites via a local proxy, disabling network access; this guarantees that page content is identical across trials, even if the original site changes.

**D) Why it measures what we claim** – The restoration of the VM snapshot (`restore_vm_snapshot`) measures **reproducibility** because it ensures that every agent trial starts from the identical disk and memory state, making the environment a deterministic function of actions; this assumption fails when the web pages themselves change (e.g., live news sites), in which case trials become incomparable. To address this, we serve static web archives via a local proxy, so page content is frozen. We quantify reproducibility as the variance of task success across 10 identical trials: if variance > 0.05, the environment is considered non-deterministic. The `stable-state waiting` measures **fairness** because it prevents the agent from acting on intermediate animation frames; the assumption is that all state transitions are deterministic given the action, which fails if the OS processes have scheduling nondeterminism, causing the agent to see slightly different frames across runs. The `task_completion` check (programmatic for file operations, LLM-judged for retrieval) measures **ecologically valid task success** because it directly tests whether the agent achieved the stated goal; the assumption is that the goal is objectively verifiable, which fails for ambiguous tasks where human judgment disagrees. The combined observation (screenshot + state text) measures **information completeness** because the agent receives both visual and structured data; the assumption is that the state text captures all task-relevant OS state, which fails if a task requires reading a notification that disappeared before the observation extractor ran.

## Contribution

(1) HybridOS-Bench, a reproducible benchmark environment that combines a real Chromium browser inside a simulated Linux desktop with file system, notifications, and helper applications, enabling cross-application task evaluation. (2) A snapshot-based determinism mechanism that ensures identical initial conditions across runs, making agent comparisons statistically reliable. (3) A curated set of 40 cross-application tasks (e.g., 'download a PDF from a website and open it in a text editor') that require both web navigation and OS-level interactions, publicly released along with the environment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 100 tasks across file, email, web (derived from a survey of common cross-application workflows, e.g., downloading and editing files) | Realistic agent use cases |
| Primary metric | Task success rate (binary) | Directly measures goal achievement |
| Secondary metric | Reproducibility (variance of success rate across 10 identical trials) | Quantifies determinism of environment |
| Baseline 1 | TheAgentCompany | Dockerized services; less deterministic |
| Baseline 2 | REAL | Headless simulated websites; no real browser |
| Baseline 3 | WebVoyager | Live web with human evaluation; low reproducibility |
| Ablation | Ours without screenshots (text-only) | Tests importance of visual observation |

### Why this setup validates the claim

The combination of diverse baselines targets each claimed advantage: TheAgentCompany tests reproducibility (our VM snapshot vs. Docker restart); REAL tests ecological validity (real browser rendering vs. headless simulation); WebVoyager tests fairness (stable-state waiting vs. live page variability). The ablation isolates the value of combined visual+textual observation. The dataset spans core OS and web tasks, ensuring generalizability. Task success rate is a direct, falsifiable measure: if our environment truly improves reproducibility and ecological validity, agents should achieve higher and more consistent success rates on our benchmark compared to baselines, especially on tasks where the baselines’ mechanism gaps (e.g., non-deterministic web state, missing visual cues) cause failures. We will also report reproducibility (variance across 10 runs) for each baseline and our method to directly compare determinism. If our method fails to show a clear advantage in those targeted subsets, the central claim is undermined.

### Expected outcome and causal chain

**vs. TheAgentCompany** — On a case where a task requires persistent browser cookies across steps (e.g., e‑commerce checkout), TheAgentCompany’s Dockerized services may reset browser state inconsistently, causing session loss and task failure. Our method’s VM snapshot restores exact disk and memory, preserving cookies, so we expect a noticeable success rate gap on such session-dependent tasks (e.g., 80% vs. 50%). Additionally, we expect our reproducibility metric to be lower (better) than TheAgentCompany's (e.g., variance 0.02 vs. 0.15).

**vs. REAL** — On a task that involves JavaScript‑rendered content (e.g., a dynamic chart), REAL’s headless simulation may omit interactive elements, making the task impossible. Our real Chromium browser renders the full page, enabling correct interaction. We expect a clear gap on tasks requiring client‑side rendering (e.g., 70% vs. 30%).

**vs. WebVoyager** — On a task that depends on a specific webpage state (e.g., news article), WebVoyager’s live web may have changed content between trials, causing inconsistent results. Our snapshotted environment freezes the page content via static web archives, so we expect higher reproducibility (low variance across runs) and a moderate success rate advantage on time‑sensitive tasks (e.g., 75% vs. 60%).

### What would falsify this idea

If our method shows no advantage over TheAgentCompany or REAL on tasks explicitly requiring real browser features (e.g., identical success rates on JS‑heavy tasks), or if the text‑only ablation matches the full method on visually‑dependent tasks, then the claimed benefits of real browser and combined observation are not supported. Additionally, if the reproducibility variance exceeds 0.05 on any task (indicating non-determinism despite web archives), the environment fails to provide the claimed reproducibility.

## References

1. Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments
2. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models
3. REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites
4. GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation
5. The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use
6. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
7. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
8. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents
