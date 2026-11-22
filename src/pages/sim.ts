import { readyScene } from "../gallery/registry.ts";
import { requestDevice } from "../sim/engine/gpu.ts";
import { Solver } from "../sim/engine/solver.ts";
import type { Diagnostics, ParamValues, Scene } from "../sim/types.ts";
import { emptyDiagnostics, paramValues } from "../sim/types.ts";
import { exaggeration, Renderer, type FieldMode } from "../sim/viz/renderer.ts";
import { renderMissing } from "./missing.ts";

const TICK_SECONDS = 8 / 15;
const MODES: Array<{ id: FieldMode; label: string }> = [
  { id: 0, label: "Composed" },
  { id: 1, label: "Temperature" },
  { id: 2, label: "Cloud" },
  { id: 3, label: "Rain" },
  { id: 4, label: "Divergence" },
];

function celsius(theta: number): string {
  return `${(theta - 273.15).toFixed(0)} °C`;
}

function meanRange(values: ArrayLike<number>, x0: number, x1: number, extent: number): number {
  const width = extent / values.length;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < values.length; i += 1) {
    const left = i * width;
    if (left < x1 && left + width > x0) {
      sum += values[i];
      n += 1;
    }
  }
  return n ? sum / n : 0;
}

function rainText(diag: Diagnostics): string {
  if (diag.rainRate > 1e-6) return `${(diag.rainRate * 3.6e6).toFixed(1)} mm/h`;
  return "none";
}

function sci(value: number): string {
  if (!Number.isFinite(value) || value === 0) return "0";
  return value.toExponential(1);
}

export function renderSim(root: HTMLElement, id: string): () => void {
  const scene = readyScene(id);
  if (!scene) {
    return renderMissing(root, "That simulation is not here.", "The gallery only opens scenes that are ready.");
  }
  return mount(root, scene);
}

function mount(root: HTMLElement, scene: Scene): () => void {
  let stopped = false;
  let frame = 0;
  let solver: Solver | undefined;
  let renderer: Renderer | undefined;
  let device: GPUDevice | undefined;
  const values: ParamValues = paramValues(scene.params);
  let speed = values.speed ?? 2;
  let paused = false;
  let mode: FieldMode = 0;
  let accumulator = 0;
  let last = performance.now();

  root.replaceChildren();
  const page = document.createElement("div");
  page.className = "sim";
  page.innerHTML = `
    <header class="topbar">
      <a class="wordmark" href="/">Weather Sim</a>
      <h1></h1>
      <p class="clock" aria-label="Simulation time">0.00 h</p>
      <button type="button" class="pause" aria-pressed="false">Pause</button>
      <button type="button" class="reset">Reset</button>
    </header>
    <div class="stage">
      <div class="frame">
        <canvas></canvas>
        <p class="banner" hidden></p>
        <p class="front tag" hidden>Cold front</p>
        <p class="cloud tag" hidden>Cloud top</p>
        <div class="scale"><i></i><span>20 km</span></div>
        <p class="exag"></p>
        <p class="legend">Composed</p>
        <p class="caption" hidden></p>
        <div class="toggle" role="group" aria-label="Field"></div>
      </div>
      <aside class="panel">
        <p class="kicker"></p>
        <p class="summary"></p>
        <ol class="stages"></ol>
        <dl class="readouts"></dl>
        <form class="controls"></form>
        <p class="note"></p>
      </aside>
    </div>
  `;
  root.append(page);
  const title = page.querySelector("h1") as HTMLElement;
  title.textContent = scene.title;
  (page.querySelector(".kicker") as HTMLElement).textContent = scene.kicker;
  (page.querySelector(".summary") as HTMLElement).textContent = scene.modelNoteShort;
  (page.querySelector(".note") as HTMLElement).textContent = scene.modelNote;
  const clock = page.querySelector(".clock") as HTMLElement;
  const canvas = page.querySelector("canvas") as HTMLCanvasElement;
  const banner = page.querySelector(".banner") as HTMLElement;
  const frontTag = page.querySelector(".front") as HTMLElement;
  const cloudTag = page.querySelector(".cloud") as HTMLElement;
  const exag = page.querySelector(".exag") as HTMLElement;
  const legend = page.querySelector(".legend") as HTMLElement;
  const caption = page.querySelector(".caption") as HTMLElement;
  const pauseButton = page.querySelector(".pause") as HTMLButtonElement;
  const resetButton = page.querySelector(".reset") as HTMLButtonElement;
  const stages = page.querySelector(".stages") as HTMLOListElement;
  const readouts = page.querySelector(".readouts") as HTMLDListElement;
  const controls = page.querySelector(".controls") as HTMLFormElement;
  const toggle = page.querySelector(".toggle") as HTMLElement;
  let lessonStage = 0;

  const stageNodes = scene.interpret(emptyDiagnostics(), values).stages.map((stage, index) => {
    const item = document.createElement("li");
    const indexLabel = document.createElement("span");
    indexLabel.textContent = String(index + 1).padStart(2, "0");
    const body = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = stage.title;
    const text = document.createElement("p");
    body.append(heading, text);
    item.append(indexLabel, body);
    stages.append(item);
    return { item, text };
  });

  const readout = (label: string): HTMLElement => {
    const term = document.createElement("dt");
    term.textContent = label;
    const value = document.createElement("dd");
    value.textContent = "—";
    readouts.append(term, value);
    return value;
  };
  const frontOut = readout("Front");
  const updraftOut = readout("Updraft");
  const cloudOut = readout("Cloud top");
  const qcOut = readout("Cloud water");
  const qrOut = readout("Rain water");
  const rainOut = readout("Surface rain");
  const coldOut = readout("Cold side");
  const warmOut = readout("Warm side");
  const divOut = readout("Divergence");
  const firstOut = readout("First step");

  for (const entry of MODES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.label;
    button.setAttribute("aria-pressed", entry.id === 0 ? "true" : "false");
    button.addEventListener("click", () => {
      mode = entry.id;
      legend.textContent = entry.label;
      for (const child of toggle.children) {
        child.setAttribute("aria-pressed", child === button ? "true" : "false");
      }
    });
    toggle.append(button);
  }

  for (const param of scene.params) {
    const wrap = document.createElement("label");
    wrap.className = "control";
    const name = document.createElement("span");
    const live = document.createElement("b");
    name.append(param.label, live);
    const note = document.createElement("small");
    note.textContent = param.note;
    if (param.options) {
      const row = document.createElement("span");
      row.className = "options";
      for (const option of param.options) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `${option}×`;
        button.setAttribute("aria-pressed", option === speed ? "true" : "false");
        button.addEventListener("click", () => {
          speed = option;
          values.speed = option;
          live.textContent = `${option}×`;
          for (const child of row.children) {
            child.setAttribute("aria-pressed", child === button ? "true" : "false");
          }
        });
        row.append(button);
      }
      live.textContent = `${speed}×`;
      wrap.append(name, row, note);
    } else {
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(param.min);
      input.max = String(param.max);
      input.step = String(param.step);
      input.value = String(param.value);
      const format = () => {
        live.textContent = param.id === "humidity"
          ? `${Math.round(Number(input.value) * 100)}%`
          : `${Number(input.value).toFixed(2)}×`;
      };
      format();
      input.addEventListener("input", format);
      input.addEventListener("change", () => {
        values[param.id] = Number(input.value);
        if (param.restarts) restart();
      });
      wrap.append(name, input, note);
    }
    controls.append(wrap);
  }



  function showLesson(diag: Diagnostics): void {
    const lesson = scene.interpret(diag, values);
    if (diag.ready) lessonStage = Math.max(lessonStage, lesson.stage);
    const shown = diag.ready ? lessonStage : 0;
    stageNodes.forEach((node, index) => {
      node.item.classList.toggle("current", index === shown);
      node.item.classList.toggle("reached", index <= shown);
      node.text.textContent = index === shown ? lesson.stages[index].text : "";
    });
  }

  function paintHud(): void {
    const diag = solver?.diagnostics() ?? emptyDiagnostics();
    showLesson(diag);
    caption.hidden = !diag.ready || mode !== 4;
    if (!diag.ready) {
      frontOut.textContent = "—";
      updraftOut.textContent = "—";
      cloudOut.textContent = "—";
      qcOut.textContent = "—";
      qrOut.textContent = "—";
      rainOut.textContent = "—";
      coldOut.textContent = "—";
      warmOut.textContent = "—";
      divOut.textContent = "—";
      firstOut.textContent = "—";
      frontTag.hidden = true;
      cloudTag.hidden = true;
      banner.hidden = true;
      return;
    }
    const extent = scene.domain.extentX;
    frontOut.textContent = `${(diag.frontX / 1000).toFixed(1)} km`;
    updraftOut.textContent = `${diag.maxUpdraft.toFixed(1)} m/s`;
    cloudOut.textContent = diag.maxQc > 1e-5 ? `${(diag.cloudTop / 1000).toFixed(1)} km` : "none";
    qcOut.textContent = `${(diag.maxQc * 1000).toFixed(2)} g/kg`;
    qrOut.textContent = `${(diag.maxQr * 1000).toFixed(2)} g/kg`;
    rainOut.textContent = rainText(diag);
    coldOut.textContent = celsius(meanRange(diag.sfcTheta, 0, 20_000, extent));
    warmOut.textContent = celsius(meanRange(diag.sfcTheta, 55_000, 80_000, extent));
    divOut.textContent = `${sci(diag.divAfter)} /s`;
    firstOut.textContent = `${sci(diag.divFirst)} /s`;
    frontTag.hidden = diag.frontGrad < 1e-4;
    frontTag.style.left = `${(diag.frontX / extent) * 100}%`;
    cloudTag.hidden = diag.maxQc < 1e-5;
    cloudTag.style.left = `${(diag.cloudX / extent) * 100}%`;
    cloudTag.style.bottom = `${Math.min(92, (diag.cloudTop / scene.domain.extentZ) * 100)}%`;
    caption.textContent = `Residual ${sci(diag.divAfter)} /s · first step ${sci(diag.divFirst)} /s. A flat field at this scale means the projection is holding.`;
    const unstable = !Number.isFinite(diag.maxSpeed) || diag.maxSpeed > 100;
    banner.hidden = !unstable;
    banner.textContent = unstable ? "The maximum speed has left the stable range." : "";
  }

  function restart(): void {
    if (!solver) return;
    lessonStage = 0;
    solver.reset(scene.createInitial(values));
    accumulator = 0;
    clock.textContent = "0.00 h";
    paintHud();
  }
  showLesson(emptyDiagnostics());


  function onKey(event: KeyboardEvent): void {
    const tag = (event.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "BUTTON" || tag === "A") return;
    if (event.code === "Space") {
      event.preventDefault();
      paused = !paused;
      pauseButton.textContent = paused ? "Play" : "Pause";
      pauseButton.setAttribute("aria-pressed", paused ? "true" : "false");
    }
    if (event.code === "KeyR") restart();
  }

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    pauseButton.textContent = paused ? "Play" : "Pause";
    pauseButton.setAttribute("aria-pressed", paused ? "true" : "false");
  });
  resetButton.addEventListener("click", restart);
  window.addEventListener("keydown", onKey);

  const loop = (now: number) => {
    if (stopped || !solver || !renderer) return;
    const wall = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!paused) {
      accumulator += wall;
      if (accumulator >= TICK_SECONDS) {
        solver.step(speed);
        accumulator -= TICK_SECONDS;
        if (accumulator > TICK_SECONDS) accumulator = TICK_SECONDS;
      }
      clock.textContent = `${(solver.simTime / 3600).toFixed(2)} h`;
    }
    const simRate = paused ? 0 : (speed * solver.stepSeconds()) / TICK_SECONDS;
    solver.writeTracerDelta(simRate * wall);
    renderer.resize(canvas);
    const factor = exaggeration(canvas, scene.domain.extentX, scene.domain.extentZ);
    const shown = Math.round(factor * 10) / 10;
    exag.textContent = `Height exaggerated ${Number.isInteger(shown) ? shown.toFixed(0) : shown.toFixed(1)}×`;
    const diag = solver.diagnostics();
    renderer.draw(
      canvas,
      solver.view(),
      solver.tracerState(),
      mode,
      solver.simTime,
      diag.divFirst,
      mode === 0 || mode === 1,
    );
    paintHud();
    frame = requestAnimationFrame(loop);
  };

  void (async () => {
    try {
      device = await requestDevice();
      if (stopped) {
        device.destroy();
        return;
      }
      device.lost.then((info) => {
        if (stopped) return;
        fail(`The GPU device was lost (${info.reason}). Reload the page.`);
      });
      renderer = await Renderer.create(device, canvas);
      solver = await Solver.create(device, scene.domain, scene.createInitial(values));
      if (stopped) {
        solver.destroy();
        renderer.destroy();
        device.destroy();
        return;
      }
      frame = requestAnimationFrame(loop);
    } catch (error) {
      if (!stopped) fail(error instanceof Error ? error.message : "The simulation could not start.");
    }
  })();

  function fail(message: string): void {
    stopped = true;
    cancelAnimationFrame(frame);
    renderMissing(root, "This browser cannot run the simulation.", message);
  }

  return () => {
    stopped = true;
    cancelAnimationFrame(frame);
    window.removeEventListener("keydown", onKey);
    solver?.destroy();
    renderer?.destroy();
    device?.destroy();
    root.replaceChildren();
  };
}
