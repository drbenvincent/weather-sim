import { renderGallery } from "./gallery/page.ts";
import { renderMissing } from "./pages/missing.ts";
import { renderSim } from "./pages/sim.ts";

const root = document.querySelector("#app");
if (!root) throw new Error("The page is missing #app.");

let cleanup = () => {};

function show(path: string): void {
  cleanup();
  if (path === "/" || path === "") {
    cleanup = renderGallery(root as HTMLElement);
    return;
  }
  if (path.startsWith("/sim/")) {
    cleanup = renderSim(root as HTMLElement, decodeURIComponent(path.slice(5)));
    return;
  }
  cleanup = renderMissing(root as HTMLElement, "That page is not here.", "The gallery is the place to start.");
}

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const link = target.closest("a");
  if (!link || link.target || link.origin !== location.origin) return;
  event.preventDefault();
  history.pushState(null, "", link.pathname);
  show(link.pathname);
});

window.addEventListener("popstate", () => show(location.pathname));
show(location.pathname);
