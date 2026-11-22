import { cards } from "./registry.ts";

export function renderGallery(root: HTMLElement): () => void {
  root.replaceChildren();
  const page = document.createElement("main");
  page.className = "gallery";
  page.innerHTML = `
    <header class="mast">
      <p class="kicker">Weather Sim</p>
      <h1>Watch the air decide.</h1>
      <p class="deck">Browser simulations of the atmosphere. The clouds are not sprites. They condense because the air got cold.</p>
    </header>
    <section class="cards"></section>
  `;
  const grid = page.querySelector(".cards") as HTMLElement;
  for (const card of cards) {
    const node = document.createElement(card.status === "ready" ? "a" : "article");
    node.className = `card ${card.status}`;
    if (node instanceof HTMLAnchorElement) node.href = `/sim/${card.id}`;
    node.innerHTML = `
      <div class="poster">${card.poster}</div>
      <div class="card-copy">
        <p class="kicker">${card.kicker}</p>
        <h2>${card.title}</h2>
        <p>${card.summary}</p>
      </div>
    `;
    grid.append(node);
  }
  root.append(page);
  return () => root.replaceChildren();
}
