export function renderMissing(root: HTMLElement, title: string, message: string): () => void {
  root.replaceChildren();
  const page = document.createElement("main");
  page.className = "missing";
  const home = document.createElement("a");
  home.href = "/";
  home.textContent = "Weather Sim";
  home.className = "wordmark";
  const heading = document.createElement("h1");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = message;
  page.append(home, heading, copy);
  root.append(page);
  return () => root.replaceChildren();
}
