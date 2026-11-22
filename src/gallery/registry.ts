import { frontal } from "../sim/scenes/frontal.ts";
import type { Scene } from "../sim/types.ts";
import { frontalPoster, seaBreezePoster, stormPoster } from "./posters.ts";

export interface GalleryCard {
  id: string;
  title: string;
  kicker: string;
  summary: string;
  status: "ready" | "next";
  poster: string;
  scene?: Scene;
}

export const cards: GalleryCard[] = [
  {
    scene: frontal,
    id: frontal.id,
    title: frontal.title,
    kicker: frontal.kicker,
    summary: frontal.summary,
    status: "ready",
    poster: frontalPoster,
  },
  {
    id: "sea-breeze",
    title: "Sea breeze",
    kicker: "Next",
    summary: "Land heats, the coast answers, and a shallow circulation sets up.",
    status: "next",
    poster: seaBreezePoster,
  },
  {
    id: "thunderstorm",
    title: "Thunderstorm",
    kicker: "Next",
    summary: "A tower, an anvil, and the downdraft that follows the rain.",
    status: "next",
    poster: stormPoster,
  },
];

export function readyScene(id: string): Scene | undefined {
  return cards.find((card) => card.id === id && card.scene)?.scene;
}
