# Weather Sim

A browser gallery of atmospheric simulations. The first scene is a cold front meeting a warm front: a two-dimensional Boussinesq solver on the GPU. Cloud and rain come from the microphysics, not from drawn sprites.

## Run

```
npm install
npm run dev
```

Open `/` for the gallery and `/sim/frontal` for the scene. `npm run check` typechecks, runs the CPU tests, and builds.

A current WebGPU browser is required. Chrome and Edge work. If `navigator.gpu` is missing, the page says so instead of showing a blank canvas.

## The model

The slice is 160 km by 10 km, drawn with the height exaggerated and labeled. It solves momentum, a divergence-free projection, and advection of potential temperature, vapor, cloud water, and rain. Condensation is a saturation adjustment and releases latent heat. Rain forms by Kessler autoconversion and falls at a terminal velocity. Small eddies are only an eddy viscosity.

It omits Coriolis force, radiation, ice, and a third dimension. The warm-front slope begins as a wedge of cooler air, steeper than a synoptic front, so the meeting fits the domain. The clouds and the cold surge are computed.

## Add a scene

1. Add a module in `src/sim/scenes/` that exports a `Scene`: domain, parameter definitions, `createInitial`, and `interpret`.
2. Register it in `src/gallery/registry.ts` with poster art.
3. The route is `/sim/<id>`. The solver does not need to change.
