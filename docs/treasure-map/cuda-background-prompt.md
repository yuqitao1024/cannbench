# CUDA Treasure Map Background Prompt

## Purpose

Generate a single fixed dark-theme background image for the CUDA treasure map modal.

- Visual direction: dark old-paper treasure map
- Scope: fixed background for the current CUDA route only
- Frontend overlays remain code-driven:
  - node labels
  - hover/focus field notes
- The main golden route may be baked into the generated image

## Recommended Prompt

```text
Create a dark vintage treasure-map background for a GPU operator optimization route UI.

Style:
old paper texture, aged parchment, subtle burnished edges, warm sepia palette, yellowed antique paper, restrained charcoal shadows, faint golden glow, hand-drawn cartography feeling, cinematic but restrained, not fantasy cartoon, not colorful, not glossy, not modern infographic.
Keep the parchment slightly brighter and warmer than a near-black map: more amber, more yellowed paper, less heavy charcoal darkness.

Composition:
16:10 landscape canvas, centered composition, large map board filling most of the frame, generous empty space for frontend overlays, especially around the middle and lower-middle areas. The image should feel like an old navigational chart laid on a dark desk, but the desk itself should be minimal or invisible.

Main route:
include exactly one continuous bold golden treasure path that travels diagonally across the map in a smooth, elegant journey.
The route should:
- start from upper-left area
- travel through upper-middle
- descend through center-right
- continue toward lower-middle
- remain fully inside the visible map area
- never leave the frame
- never be cropped
- never break in the middle
- never split into separate golden segments
- never create two independent golden routes
- never create isolated golden islands
- feel smooth and intentional, like a plotted treasure route
- be visually prominent but not so thick that it dominates the whole image
- read clearly as one uninterrupted treasure journey from start to finish

Branch hints:
add a few faint secondary dotted or dashed side-routes in muted desaturated teal-gray, subtle and atmospheric, clearly secondary to the main golden route.
Only the single main route may use bright gold.

Map details:
faint contour lines, subtle grid hints, worn navigation marks, tiny cartographic symbols, very light coordinate lines, soft paper grain, delicate ink fading, slight vignette, understated glow near the golden route.
Preserve the dark scholarly mood, but let the paper surface read as aged golden parchment rather than deep black-brown leather.

Do NOT include:
no text
no labels
no icons for nodes
no legends
no compasses as a central focal point
no characters
no ships
no monsters
no decorative clutter that would block overlay UI
no heavy center illustration
no bright colors
no purple
no large stamps or seals
no watermark

Important UI constraints:
leave clean negative space where frontend labels can be placed:
- upper-left
- upper-center
- upper-right
- center
- center-right
- lower-left
- lower-center
- lower-right

The background must support overlaying interactive rectangular node labels and floating tooltips. Keep the path readable, but avoid busy texture directly beneath likely label areas.

Keep the main golden route roughly passing through these relative anchor regions in order:
upper-left, left-center, upper-middle, center-right, lower-center, lower-right, lower-middle, left-middle.

Overall mood:
dark scholarly treasure chart, retro-technical, elite hacker cartography, old paper meets performance engineering, elegant, sparse, high-end.
```

## Use Notes

- Dark-theme only for now.
- This prompt is intended for a fixed background asset, not a reusable template.
- If the generated path conflicts with frontend node positions, regenerate the image rather than distorting the frontend labels.
- Keep the image free of text so the frontend remains the single source of truth for route naming.
