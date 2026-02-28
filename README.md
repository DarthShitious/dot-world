Dot Evolver Rewrite (High-Performance Python + optional CUDA/Torch)

Goals
- 60 FPS cap when rendering is ON.
- When rendering is OFF (press B), run the simulation as fast as possible.

Key ideas
- Structure-of-arrays state (Torch tensors) instead of Python objects.
- Per-dot small MLP brains, batched inference on GPU when available.
- Baldwin learning updates ONLY the final layer (policy head), manual REINFORCE-style update (no autograd).
- Observations sampled from a low-res world raster (hue/sat + food mask) so observation building is batchable.

Controls
- B : toggle rendering ON/OFF (fast-forward)
- L : toggle Baldwin learning ON/OFF
- R : hard reset (new random population + clears recall buffer)
- ESC : quit

Run
pip install -r requirements.txt
python main.py
