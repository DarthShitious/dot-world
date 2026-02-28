# Dot World

Dot World is a real-time evolutionary simulation of simple agents (“dots”) interacting in a 2D environment.  
Agents move, perceive, fight, eat, and reproduce under explicit energy constraints, with evolution driven by both genetic algorithms and within-lifetime learning (the Baldwin effect).

The project is designed as an experimental sandbox for studying **emergent behavior**, **energy-constrained evolution**, and **the interaction between learning and evolution**.

---

## Core Concepts

### Agents (“Dots”)
Each dot is a circular agent with:
- **Mass** (affects max energy and movement cost)
- **Strength** (affects interactions)
- **Hue** (purely genetic, used for visualization)
- **Energy** (single conserved resource for survival and action)

Energy is required for:
- Movement (quadratic cost in distance)
- Attacking
- Reproduction
- Passive decay over time

When energy reaches zero, the dot dies.

---

### Perception
Dots observe their local environment through an **egocentric grid-based retina**:
- Color information from nearby dots and food
- Boundary / world information
- Fixed observation radius
- Toroidal (cyclic) world topology

The rectangular observation grid is an implementation detail for performance; perceptually, the field of view is local and radial.

---

### Brains
Each dot has a small neural network:
- Fixed architecture shared by all dots
- Weights are genetically inherited and mutated
- The final layer is adjusted via reinforcement learning during the dot’s lifetime (Baldwin effect)

The network outputs probabilities over high-level actions:
- Move
- Attack
- Sexual reproduction

---

### Interactions
When two dots are close enough, an interaction game occurs:

| Dot A | Dot B | Outcome |
|-----|-----|-----|
| Mate | Mate | Sexual reproduction |
| Mate | Attack | Attacker survives, mate dies |
| Attack | Attack | Both die |

Only the closest interaction per tick is resolved to avoid combinatorial explosions.

---

### Reproduction

#### Sexual Reproduction
- Requires both dots to exceed a minimum energy fraction
- Parents drop to a fixed post-mating energy fraction
- Produces **4 offspring**
- **Energy-conserving**: offspring energy comes from parents’ pre-mating energy

#### Asexual Reproduction (Fission)
- Triggered when eating food would exceed max energy
- Parent is destroyed
- Produces **2 offspring**
- **Energy-conserving**: parent’s energy is split among children

---

### Food and Energy
- Food spawns stochastically in the environment
- Food has fixed energy value
- When dots die from being killed, a fraction of their remaining energy is scattered as food
- Energy is explicitly tracked and conserved (except for decay and food spawning)

The HUD displays **total energy in the simulation** to help detect energy leaks.

---

## Evolution and Recall Buffer

To avoid restarting from scratch after extinctions:
- A FIFO **genetic recall buffer** stores genomes from reproducing dots
- On extinction:
  - Half the population is randomly initialized
  - Half is spawned from mutated recall-buffer genomes

This allows long-term evolutionary continuity across collapses.

---

## Metrics and HUD

The on-screen HUD displays:
- Alive population
- Food count
- Total energy in simulation
- Reset count
- Average reward
- Median reward
- **Asexual reproduction rate**
- **Sexual reproduction rate**
- **Kill rate**

Rates are averaged over a configurable rolling window (default: 16 ticks).

---

## Controls

- **B** – Toggle rendering (simulation runs as fast as possible when rendering is off)
- **R** – Hard reset (clears recall buffer)
- **ESC / Close window** – Exit

---

## Performance Philosophy

The simulation is structured to:
- Run at a stable frame rate when rendering is enabled
- Advance as fast as possible when rendering is disabled
- Favor simple, explicit physics and energy accounting
- Be amenable to future optimization (Rust, GPU inference, headless batch runs)

Current implementation is in Python using PyTorch and Pygame for rapid iteration.

---

## Installation

```bash
git clone <repo-url>
cd dot-evolver
pip install -r requirements.txt
python main.py
