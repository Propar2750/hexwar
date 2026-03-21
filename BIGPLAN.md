## Project: HexWar — Multi-Agent Territory Control on a Hex Grid

### The Game Design

**Core concept:** 4 players start in corners of a hex grid (~80-120 hexes). Each player controls territories, gathers resources, builds units, and tries to dominate the map. All moves are simultaneous (resolved with a conflict system). The game ends when one player controls 60%+ of the map or after N turns (highest territory count wins).

**Key mechanics you'll implement:**

- **Hex grid world** with 3 terrain types: plains (easy to move/hold), mountains (defensive bonus), and fertile land (extra resource generation)
- **Resource system:** territories generate resources each turn, you spend them to build scouts (cheap, fast) and soldiers (expensive, strong)
- **Fog of war:** you only see tiles within range of your units — everything else is hidden
- **Simultaneous movement:** all players submit orders at once, conflicts resolved by unit strength + terrain bonus + a small random factor
- **Simple diplomacy:** before each turn, players can send binary signals — "peace proposal" or "threat" — to other players (cheap talk, not binding)

---

### Week-by-Week Roadmap (~12-14 weeks)

**Phase 1: Environment Engineering (Weeks 1-4)**

_Week 1 — Hex Grid Engine_ Build the hex grid library from scratch. Axial coordinate system, neighbor lookups, distance calculations, pathfinding (A* on hex). Render the grid visually (matplotlib or pygame — you'll want to watch your agents play). This alone is a satisfying mini-project. Key resource: Red Blob Games' hex grid guide (the gold standard).

_Week 2 — Game State Machine_ Implement the full game loop: resource generation → diplomacy signals → order submission → simultaneous resolution → fog of war update → victory check. Write it as a proper Gym-compatible environment with `reset()`, `step()`, `render()`. The simultaneous move resolution is the trickiest part — when two units try to enter the same hex, you need clean conflict rules.

_Week 3 — Action/Observation Spaces_ This is where you'll iterate a lot. The observation space needs to encode: your visible hex grid (terrain, ownership, units), your resource count, your unit positions, and the diplomacy signals received. The action space is combinatorial — each unit gets a move order, plus you decide what to build and where, plus diplomacy signals. You'll likely need a multi-head action approach (one output per unit + build action + diplomacy action).

_Week 4 — Testing & Baselines_ Write rule-based bots: a random agent, a greedy-expansion agent, a turtle-and-defend agent. Play thousands of games between them. This validates your env (no bugs, no degenerate outcomes) and gives you baselines to beat. If greedy always wins, your game design needs rebalancing. Iterate on map gen, resource rates, unit costs until games are interesting.

**Phase 2: First RL Agents (Weeks 5-7)**

_Week 5 — State Representation & DQN_ Your first real RL agent. Encode the hex grid as a multi-channel 2D array (think of it like an image — channel 1 is terrain, channel 2 is ownership, channel 3 is unit counts, channel 4 is fog). Use a CNN to process it. Start with a simplified version: 2 players, small map (~30 hexes), no fog, no diplomacy. Just learn to expand and fight. Implement DQN from scratch — replay buffer, target network, epsilon-greedy.

_Week 6 — Curriculum Learning_ Scale up gradually. 2 players on small map → 2 players on medium map → 3 players → 4 players. Each step introduces new challenges (multi-front wars, kingmaker dynamics). You'll find training breaks at each transition — that's expected. Key technique: train against a mix of old checkpoints + current self-play (avoids strategy cycling).

_Week 7 — Reward Shaping Deep Dive_ This is where the real iteration happens. Raw "win/lose" reward is too sparse — agents won't learn. You'll need intermediate rewards: territory gained, resources collected, enemy units killed, survival bonus. But be careful: badly shaped rewards create degenerate behavior (e.g., agents hoard resources and never attack). Expect to redesign rewards 3-4 times. This is the phase where AI can't help you — it requires watching replays, diagnosing failure modes, and intuiting what's going wrong.

**Phase 3: Advanced Algorithms (Weeks 8-10)**

_Week 8 — PPO Implementation_ Move from DQN to PPO. Implement from scratch: actor-critic architecture, GAE (Generalized Advantage Estimation), clipped objective, value function loss. The CNN backbone stays the same — you're swapping the learning algorithm. PPO handles the continuous/large action spaces much better than DQN.

_Week 9 — Multi-Agent PPO + Self-Play_ The hardest training challenge. With 4 agents learning simultaneously, the environment is non-stationary from each agent's perspective. Techniques to try: shared parameters (all agents use the same network — they differentiate through observation), population-based training (maintain a pool of agents at different skill levels), and league training (inspired by AlphaStar — agents specialize as exploiters vs generalists).

_Week 10 — Communication & Diplomacy Learning_ Re-enable the diplomacy signals. Now agents must learn when to propose peace (to focus on a different front), when to threaten (to deter attack), and when to ignore signals entirely. This is emergent communication — you're not telling them what the signals mean, they figure it out through play. Analyze: do agents learn that "peace" actually means peace? Or do some agents learn to send "peace" and then backstab?

**Phase 4: Analysis & Polish (Weeks 11-14)**

_Week 11 — Spatial Strategy Visualization_ Build tools to analyze what your agents learned. Heatmaps of where agents choose to expand first. Attention maps showing what part of the grid the CNN focuses on. Strategy clustering: do different training runs produce different "personalities" (aggressive vs defensive vs economic)?

_Week 12 — Fog of War & Scouting Behavior_ Re-enable fog of war. This forces agents to learn scouting — sending cheap units to explore before committing armies. This is a POMDP now, much harder. Do agents learn to scout? Do they learn to hide army movements from enemy scouts? This phase alone could be a project.

_Week 13 — Tournament & Ablation Studies_ Run a round-robin tournament: PPO agents vs DQN agents vs rule-based bots. Measure Elo ratings. Run ablations: how much does diplomacy help? How much does fog of war change strategies? How sensitive is training to reward shaping? Write this up properly.

_Week 14 — Writeup & Demo_ Polish the visualization (a clean pygame or web-based replay viewer). Write a proper README/report documenting your design decisions, training curves, emergent behaviors, and lessons learned. Record a few replays of interesting games (agent forms alliance, then betrays at the right moment).

---

### Key Papers/Resources to Read

For each phase, a couple of targeted reads:

- **Hex grids:** Red Blob Games hex grid tutorial (the only resource you need)
- **DQN:** Original DeepMind Atari paper (Mnih et al. 2015) — implement from this
- **PPO:** Schulman et al. 2017 — the original paper is surprisingly readable
- **Multi-agent:** "Emergent Complexity via Multi-Agent Competition" (Bansal et al. 2018)
- **Self-play:** OpenAI Five blog post + AlphaStar paper (Vinyals et al. 2019) for league training ideas
- **Emergent communication:** "Learning to Communicate with Deep Multi-Agent RL" (Foerster et al. 2016)
- **Spatial RL:** "Relational Deep RL" (Zambaldi et al. 2019) — if you want to try GNNs instead of CNNs for the grid

---

### Why This Can't Be One-Shotted

An AI can generate you a hex grid class. It can write a DQN template. But it absolutely cannot: tune the game balance so games are interesting (requires playing hundreds of games and watching replays), debug why your PPO agents collapse into passive strategies (requires analyzing training curves + reward diagnostics specific to your env), figure out the right reward shaping (requires iterating based on emergent behavior), or make self-play training stable with 4 agents (requires curriculum design specific to what your agents are doing wrong). Each of these is a multi-day debugging loop that requires human judgment.

Want me to dive deeper into any specific phase, or should I sketch out the hex grid coordinate system and environment API to get you started right now?