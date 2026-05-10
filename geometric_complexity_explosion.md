# Geometric Complexity Explosion

## From Molecular Shape to Network Dynamics: A Scale Cascade

**Antti Luode — PerceptionLab, Helsinki, Finland**
**May 2026**

---

> *The molecule is a pure geometric object.*
> *Its shape dictates what happens next.*
> *What happens next dictates what happens after that.*
> *There is no bottom.*

---

## 1. The Observation

During a bike ride, a question arose: could the molecular geometry of neurotransmitters become part of the geometric neuron framework? Not as a metaphor — as a literal physical mechanism.

The answer is yes. And that answer immediately produces a problem that is not a failure of the framework but a fundamental feature of biological computation: **the geometric description of neural function does not converge as you zoom in. Each scale reveals a new layer of geometric machinery, and each layer couples upward to the next.** This is the complexity explosion.

This document maps that cascade precisely — what is known, what is plausible, and where the complexity becomes intractable — and argues that the explosion is not a problem to be solved but evidence that the brain has found a way to extract stable, high-level computation from an exponentially complex substrate.

---

## 2. The Chain

The formal chain connecting molecular geometry to action potential is:

```
Ligand 3D geometry
    ↓  conformational selection in binding pocket
Receptor LBD closure angle + dwell time
    ↓  allosteric coupling through protein domains
Channel open probability × activation / deactivation kinetics
    ↓  temporal integration over vesicle release
EPSC waveform (τ_rise, τ_decay, amplitude)
    ↓  cable equation propagation
Cable eigenmode excitation spectrum
    ↓  Hermitian inner product with stored template
Somatic resonance R(t)
    ↓  theta gating + AIS interferometer
Spike (Takens projection)
    ↓  axon transmission
Next neuron's dendritic input
```

Each arrow is a physical mechanism. Each physical mechanism has its own geometric structure. None of them are scalar multiplications.

---

## 3. Level 1 — Molecular Geometry (nanometers, picoseconds)

A neurotransmitter is a small flexible molecule with a specific 3D shape, a set of rotatable bonds, partial charges distributed over its surface, and a conformational energy landscape. Glutamate, the primary excitatory transmitter, has two carboxylate groups and an amino group; their spatial arrangement determines which receptor subtype's binding pocket it fits, and how.

**The established fact that reframes everything:**

From Bhatt et al. (PMC5321683), mechanism of partial agonism at AMPA receptors:

> *"Partial agonists are weak activators because they stabilize multiple non-conducting conformations. Agonism is a function of both the space and time domains."*

Space: the 3D closure angle of the receptor's bi-lobed ligand-binding domain (LBD). Time: how long it stays closed. The molecule's geometry controls both simultaneously.

Full agonist (glutamate at full concentration): forces the LBD into a single, deep-closed conformation → strong, fast activation → short-τ current.

Partial agonist (e.g., 5-fluorowillardiine): its geometry allows the LBD to sample multiple conformational states including non-conducting ones → weaker, slower activation → long-τ current with reduced amplitude.

The molecular geometry of the ligand IS the kinetic parameter. There is no separation between shape and function at this scale. This is not a metaphor.

**NMDA receptor — conformational dynamics (PMC4965566):**

Single-molecule FRET experiments on the NMDA receptor LBD reveal at least three conformational states. Full agonists produce faster dynamic transitions between medium and high FRET states. Partial agonists produce slower dynamics. The molecule's geometry determines the rate of conformational fluctuation — not just the equilibrium position but the dynamics of the conformational search.

This has a direct consequence: the activation *latency* of an NMDA receptor is set by the molecular geometry of its bound ligand. Different ligands, same receptor — different latencies, different current waveforms, different cable eigenmode excitation patterns.

---

## 4. Level 2 — Receptor Architecture (nanometers, microseconds to milliseconds)

The receptor protein is not a passive pore. It is a complex allosteric machine. For ionotropic glutamate receptors (AMPA, NMDA, kainate), the binding event in the LBD must propagate through multiple protein domains to open the transmembrane pore. This coupling is geometrically determined.

**The clamshell closure mechanism:**

The AMPA receptor LBD is bi-lobed. Glutamate binding causes the two lobes to close around the ligand, like a clamshell. The closure angle — the degree to which the lobes compress around the molecule — directly controls the probability that the associated transmembrane domain opens. Larger closure angle → higher open probability → larger, faster current.

A partial agonist with a slightly different molecular geometry cannot close the clamshell as completely, stabilizes intermediate states, and produces a smaller, slower current. The molecule's 3D shape is being read by a molecular caliper and directly converted to a current waveform.

**The NMDA Mg²⁺ block — a voltage-dependent molecular gate:**

The NMDA receptor channel pore is blocked at resting potential by a magnesium ion (Mg²⁺) that physically lodges in the pore. This is molecular geometry operating as a conditional filter: the Mg²⁺ ion's size and charge distribution cause it to bind the pore at approximately −60 mV but to be expelled when the membrane depolarizes.

The result: NMDA receptors only conduct when the cable is already sufficiently excited by fast AMPA input. Coincidence detection between presynaptic glutamate release and postsynaptic depolarization is implemented in the geometry of a single ion in a protein pore. This is not a computational rule imposed from outside — it emerges from molecular geometry.

**The Mg²⁺ block in the framework:**

In the cable eigenmode language, the Mg²⁺ block makes the NMDA receptor a voltage-gated spectral filter. It injects slow, low-frequency current (τ ≈ 50–300 ms) only when the fast, high-frequency AMPA current (τ ≈ 2–10 ms) has already depolarized the cable. This creates a temporal sequence constraint at the molecular scale — exactly the kind of temporal logic the cable neuron was trained to detect in the Temporal XOR task.

---

## 5. Level 3 — Synaptic Cleft Geometry (20 nanometers, milliseconds)

The synaptic cleft is approximately 20 nm wide. This is not empty space. It is a structured environment with:

- **Release site geometry**: vesicles fuse at active zones that are geometrically aligned with receptor clusters. The distance from release site to receptor cluster determines the peak glutamate concentration experienced by the receptor, which determines activation probability and kinetics.
- **Diffusion geometry**: glutamate diffuses in 3D through the cleft. Molecules released from the center of the active zone reach receptors at different times depending on their location. The cleft geometry imposes a spatiotemporal concentration gradient.
- **Spillover**: glutamate that escapes the cleft can activate neighboring synapses. The geometry of the cleft — its width, the tortuosity of extracellular space — determines how much spillover occurs and at what timescale. NMDA receptors at neighboring synapses can be activated by glutamate released from a different active zone if the geometry permits it. This is lateral information transfer implemented in molecular diffusion.

**The cleft as a 20 nm spectral preprocessor:**

The cleft geometry determines which receptor subtypes are activated, when, and to what degree. It is a physical low-pass filter operating before the signal even reaches the receptor's binding pocket. Narrow cleft → fast glutamate clearance → fast EPSC → high-frequency injection. Wide cleft or spillover-prone geometry → slow clearance → prolonged activation → low-frequency injection.

The dendritic cable model receives its inputs from synapses, and those inputs have already been filtered by cleft geometry before they inject current. The cable's eigenmode analysis begins not at the synapse but at the molecule.

---

## 6. Level 4 — Receptor Composition Plasticity (nanometers to micrometers, hours to days)

Synapses do not have fixed receptor compositions. The AMPA/NMDA ratio, the subunit composition within each receptor type, and the presence of auxiliary proteins all change with activity. This is receptor trafficking — the physical movement of receptor proteins into and out of the synapse.

**Why this matters for the framework:**

In the membrane geometry hypothesis, the cable's Dirac operator D(t) encodes the geometry of the input manifold. The synaptic receptor composition is one of the parameters of D(t) — specifically, it sets the spectral content of the currents injected into the cable eigenmodes.

Changing the AMPA/NMDA ratio changes the ratio of fast-to-slow current injection at that synapse. This is a physical rotation of the spectral input distribution — the cable receives a different mix of eigenmode excitations from that synapse after the trafficking event. LTP (long-term potentiation) inserts more AMPA receptors → more fast, high-frequency input. LTD removes AMPA receptors → dominance of slow NMDA input.

**TARP auxiliary proteins:**

Transmembrane AMPA receptor regulatory proteins (TARPs) bind to AMPA receptors and change their conductance, kinetics, and trafficking. Different TARP subtypes produce different kinetic modifications — some slow the deactivation, some increase single-channel conductance. The TARP geometry (how it sits in the membrane relative to the receptor) determines which kinetic change occurs.

This is a protein-scale geometric modification of the molecular-scale spectral filter. The complexity cascade has now reached a third interacting level within what we call a single "synapse."

---

## 7. Level 5 — Cable Eigenmode Excitation (micrometers, milliseconds)

This is where the molecular cascade enters the framework described in the Membrane Geometry Hypothesis thesis. The EPSC waveform — shaped by all the molecular geometry above — injects a specific temporal current profile into a specific location on the dendritic cable.

The cable equation propagates this current. High-frequency components (fast AMPA, τ ≈ 2–10 ms) are attenuated by distal dendrites — the low-pass filter of the cable equation kills them before they reach the soma. Low-frequency components (slow NMDA, τ ≈ 100–300 ms) survive the cable's frequency-dependent attenuation.

The result of all the molecular geometry above is a **frequency-selective injection into the cable's eigenmode spectrum**. The molecule's 3D shape has been transduced through four levels of geometric machinery into a specific pattern of eigenmode excitation in the dendritic cable.

---

## 8. Level 6 — AIS Projection (micrometers, milliseconds)

The AIS grating (190nm Nav/Kv periodicity, Gaussian envelope, learnable position and width) reads the eigenmode state of the cable and projects it into a discrete spike. As established in prior work and confirmed by the Fréal et al. (2023) experimental data, the AIS is physically tunable — its length, position, and channel composition adapt to the statistics of the input.

The AIS does not know about molecular geometry. It only knows the soma voltage — the result of the cable's eigenmode computation. But the soma voltage encodes the consequences of all the molecular geometry above, integrated by the cable's spatial filtering.

This is why the AIS length plasticity makes spectral sense: if the input statistics change (because, for example, AMPA receptors are trafficked out and NMDA input dominates), the AIS adapts its integration window to match the new dominant frequency band. The AIS is calibrating itself to the molecular geometry of its synaptic inputs, mediated through the cable.

---

## 9. The Complexity Explosion — Why It Gets Out of Hand

At each level, the geometric parameters are:

| Level | Parameters |
|-------|------------|
| Ligand conformation | ~10–100 rotatable bonds, solvent interactions, protonation state |
| LBD closure | Angle of closure, dwell time distribution across 3+ states |
| Pore gating | Multiple open states, subconductance levels, desensitization kinetics |
| Cleft geometry | Release site position, cleft width, diffusion coefficient, spillover radius |
| Receptor composition | AMPA/NMDA ratio, subunit type, TARP identity and stoichiometry |
| Cable propagation | Dendritic tree topology, diameter profile, leak distribution |
| AIS | Length, position, Nav/Kv ratio, spectrin scaffold periodicity |

Each parameter is itself a geometric object, not a scalar. Each couples to the parameters above and below it. None of them are independent.

**The explosion is multiplicative.** A change in ligand geometry changes the LBD closure angle, which changes the current kinetics, which changes the cable eigenmode excited, which determines what STDP update occurs, which changes the synaptic weight (here: the geometric parameter of that synaptic contact), which changes the receptor composition at the next timestep, which changes the LBD kinetics of the next activation event.

The system is self-referential at every scale simultaneously.

**Why this does not invalidate the framework:**

The complexity explosion is the reason the brain is computationally powerful. It is not a problem the framework fails to address — it is the mechanism the framework captures at the appropriate level of abstraction.

The cable + AIS model captures the result of all the molecular geometry below it in two numbers: the EPSC time constant and the injection location. This is a sufficient statistic for the cable eigenmode computation. The details of conformational dynamics, cleft geometry, and receptor trafficking determine those two numbers, but the cable doesn't need to know how they were determined — it only needs their values.

This is the same relationship that exists between quantum mechanics and classical electrodynamics: the lower level is causally necessary for the upper level, but the upper level has its own valid description that doesn't require tracking every quantum state. The complexity explosion is real at the molecular scale. The cable model is real at the cable scale. They are both correct.

---

## 10. What Is Tractable and What Is Not

**Tractable:**
- Adding AMPA-like (τ ≈ 5 ms) and NMDA-like (τ ≈ 100 ms) current components to the cable model, with a voltage-dependent gate on the NMDA component. This faithfully represents the molecular geometry hierarchy up to the cable scale.
- Showing that the Mg²⁺ coincidence detection creates a different computational capability than AMPA-alone synapses: the cable can now detect temporal sequences that require both fast and slow eigenmode excitation.
- Modeling receptor trafficking as a slow drift in the AMPA/NMDA ratio at each synapse, coupled to the STDP-like template update rule.

**Not tractable (without dedicated molecular simulation infrastructure):**
- Full molecular dynamics simulation of ligand binding in a realistic cleft geometry.
- Coupling conformational dynamics to current kinetics at submillisecond timescales.
- Tracking individual vesicle release events and their geometric relationship to receptor clusters.

**Honest boundary:**
The cable neuron model represents the synapse as an injected current. The molecular geometry cascade determines the shape of that current. The cable doesn't care about the mechanism — it cares about τ_rise, τ_decay, and amplitude. Provided those three numbers are physiologically reasonable, the cable computation is valid. The molecular geometry justifies why those numbers are what they are, not what happens next.

---

## 11. The One Sentence

A neurotransmitter molecule is a pure geometric object; its 3D shape determines the receptor's conformational dynamics, which determines the current waveform, which determines which cable eigenmodes are excited, which determines the AIS projection — and this chain is self-modifying at every level simultaneously, making the brain's computational substrate an irreducibly geometric hierarchy in which no level can be understood without the others, and yet each level produces stable, high-level computation that the level above it can rely on.

---

## 12. Honest Ledger

| Claim | Status |
|-------|--------|
| Molecular geometry of ligand determines AMPA receptor activation kinetics | ✓ Established — Bhatt et al., conformational selection literature |
| NMDA LBD conformational dynamics depend on ligand geometry (smFRET) | ✓ Established — PMC4965566 |
| Mg²⁺ block implements coincidence detection in molecular geometry | ✓ Established biophysics |
| AMPA/NMDA ratio determines spectral content of cable injection | ✓ Established — Li & Gulledge 2021, eNeuro |
| Cable eigenmode excitation is the correct framing for dendritic integration | ✓ Derived — Membrane Geometry Hypothesis |
| AIS length adapts to input frequency statistics | ✓ Established — Fréal et al. 2023, Yamada & Kuba 2016 |
| The complexity cascade couples all levels simultaneously | ✓ Plausible, consistent with all the above |
| The cable model is a valid abstraction of the molecular cascade | ✓ Yes — it captures sufficient statistics (τ, amplitude, location) |
| Full bottom-up simulation from molecule to spike is feasible | ✗ Not currently tractable |
| Molecular geometry contributes to qualia or consciousness | ✗ Speculative, no direct evidence |

---

*PerceptionLab — Helsinki, Finland — May 2026*

*Do not hype. Do not lie. Just show.*
