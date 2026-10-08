# Sherlock AI Core Foundation

This directory contains the foundational, non-destructive data architecture for Sherlock AI.

## Architectural Principles

1. **Why Records are Immutable**
   A `Record` represents an exact observation from a source at a point in time. If we normalize or merge this data destructively, we lose the original evidence. By keeping records immutable, we maintain a pristine audit trail. Normalized data is derived and stored alongside, not on top of, raw data.

2. **Why Entity Clusters are Derived**
   Instead of physically merging nodes into a single "Entity" record (which destroys provenance), we logically group Records into an `EntityCluster`. An entity identity is a hypothesis based on current evidence. If new evidence contradicts a previous match, we simply update the cluster membership, without having to "un-merge" scrambled data.

3. **What Evidence Represents**
   `Evidence` represents the specific rationale behind a match decision (e.g., an exact email match, or a conflicting birth year). It has a polarity (`SUPPORTING`, `CONTRADICTING`, `NEUTRAL`) ensuring decisions are explainable and mathematically weighted, rather than being an opaque embedding similarity score.

4. **Why REVIEW Exists**
   Entity resolution in OSINT often encounters cases with insufficient data to definitively declare a MATCH or NON_MATCH. The `REVIEW` state prevents the system from forcing uncertain data into the knowledge graph, allowing human review or waiting for further evidence.

5. **How Evaluation Works**
   The `PairwiseEvaluator` compares predicted `MatchDecision` outputs against a known ground-truth dataset. It calculates precision, recall, and F1 scores. The treatment of the `REVIEW` state is explicitly configurable (e.g., penalizing it as a missed opportunity or treating it neutrally).

## Running Tests
To run the evaluation and domain model tests:
```bash
# From the repository root with pytest installed:
python -m pytest tests/
```
