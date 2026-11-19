# Wearable Biomechanics

> Project by: Multidisciplinary AI Research Centre (MARC), University of Peradeniya

## Overview

This project implements an integrated wearable biomechanics system that combines multi-node IMU sensor arrays, video-based 3D pose estimation, and physics-informed machine learning to monitor musculoskeletal health, quantify internal joint stresses, and support injury prediction and rehabilitation.

---

## Repository Branches

This repository is organized into dedicated branches by purpose:

| Branch | Purpose |
| --- | --- |
| `main` | Clean entry point containing only this `README.md` with project overview and branch navigation. |
| `cleannjerk_dataset` | Active clean workspace for the Clean & Jerk dataset pipeline (dataset management, IMU cleanup, temporal synchronization, 3D pose estimation, dataset summary/auditing, and video annotation). |
| `old-work` | Archive of the earlier repository state, including legacy experiments (`Device_Performances/`, `Entropy_Analysis/`, `Other/`) and prior workspace files. |
| `web` | Web-related resources and frontend/interface work. |

---

## Getting Started

Switch to the active dataset branch to work on the Clean & Jerk pipeline:

```bash
git checkout cleannjerk_dataset
```

To browse archived experiments and earlier code:

```bash
git checkout old-work
```

