# UserExpererienceNeuralNetwork

# Neural UI Widget Selection Engine

## Overview

This project implements a neural network that automatically selects the optimal UI widget based on:

- Data semantics (type, cardinality, volume)
- Task type (selection, exploration, comparison)
- Interaction context (device)
- Aggregation level

It models UI design as a **learning problem**:

> (data + task + context) → optimal widget

---

## Features

- Synthetic dataset generator
- Neural network (PyTorch)
- Multi-class widget prediction
- End-to-end pipeline in a single file

---

## Widget Types

- dropdown
- radio
- slider
- textbox
- autocomplete
- table
- chart

---

## Input Features

| Feature        | Description |
|----------------|-------------|
| data_type      | nominal / ordinal / numeric |
| cardinality    | number of unique values |
| volume         | dataset size |
| aggregation    | raw vs aggregated |
| task_type      | select / explore / compare |
| device         | mobile / desktop |

---

## Installation

```bash
pip install torch numpy
