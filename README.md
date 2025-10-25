# MendikotZero: An AI for the Card Game Mendikot

This repository contains the code for **MendikotZero**, an AI agent trained using modern Reinforcement Learning techniques to play the popular Indian card game Mendikot (also known as Mindi). The project chronicles the development from a simple rule-based agent to a sophisticated, planning-capable AI inspired by the AlphaZero architecture.

The entire development process, including debugging, architectural decisions, and performance analysis, was guided by conversations with Google's Gemini.

## Table of Contents
- [About the Game](#about-the-game)
- [Project Overview](#project-overview)
- [Final AI Architecture](#final-ai-architecture)
- [How to Use](#how-to-use)
  - [Training the AI](#training-the-ai)
  - [Analyzing a Model](#analyzing-a-model)
  - [Playing with the AI Assistant](#playing-with-the-ai-assistant)
- [Key Learnings & Results](#key-learnings--results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)

## About the Game
Mendikot is a 4-player partnership trick-taking game popular in India. The primary objective is for a team to capture tricks that contain the Tens (10s) of each suit. This implementation uses a 48-card deck and a dynamic trump-declaration rule, where the first player unable to follow suit sets the trump for the round.

## Project Overview
This project was an iterative journey to build a strong Mendikot AI. We progressed through several versions:

1.  **Simple A2C Agent (v1):** An initial agent trained with a simple Actor-Critic algorithm on a flawed set of rules. It learned basic card values but had no strategic depth.
2.  **Rule-Aware A2C Agent (v2):** The game environment was perfected with all core rules (following suit, trump declaration, etc.). The AI learned to play legally but still lacked foresight and partnership skills.
3.  **MendikotZero (v3 - Final):** The final version, which implements a powerful **Monte Carlo Tree Search (MCTS)** algorithm. This agent has:
    *   **Forward Planning:** It "thinks" ahead by running hundreds of simulations before each move.
    *   **Perfect Memory:** It counts cards and remembers which player played which card and who is void in which suit.
    *   **Creative Exploration:** It uses Dirichlet noise during training to discover novel and robust strategies.

## Final AI Architecture
The final agent uses a methodology inspired by AlphaZero:
- **Neural Network:** A deep neural network with two heads:
    1.  **Policy Head:** Provides the "intuition" to guide the MCTS search.
    2.  **Value Head:** Evaluates game positions, predicting the eventual winner.
- **Monte Carlo Tree Search (MCTS):** The "thinking" component. It builds a search tree of possible future moves, using the neural network to intelligently explore the most promising paths.
- **Self-Play:** The AI learns entirely from scratch by playing millions of games against itself, with two AI agents on the same team learning to cooperate by sharing a final team reward.

## How to Use

### Training the AI
The entire training pipeline is contained within `notebooks/train_agent.ipynb`. This notebook is designed to be self-contained and can be run in a local Jupyter environment or on Google Colab.

1.  Set up the environment as described in the [Setup](#setup-and-installation) section.
2.  Open and run the cells in `notebooks/train_agent.ipynb`.
3.  Trained models will be saved automatically in the `models/` directory, organized by the timestamp of the training run.

### Analyzing a Model
A separate analysis notebook, `test_agent.ipynb`, is provided to evaluate the performance of a trained model checkpoint.

1.  Open the `test_agent.ipynb` notebook.
2.  Modify the `MODEL_CHECKPOINT_TO_TEST` variable to point to the desired `.pth` file.
3.  Run the notebook. It will simulate 500 games, pitting the AI and its AI partner against two random agents, and generate a performance graph.

### Playing with the AI Assistant
The `ai_assistant.py` script allows you to play a real-world game of Mendikot with the AI as a player. You act as the game master.

1.  Run the script from your terminal:
    ```bash
    python ai_assistant.py
    ```
2.  Follow the on-screen prompts. You will tell the AI its seating position and the cards in its hand.
3.  During the game, you will input the cards played by the human players, and the AI will tell you which card it has decided to play on its turn.

## Key Learnings & Results
The development process provided several key insights:
- The MCTS agent significantly outperforms the simpler A2C agents, demonstrating the power of search-based planning.
- The AI successfully learned complex strategies, such as resource management, safe Mendi plays, and calculated risk-taking.
- Performance is highly dependent on the quality of the game simulation and the richness of the state representation (especially card-counting memory).
- Evaluation showed a clear learning trend, with the AI evolving from a reckless "gambler" to a more balanced and strategic "grinder" as training progressed.

(Mendikot_ai_project
/graph.png)

## Project Structure
Mendikot_ai_project/
├── models/ # Saved model checkpoints (.pth files)
├── notebooks/ # Jupyter notebooks for training and analysis
├── src/ # Core Python source code
│ ├── agent.py # MendikotModel neural network class
│ ├── cards.py # Card and deck definitions
│ ├── game.py # The main GameState simulation engine
│ └── mcts.py # Monte Carlo Tree Search implementation
├── ai_assistant.py # Interactive CLI to play with the AI
├── play_cli.py # Fully simulated CLI game
└── README.md # This file

## Setup and Installation
This project is managed using Anaconda.

1.  **Create the Conda environment:**
    ```bash
    conda create -n mendikot_ai python=3.10
    ```
2.  **Activate the environment:**
    ```bash
    conda activate mendikot_ai
    ```
3.  **Install dependencies:**
    This project uses PyTorch. Install the appropriate (CPU or GPU) version from the official PyTorch website. For CPU:
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
    Then, install the remaining packages:
    ```bash
    conda install numpy pandas tqdm jupyterlab matplotlib seaborn
    ```
