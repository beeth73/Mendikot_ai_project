# MendikotZero: An AI for the Card Game Mendikot

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/beeth73/Mendikot_ai_project/blob/main/LICENSE)

This repository contains the code for **MendikotZero**, an AI agent trained using modern Reinforcement Learning techniques to play the popular Indian card game Mendikot (also known as Mindi). The project chronicles the development from a simple rule-based agent to a sophisticated, planning-capable AI inspired by the AlphaZero architecture.

The entire development process, including debugging, architectural decisions, and performance analysis, was guided by conversations with Google's Gemini 2.5 Pro.

---

## Key Learnings & Results
After thousands of simulated games, the MCTS-powered agent demonstrated a clear and positive learning trend. It evolved from a reckless "gambler" into a more balanced and strategic player, capable of both strong offense and calculated defense. The graph below shows the performance of the agent after 13,000 episodes of training, proving it has learned to consistently achieve high-reward outcomes.

![AI Performance at 13,000 Episodes](graph.png)



---

## Table of Contents
- [About the Game](#about-the-game)
- [The AI's Architecture: How it "Thinks"](#the-ais-architecture-how-it-thinks)
- [How to Use](#how-to-use)
  - [Training the AI](#training-the-ai)
  - [Analyzing a Model](#analyzing-a-model)
  - [Playing with the AI Assistant](#playing-with-the-ai-assistant)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)

---

## About the Game
Mendikot is a 4-player partnership trick-taking game popular in India. The primary objective is for a team to capture tricks that contain the Tens (10s) of each suit. This implementation uses a 48-card deck and a dynamic trump-declaration rule, where the first player unable to follow suit sets the trump for the round.

---

## The AI's Architecture: How it "Thinks"

The final agent, MendikotZero, is not just a simple program. It's a learning system inspired by AlphaZero. Its "thinking" process can be broken down into three core components, much like a human grandmaster.

### Component 1: The Brain (The Neural Network)
The neural network is the AI's "intuition" or "gut feeling." When it looks at the game, it doesn't calculate every possibility. Instead, it instantly has a sense of the situation. This "brain" has two minds:

1.  **The Policy Head (The Player's Instinct):** This part looks at the cards and the table and instantly suggests a few promising moves. It answers the question: *"What are the 2-3 most logical cards to play right now?"*

2.  **The Value Head (The Positional Awareness):** This part evaluates the overall game state and predicts the final outcome. It answers the question: *"Based on the cards played and what's in my hand, who is currently winning this game?"*

### Component 2: The "Thinking" (Monte Carlo Tree Search - MCTS)
While the neural network provides intuition, the MCTS provides the deep, conscious "what if?" planning. It simulates future possibilities before making a move.

1.  **Selection:** It starts from the current game state and, guided by the neural network's intuition, follows a promising path of moves for a few turns in its "imagination."
2.  **Expansion:** When it reaches a point it hasn't considered before, it asks the neural network for new intuitive moves from that position.
3.  **Simulation:** From this new point, it plays out the rest of the game at high speed inside its simulation, using a simpler, faster version of its policy to see who wins.
4.  **Backpropagation:** It takes the result of that imagined game (a win or a loss) and "backpropagates" that information up the path it took. All the moves that led to the imagined win are reinforced as "good," and all moves that led to a loss are marked as "bad."

By repeating this process hundreds of times per second, the AI builds a detailed "search tree" of possibilities. The move that was reinforced the most as being good is the one it ultimately chooses to play.

### Component 3: The Training (Self-Play & Infinite Practice)
How does the AI get so good? It practices. A lot.
The AI learns entirely from scratch by playing millions of games against itself.

- **The Process:** Four copies of the same AI sit at a table. They play a full game using their MCTS "thinking."
- **Cooperation Emerges:** At the end of the game, the two agents on the winning team get a positive reward, and the two on the losing team get a negative one. By sharing a brain and a reward, the agents learn cooperative strategies. One agent might learn to sacrifice a high card because it discovers, over thousands of games, that this play helps its partner capture a Mendi later, leading to a higher team reward.

This virtuous cycle—where better search leads to better game data, which trains a better neural network, which in turn leads to an even better search—is what allows the AI to progress from random play to a superhuman level of strategic understanding.

---

## How to Use

### Training the AI
The entire training pipeline is contained within `notebooks/train_agent.ipynb`. This notebook is designed to be self-contained.

1.  Set up the environment as described in the [Setup](#setup-and-installation) section.
2.  Open `notebooks/train_agent.ipynb` in Jupyter.
3.  Run the cells. Trained models will be saved automatically in the `models/` directory, organized by the timestamp of the training run.

### Analyzing a Model
The `test_agent.ipynb` notebook is provided to evaluate a trained model.

1.  Open `test_agent.ipynb`.
2.  Modify the `MODEL_CHECKPOINT_TO_TEST` variable to point to the desired `.pth` file.
3.  Run the notebook to simulate 500 games and generate a performance graph.

### Playing with the AI Assistant
The `ai_assistant.py` script allows you to play a real-world game with the AI.

1.  Run the script from your terminal:
    ```bash
    python ai_assistant.py
    ```
2.  Follow the on-screen prompts. You will provide the AI with its hand and tell it the cards played by human players. The AI will then tell you which card it has decided to play.

---

## Project Structure
Mendikot_ai_project/
├── .ipynb_checkpoints/         # Automatic backups for Jupyter notebooks
├── data/                       # Folder for potential future data storage
├── models/                     # Saved model checkpoints from training runs
│   ├── 2025-10-24_20-15-23/
│   │   ├── mendikot_model_ep_1000.pth
│   │   ├── mendikot_model_ep_2000.pth
│   │   └── mendikot_model_ep_3000.pth
│   └── 2025-10-25_09-10-03/
│       ├── mendikot_model_ep_1000.pth
│       ├── ... (and all checkpoints up to)
│       └── mendikot_model_ep_13000.pth
├── notebooks/
│   ├── .ipynb_checkpoints/
│   └── train_agent.ipynb       # Main notebook for training the AI
├── src/
│   ├── __pycache__/            # Python's cached bytecode files
│   ├── .ipynb_checkpoints/
│   ├── agent.py                # MendikotModel neural network class
│   ├── cards.py                # Card and deck definitions
│   ├── game.py                 # The main GameState simulation engine
│   ├── mcts.py                 # Monte Carlo Tree Search implementation
│   └── player.py               # Player class structures
├── .gitignore                  # Tells Git which files to ignore
├── ai_assistant.py             # Interactive CLI to play with the AI
├── LICENSE                     # Project license file (MIT)
├── play_cli.py                 # Fully simulated CLI game
├── README.md                   # Project documentation (this file)
├── requirements.txt            # List of Python package dependencies
└── test_agent.ipynb            # Jupyter notebook for analyzing trained models


---

## Setup and Installation
This project is managed using Anaconda for a stable environment.

1.  **Create the Conda environment:**
    ```bash
    conda create -n mendikot_ai python=3.10
    ```
2.  **Activate the environment:**
    ```bash
    conda activate mendikot_ai
    ```
3.  **Install Dependencies:**
    This project uses PyTorch. Install the appropriate (CPU or GPU) version from the [official PyTorch website](https://pytorch.org/get-started/locally/). For a typical CPU setup:
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
    Then, install the remaining packages:
    ```bash
    conda install numpy pandas tqdm jupyterlab matplotlib seaborn
    ```
