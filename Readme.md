# Mann Neural Network Engine

A handcrafted C++ neural network engine with ImGui-powered interactive UI, designed and built entirely from scratch. All mathematical operations—including matrix multiplication, network forward and backpropogation, and training logic—are implemented manually, offering full control and understanding of neural network inner workings. This project runs and visualizes feedforward networks for the MNIST digit dataset.

---

## Features

- Fully custom neural network engine (no external ML frameworks).
- Manual implementation of matrix math, layer computations, and training algorithms.
- Configurable models: set layer sizes, learning rate, batch size, and more.
- Interactive UI (ImGui) for workflow:
  - **Models Select/Create Window**: List of all the models and option to create new models.
  - **Training Window**: To train the model. Real-time graphs for accuracy vs. epoch, loss vs. epoch, and batch accuracy.
  - **Testing Window**: Predict digits from MNIST images, with output visualization.
  - **Canvas Testing**: Draw a digit, instantly infer and display predictions.
  - **Network Visualizer**: See all nodes, weights, biases in the trained network in one render (WARNING!! - highly CPU intensive).
  - **Log Output**: Displays events and messages.
  - **Profiler**: Monitors CPU, GPU, RAM usage.
- Models and results are saved/loaded from disk on startup.
- Matrix multiplications are currently multi-threaded, with planned Metal GPU support for acceleration.

---

## Model Architecture

- Feedforward, fully-connected neural networks.
- Layer sizes are user-defined (e.g., `784 -> 60 -> 15 -> 10` for MNIST).
- Adjustable hyperparameters: learning rate, batch size.
- Output layer: 10 classes, Sigmoid for digit classification (MNIST).
- All operations (forward, backward, training) manually programmed.


---

## Training Details

- **Dataset**: MNIST (Handwritten Digits)
- **Optimizer**: Custom implementation
- **Activation Function**: Sigmoid
- **Learning Rate, Batch Size**: Can be edited while training

Training/testing windows provide live graphs of accuracy and loss.

---

## Installation

1. Clone the repo:
    ```
    git clone https://github.com/YOURUSERNAME/mann-neural-network.git
    ```
2. Install dependencies:
    - imgui
    - implot
    - glfw
3. Install tools:
    - cmake
    - vlang
4. Build and run:
    - Create a `build/` folder.
    - Place `run.v` file inside `build/`.
    - In `build/` directory, run:
       ```
       v run run.v
       ```

---

## Screenshots

- Will Add Later!!!

---

## Acknowledgments

- imgui, implot, glfw library authors
- MNIST dataset providers
