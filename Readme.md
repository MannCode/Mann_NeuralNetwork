# Mann Neural Network

A desktop neural network studio written in C++ with ImGui for intuitive GUI interaction. Design, configure, train, test, and visualize feedforward neural networks using the MNIST dataset. Models, logs, and training graphs are stored on disk and loaded at startup.

---

## Features

- Create neural network models, configure layers, learning rates, batch size.
- Four main windows:
  - **Training**: Train selected model; three live graphs (accuracy/epoch, loss/epoch, batch accuracy).
  - **Testing**: Test prediction on images from training/testing dataset; bar graph output.
  - **Canvas Testing**: Draw a digit on canvas and predict with NN.
  - **Network Visualizer**: Visualizes full network architecture (nodes, weights, biases).
- Other windows:
  - **Log Output**: View log messages and events.
  - **Profiler**: See CPU, GPU, RAM usage.
- Matrix multiplication is multi-threaded (conversion to Metal GPU planned).
- All session/model data persists in files (auto-load on startup).

---

## Model Architecture

- Configurable fully-connected neural networks.
- Layers can be set (e.g. `784 → 60 → 15 → 10 for MNIST`).
- Parameters: learning rate, batch size, epochs.
- Activation functions: (specify if applicable).
- Output layer: 10 classes, Softmax for digit classification.


---

## Training Details

- **Dataset**: MNIST (Handwritten Digits)
- **Optimizer**: (specify, e.g., SGD, Adam if implemented)
- **Loss function**: Cross-entropy
- **Learning Rate**: User configurable
- **Batch Size**: User configurable
- **Epochs**: User configurable

**Sample training results:**
| Model      | Training Accuracy | Test Accuracy |
|------------|------------------|--------------|
| DriftNet   | 91.71%           | 91.45%       |
| Madwarhead | 81.92%           | 80.72%       |
| Axionflow  | 9.87%            | 9.80%        |

---


---

## Installation

1. Clone the repo:
    ```
    git clone https://github.com/YOURUSERNAME/mann-neural-network.git
    ```
2. Download dependencies:
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

- Training, testing, and visualization:
  ![](./path/to/screenshot1.png)
  ![](./path/to/screenshot2.png)

---

## License

Add your project license here (e.g., MIT, Apache).

---

## Acknowledgments

- imgui, implot, glfw authors
- MNIST dataset providers

