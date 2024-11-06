# NN Inspector

An interactive web application for visualizing and exploring neural network architectures. A live demo of this app is also [available online](https://frasertheking.com/nn_app/). 

## Table of Contents

- [Features](#features)
- [Why is Model Interpretability Important?](#why-is-model-interpretability-important)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Configuration](#configuration)
- [JSON Schema](#json-schema)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Inspiration](#inspiration)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Features

- **Interactive Visualization**: Adjust parameters and observe real-time changes in model behavior.
- **Multiple Neural Network Support**: Explore various types of neural networks.
- **Custom Model Analysis**: Upload and analyze your own operational models.
- **Physical Model Interpretation**: Interpret simplified versions to understand model learning behavior.
- **Real-Time Feedback**: See immediate effects of network changes on performance.

---

## Why is Model Interpretability Important?

As machine learning models become increasingly complex, understanding their decision-making processes is crucial for:

- **Transparency**: Building trust in AI systems.
- **Bias Identification**: Detecting biases and ethical issues.
- **Performance Improvement**: Understanding failure modes to enhance models.
- **Regulatory Compliance**: Meeting explainability requirements.

However, interpretability is challenging due to model complexity. NN Inspector simplifies this by allowing users to focus on individual neurons and start with simple configurations.

---

## Use Cases

NN Inspector is designed for:

- **Students**: Learn how neural networks work in an interactive environment.
- **Educators**: Use as a teaching aid to demonstrate neural network concepts.
- **Researchers**: Experiment with different configurations and gain insights into model behaviors.

---

## Getting Started

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/nn-inspector.git
```

Navigate to the project directory:
```bash
cd nn-inspector
```

Install dependencies (if applicable):
```bash
npm install
```

### Usage

Start a local server and open the application in your web browser:

```python
# Using Python's HTTP server
python -m http.server

# Or using Node.js
npx http-server
```

Access the application at http://localhost:8000

---

## Configuration

Customize the application by modifying the user configuration options in the JavaScript file.

```javascript
let model_name = '3phase';
let input_vars = ["n0", "lambda", "Rho", "Fs", "Dm", "Temperature", "Relative Humidity", "Pressure"];
let log_scaled_vars = ['n0', 'lambda', 'Rho', 'Dm'];
let response_vars = ['Rain', 'Snow', 'Mixed-Phase'];
let model_loc = 'models/phase3/';
let test_filepath = 'data/test_data.csv';
let hiddenNeuronOptions = [1, 2, 4, 8, 16];
```

---

## JSON Schema

In order for the platform to parse your model data, you need to follow a specific JSON schema. We are looking to make many of these optional as time goes on. ___N___ is total epochs, ___M___ is number of predictors, ___J___ is number of outputs, ___K___ is number of hidden layer neurons

- **predictors**: Array of ___M___ strings  
  - **Description**: Names of the predictors used in the model.
  - **Example**: `["Temperature", "Relative Humidity", "Pressure"]`

- **units**: Array of ___M___ strings  
  - **Description**: Units corresponding to each predictor.
  - **Example**: `["K", "%", "hPa"]`

- **response_variable**: Array of ___J___ strings  
  - **Description**: Target classes for the model's predictions.
  - **Example**: `["Rain", "Snow", "Mixed-phase"]`

- **N_HIDDEN**: Integer  
  - **Description**: Number of hidden layers in the model.
  - **Example**: `1`

- **weights_history**: Array of arrays (multi-dimensional) ___[N x 2 x M x K]___
  - **Description**: Historical weights for each layer in the model over training epochs.

- **biases_history**: Array of arrays (multi-dimensional)  ___[N x 2 x K]___
  - **Description**: Historical biases for each layer in the model over training epochs.

- **activations_history**: Array of arrays (multi-dimensional)  ___[N x 2 x K]___
  - **Description**: Historical activations of each layer in the model over training epochs.

- **loss_history**: Array of ___N___ floats 
  - **Description**: Loss values recorded at each training epoch.
  - **Example**: `[1.084, 0.982, 0.920, ...]`

- **val_loss_history**: Array of ___N___ floats  
  - **Description**: Validation loss values recorded at each training epoch.
  - **Example**: `[1.002, 0.925, 0.904, ...]`

- **accuracy_history**: Array of ___N___ floats  
  - **Description**: Training accuracy recorded at each epoch.
  - **Example**: `[0.372, 0.547, 0.610, ...]`

- **val_accuracy_history**: Array of ___N___ floats  
  - **Description**: Validation accuracy recorded at each epoch.
  - **Example**: `[0.448, 0.492, 0.490, ...]`

- **scaling_info**: Object  
  - **Description**: Information for scaling input features.
  - **Fields**:
    - **mean**: Array of ___M___ floats  
      - **Example**: `[0.181, 90.420, 1012.608]`
    - **scale**: Array of ___M___ floats  
      - **Example**: `[0.236, 6.134, 9.546]`

- **hyperparameters**: Object  
  - **Description**: Hyperparameters used for model training.
  - **Fields**:
    - **OPTMIZER**: String (e.g., `"ADAM"`)
    - **LR**: Float (e.g., `0.0005`)
    - **LOSS_FUNC**: String (e.g., `"categorical_crossentropy"`)
    - **ACTIVATION**: String (e.g., `"relu"`)
    - **BATCH_SIZE**: Integer (e.g., `32`)
    - **L1_flag**: Boolean (e.g., `true`)
    - **L1**: Float (e.g., `0.01`)

---

## TensorFlow Callback

Below is an example for running and saving model data using TensorFlow:

```python
import json

# Load your model data
with open('model_data.json', 'r') as file:
    model_data = json.load(file)

# Perform operations on the model data
# ...

# Save the updated model data
with open('model_data_updated.json', 'w') as file:
    json.dump(model_data, file)
```

---

## Roadmap

The application is currently under development with the following planned improvements:

- UI Enhancements: Improving the user interface for better usability.
- Regression Models: Allowing for regression models in addition to classification.
- Model Switching: Enhancing capabilities to switch between models seamlessly.
- Error Dialogues: Implementing better error handling and messaging.
- Multiple Hidden Layers: Supporting neural networks with multiple hidden layers.
- Advanced Statistics: Providing more detailed statistics and analytical tools.

---

## Inspiration

The project draws inspiration from:

- Daniel Smilkov and Shan Carter: For their work on interactive tools like TensorFlow Playground that make complex machine learning concepts accessible.
- Chris Olah and Colleagues at Anthropic: Especially their work on Toy Models of Superposition in interpretability research.

---

## Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes and push the branch.
- Submit a pull request for review.

---

## Contact
Fraser King, University of Michigan
