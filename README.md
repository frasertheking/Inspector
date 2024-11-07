<div align="center">

![logo](https://github.com/frasertheking/inspector/blob/main/images/logo.png?raw=true)

**NN Inspector** is an interactive web app for visualizing and exploring neural network architectures, maintained by [Fraser King](https://frasertheking.com/)

A live demo of this app is [available online](https://frasertheking.com/nn_app/). 

</div>


## Features
What can this tool do?

- **Interactive Visualization**: Adjust parameters and observe real-time changes in model behavior.
- **Multiple Neural Network Support**: Explore various types of neural networks (classifiers and regressors*).
- **Custom Model Analysis**: Upload and analyze your own operational models with real world data.
- **Physical Model Interpretation**: Interpret simplified versions to understand model learning behavior.
- **Real-Time Feedback**: See immediate effects of network changes on performance test data.
- **Education**: The visual and interactive nature of this tool makes it excellent for educational purposes.


## Why is Model Interpretability Important?

As machine learning models become increasingly sophisticated, understanding their decision-making processes is crucial for:

- **Transparency**: Building trust in AI systems.
- **Bias Identification**: Detecting biases and ethical issues.
- **Performance Improvement**: Understanding failure modes to enhance model reliability.
- **Regulatory Compliance**: Meeting explainability requirements.

While we don't promise a full mechanistic interpretability workflow, this tool can be useful for various explainable AI projects and facilitates movement towards an interpretable understanding of model behaviour. NN Inspector simplifies network complexity by allowing users to focus on individual neurons and start with simple configurations.

![training](https://github.com/frasertheking/inspector/blob/main/images/training.gif?raw=true)


## Use Cases

NN Inspector is designed for:

- **Students**: Learn how neural networks work in an interactive environment.
- **Educators**: Use as a teaching aid to demonstrate neural network concepts.
- **Researchers**: Experiment with different configurations and gain insights into underlying model behaviours.



## Getting Started

### Installation

Clone the repository:
```bash
git clone https://github.com/frasertheking/inspector.git
```

Navigate to the project directory, add your data and edit configs in script.js as necessary (below):
```bash
cd inspector
```

### Usage

Since we are loading JSON data, we will either need to use a public webserver, or you can start a local server and open the application in your web browser:

```python
# Using Python's HTTP server
python -m http.server

# Or using Node.js
npx http-server
```

Access the application at http://localhost:8000


## Configuration

Customize the application by modifying the user configuration options in the JavaScript file. Change your input predictors, add units and make sure the paths are pointing to the correct locations:

```javascript
let model_name = 'name-goes-here';
let input_vars = ["n0", "lambda", "Rho", "Fs", "Dm",
                  "Temperature", "Relative Humidity", "Pressure"];
let log_scaled_vars = ['n0', 'lambda', 'Rho', 'Dm'];
let response_vars = ['Rain', 'Snow', 'Mixed-Phase'];
let model_loc = 'models/name-goes-here/';
let test_filepath = 'data/data_name-goes-here.csv';
let hiddenNeuronOptions = [1, 2, 4, 8, 16];
```


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


## TensorFlow Callback

Since we aren't doing model training on the webserver, we precompute a variety of interesting model combinations and save them in the above JSON format to load on the fly. This helps save computational resources and makes things lighter on the web server. To do this, we've provided a simple callback below you can include in your TensorFlow fitting call that saves the relevant information. 

```python
class WeightBiasHistory(tf.keras.callbacks.Callback):
    def __init__(self, X_train):
        super().__init__()
        self.X_train = X_train

    def on_train_begin(self, logs=None):
        self.model.predict(self.X_train[:1])
        self.intermediate_layer_model = tf.keras.models.Model(
            inputs=self.model.layers[0].input,
            outputs=[layer.output for layer in self.model.layers if 'dense' in layer.name]
        )

    def on_epoch_end(self, epoch, logs=None):
        weights = [layer.get_weights()[0].tolist() for layer in
                  self.model.layers if layer.get_weights()]
        biases = [layer.get_weights()[1].tolist() for layer in
                  self.model.layers if layer.get_weights()]

        weights_history.append(weights)
        biases_history.append(biases)

        activations = self.intermediate_layer_model.predict(self.X_train_comb[:BATCH_SIZE])
        activations_history.append([
            [float(x) for x in activations[0][0]], 
            [float(x) for x in activations[1][0]]
        ])

        loss_history.append(logs.get('loss'))
        val_loss_history.append(logs.get('val_loss'))
        accuracy_history.append(logs.get('accuracy'))
        val_accuracy_history.append(logs.get('val_accuracy'))
```

```python
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[WeightBiasHistory(X_train)])
```

You'll then want to save this, along with the other variables in the JSON schema to a series of JSON files (one for each combination of inputs/hidden neurons:

```python
model_data = {
  'predictors': predictor_array,
  'units': units_array,
  'response_variable': response_array,
  'N_HIDDEN': N_HIDDEN,
  'weights_history': weights_history,
  'biases_history': biases_history,
  'activations_history': activations_history,
  'loss_history': loss_history,
  'val_loss_history': val_loss_history,
  'accuracy_history': accuracy_history,
  'val_accuracy_history': val_accuracy_history,
  'scaling_info': scaling_info, // from train_test_split
  'hyperparameters': {
      'OPTMIZER': OPTIMIZER,
      'LR': LR,
      'LOSS_FUNC': LOSS_FUNC,
      'ACTIVATION': ACTIVATION,
      'BATCH_SIZE': BATCH_SIZE,
      'L1_flag': L1_flag,
      'L1': L1
  }
}

with open(filepath, 'w') as f:
  json.dump(model_data, f)
```


## Roadmap

The application is currently under development with the following planned improvements:

- **Optional Features**: Finishing the ability to leave certain items out of the JSON schema.
- **UI Enhancements**: Improving the user interface for better usability.
- **Regression Models**: Allowing for regression models in addition to classification.
- **Model Switching**: Enhancing ability to switch between models seamlessly.
- **Error Dialogues**: Implementing better error handling and messaging.
- **Multiple Hidden Layers**: Supporting neural networks with multiple hidden layers.
- **Advanced Statistics**: Providing more detailed statistics and analytical tools.


## Inspiration

The project draws inspiration from:

- Daniel Smilkov and Shan Carter: For their work on interactive tools like TensorFlow Playground that make complex machine learning concepts accessible.
- Chris Olah and Colleagues at Anthropic: Especially their work on Toy Models of Superposition in interpretability research.


## Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes and push the branch.
- Submit a pull request for review.


## Contact
Fraser King, University of Michigan (kingfr@umich.edu)
