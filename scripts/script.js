/*
NN Inspector Application - JavaScript
Author: Fraser King
Date: November 2024
Description: This JavaScript file manages the interactivity and data visualization for the NN Inspector application.
Dependencies: D3.js v6

Configuration Options:
- Input variables for the models
- Toggle options for neuron visibility
- Model file paths and test data paths
*/


/*
USER CONFIGS
*/
/* MODEL 1 */
let model_name = '3phase'
let input_vars = ["n0", "lambda", "Rho", "Fs", "Dm", "Temperature", "Relative Humidity", "Pressure"];
let log_scaled_vars = ['n0', 'lambda', 'Rho', 'Dm']
let response_vars = ['Rain', 'Snow', 'Mixed-Phase'];
let model_loc = 'models/phase3/'
let test_filepath = 'data/test_data.csv'
let hidden_neuron_options = [1, 2, 4, 8, 16];

/* MODEL 2 */
// let model_name = '9phase'
// let input_vars = ['n0', 'Nt', 'lambda', 'Rho', 'Fs', 'Dm', 'Temperature', 'Relative Humidity', 'Pressure', 'Wind_Speed', 'Near Surface Reflectivity', 'Cloud Top Height']
// let log_scaled_vars = ['n0', 'Nt', 'lambda', 'Rho', 'Dm']
// let response_vars = ['Heavy Rain', 'Drizzle', 'Heavy R-M', 'Light R-M', 'Heavy Mixed', 'Heavy S-M', 'Light Mixed', 'Heavy Snow', 'Light Snow']
// let model_loc = 'models/phase9/'
// let test_filepath = 'data/test_data_umap.csv'
// let hidden_neuron_options = [1, 2, 4, 8, 16];



/*
GLOBALS
*/
let inputCheckboxes;
let svg;
let nodePositions = {};
let scalingInfo = {};
let nodeData = [];
let edgeSelection;
let nodeSelection;
let edgeLabelSelection;
let activationValueSelection;
let biasLabelSelection;
let weightsInputHidden, weightsHiddenOutput, biasesHidden, biasesOutput;
let weightColorScale;
let edgeRelevanceColorScale;
let testData = [];
let currentTestCaseType = 'predefined';
let lastScaledInputs = null;
let lastOriginalInputs = null;
let nodeActualValueSelection;
let nodeScaledValueSelection;
let testCase = null;
let inputVector = null;
let trainingInterval;
let currentNeuronIndex = 2;
let showWeights = false;
let lrpMode = false;
let isAnimating = false;
let isPaused = true;
let firstPlay = true;
let totalEpochs = 0;
let currentEpoch = null;
let weightsHistory = [];
let biasesHistory = [];
let activationsHistory = [];
let monosemanticIndices = [];
let inputNodes = [];
let hiddenNodes = [];
let outputNodes = [];
let layers = [];
let allWeights = [];
let allActivations = [];
let activationColorScale;
let testClickLineLocation;
let lossHistory = [];
let valLossHistory = [];
let accuracyHistory = [];
let valAccuracyHistory = [];
let lossSvg, lossXScale, lossYScale, lossLineTrain, lossLineVal, lossXAxis, lossYAxis;
let accuracySvg, accXScale, accYScale, accLineTrain, accLineVal, accXAxis, accYAxis;



/*
SETUP & EVENT HANDLERS
*/
loadModelJSON();
window.onload = generateCheckboxes;

const minusButton = document.getElementById('minus-button');
const plusButton = document.getElementById('plus-button');
const neuronCountDisplay = document.getElementById('neuron-count');
const epochText = document.getElementById("epoch-text");
const modeSelect = document.getElementById('mode-select');

modeSelect.addEventListener('change', handleModeChange);
minusButton.addEventListener('click', () => updateNeuronCount(-1));
plusButton.addEventListener('click', () => updateNeuronCount(1));

document.getElementById('show-weights-checkbox').addEventListener('change', function() {
    showWeights = this.checked;
    updateEdgeLabelsVisibility();
});
document.getElementById('custom-test-button').addEventListener('click', openCustomTestModal);
document.getElementById("toggle-inputs-button").addEventListener("click", () => {
    const inputOptions = document.getElementById("input-options");
    const button = document.getElementById("toggle-inputs-button");
    const currentDisplay = window.getComputedStyle(inputOptions).display;
    inputOptions.style.display = currentDisplay === "none" ? "inline-block" : "none";
    button.textContent = currentDisplay === "none" ? "Hide Inputs" : "Select Inputs";
});
document.getElementById('custom-test-cancel').addEventListener('click', closeCustomTestModal);
document.getElementById('custom-test-submit').addEventListener('click', submitCustomTestCase);
document.getElementById('model-info-button').addEventListener('click', function() {
    document.getElementById('model-info-modal').style.display = 'flex';
});
document.getElementById('close-model-info').addEventListener('click', function() {
    document.getElementById('model-info-modal').style.display = 'none';
});
document.getElementById('regularization-checkbox').addEventListener('change', function() {
    const lrpButton = document.getElementById('lrp-button');
    lrpButton.textContent = 'Activate LRP';
    lrpMode = false;
    loadModelJSON();
});
document.getElementById('lrp-button').addEventListener('click', toggleLRPMode);
handleModeChange();



/*
FUNCTIONS
*/
function handleButtonClick() {
    if (model_name == '9phase') {
        model_name = '3phase';
        input_vars = ["n0", "lambda", "Rho", "Fs", "Dm", "Temperature", "Relative Humidity", "Pressure"];
        log_scaled_vars = ['n0', 'lambda', 'Rho', 'Dm']
        response_vars = ['Rain', 'Snow', 'Mixed-Phase'];
        model_loc = 'models/phase3/'
        test_filepath = 'data/test_data.csv'
        hidden_neuron_options = [1, 2, 4, 8, 16];
    } else {
        model_name = '9phase';
        input_vars = ['n0', 'Nt', 'lambda', 'Rho', 'Fs', 'Dm', 'Temperature', 'Relative Humidity', 'Pressure', 'Wind_Speed', 'Near Surface Reflectivity', 'Cloud Top Height']
        log_scaled_vars = ['n0', 'Nt', 'lambda', 'Rho', 'Dm']
        response_vars = ['Heavy Rain', 'Drizzle', 'Heavy R-M', 'Light R-M', 'Heavy Mixed', 'Heavy S-M', 'Light Mixed', 'Heavy Snow', 'Light Snow']
        model_loc = 'models/phase9/'
        test_filepath = 'data/test_data_umap.csv'
        hidden_neuron_options = [1, 2, 4, 8, 16];
    }
    loadModelJSON();
    loadTestData();
}

function closeCustomTestModal() {
    document.getElementById('custom-test-modal').style.display = 'none';
}

function generateCheckboxes() {
    const form = document.getElementById('input-form');
    form.innerHTML = '';

    input_vars.forEach((inputName, index) => {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `input-${index}`;
        checkbox.style = 'margin-left: 15px;'
        checkbox.name = 'inputs';
        checkbox.value = inputName;
        checkbox.checked = true;

        const label = document.createElement('label');
        label.htmlFor = checkbox.id;
        label.textContent = inputName;

        form.appendChild(checkbox);
        form.appendChild(label);

    });


    inputCheckboxes = document.querySelectorAll('input[name="inputs"]');
    inputCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', handleInputSelectionChange);
    });
}

function handleInputSelectionChange(event) {
    const checked = Array.from(inputCheckboxes).filter(cb => cb.checked);
    if (checked.length === 0) {
        event.target.checked = true;
        alert("At least one input must be selected.");
    } else {
        input_vars = checked.map(cb => cb.value);
        currentEpoch = null;
        loadModelJSON();
    }
}

function updateNeuronCount(delta) {
    const newIndex = currentNeuronIndex + delta;
    if (newIndex >= 0 && newIndex < hidden_neuron_options.length) {
        currentNeuronIndex = newIndex;
        neuronCountDisplay.textContent = hidden_neuron_options[currentNeuronIndex];
        currentEpoch = null;
        loadModelJSON();
    }
}

function constructModelFilename() {
    const inputsInOrder = input_vars.filter(input => input_vars.includes(input));
    const isRegularizationEnabled = document.getElementById('regularization-checkbox').checked;
    const regularizationStatus = isRegularizationEnabled ? 'on' : 'off';
    const filename = model_loc + 'model_' + inputsInOrder.join('_') + '_hidden_' + hidden_neuron_options[currentNeuronIndex] + '_L1_' + regularizationStatus + '.json';
    return filename;
}

function loadModelJSON() {
    const modelFilename = constructModelFilename();
    d3.select('#network').selectAll('*').remove();
    d3.select('#loss-plot').selectAll('*').remove();
    d3.select('#accuracy-plot').selectAll('*').remove();

    d3.json(modelFilename)
        .then(function(data) {
            processModelData(data);

            if (data.hyperparameters) {
                displayHyperparameters(data.hyperparameters);
            } else {
                console.warn('No hyperparameters found in model data');
            }
        })
        .catch(function(error) {
            console.error('Error loading JSON:', error);
            alert('Error loading model JSON file. Please check your selections.');
        });
}

function openCustomTestModal() {
    const form = document.getElementById('custom-test-form');
    form.innerHTML = ''; 

    input_vars.forEach(inputName => {
        const container = document.createElement('div');
        container.classList.add('input-container');

        const label = document.createElement('label');
        label.textContent = inputName;
        label.style.display = 'inline-block';
        label.style.width = '150px';

        const input = document.createElement('input');
        input.type = 'text';
        input.name = inputName;
        input.required = true;
        input.style.width = '200px'; 

        container.appendChild(label);
        container.appendChild(input);

        form.appendChild(container);
    });
    document.getElementById('custom-test-modal').style.display = 'flex';
}

function submitCustomTestCase() {
    const form = document.getElementById('custom-test-form');
    const formData = new FormData(form);
    const customValues = {};

    let valid = true;
    let errorMessage = '';

    inputNodes.forEach(inputName => {
        const value = formData.get(inputName);
        if (value === null || value.trim() === '' || isNaN(value)) {
            errorMessage += `Please enter a valid numeric value for ${inputName}.\n`;
            valid = false;
            return;
        }
        customValues[inputName] = parseFloat(value);
    });

    if (!valid) {
        alert(errorMessage);
        return;
    }

    const scaledInputs = scaleInputs(customValues);

    if (scaledInputs === null) {
        return;
    }

    closeCustomTestModal();
    runCustomTestCase(scaledInputs, customValues);
}

function displayHyperparameters(hyperparameters) {
    const hyperparametersDiv = document.getElementById('hyperparameters');
    hyperparametersDiv.innerHTML = '';

    for (const key in hyperparameters) {
        const p = document.createElement('p');
        p.textContent = `${key}: ${hyperparameters[key]}`;
        hyperparametersDiv.appendChild(p);
    }
}

function handleModeChange() {
    const mode = modeSelect.value;
    const lossPlot = document.getElementById('loss-plot');
    const accuracyPlot = document.getElementById('accuracy-plot');
    if (mode === 'training') {
        document.getElementById('training-controls').style.display = 'block';
        document.getElementById('testing-controls').style.display = 'none';
        document.getElementById('activation-heatmap').style.display = 'none';
        document.getElementById('activation-sparsity').style.display = 'none';
        document.getElementById('confusion-matrix').style.display = 'none';
        document.getElementById('metrics').style.display = 'none';
        lossPlot.classList.add('shifted-down');
        accuracyPlot.classList.add('shifted-down');
    } else if (mode === 'testing') {
        document.getElementById('training-controls').style.display = 'none';
        document.getElementById('testing-controls').style.display = 'block';
        document.getElementById('activation-heatmap').style.display = 'block';
        document.getElementById('activation-sparsity').style.display = 'block';
        document.getElementById('confusion-matrix').style.display = 'block';
        document.getElementById('metrics').style.display = 'block';
        lossPlot.classList.remove('shifted-down');
        accuracyPlot.classList.remove('shifted-down');
    }

    if (testData.length === 0) {
        loadTestData();
    } else {
        computeConfusionMatrix();
    }

    if (mode == 'testing') {
        loadTestCaseIntoNN(0);
    }
}


function loadTestData() {
    return d3.csv(test_filepath).then(function(data) {
        data.forEach(function(d) {
            for (let key in d) {
                d[key] = +d[key];
            }
        });
        testData = data;
        computeConfusionMatrix();
    }).catch(function(error) {
        console.error('Error loading test data:', error);
        alert('Error loading test data. Please check that test data is available.');
    });
}

function computeConfusionMatrix() {
    const numClasses = response_vars.length;
    let confusionMatrix = Array.from({
            length: numClasses
        }, () =>
        Array(numClasses).fill(0)
    );

    const activationData = [];
    testData.forEach(function(testCase) {
        const result = predict(testCase);
        const predictedLabel = result.predictedLabel;
        const hiddenActivations = result.hiddenActivations;
        activationData.push(hiddenActivations);

        const trueLabel = testCase['phase'];
        confusionMatrix[trueLabel][predictedLabel] += 1;
    });

    displayConfusionMatrix(confusionMatrix);
    computeMetrics(confusionMatrix);
    computeAndDisplayActivationPlots(activationData);
}

function predict(testCase) {
    const inputVector = inputNodes.map(nodeName => {
        const inputNode = nodeData.find(n => n.name === nodeName);
        return inputNode.dead ? 0 : testCase[nodeName];
    });

    const finalWeights = weightsHistory[currentEpoch - 1];
    const finalBiases = biasesHistory[currentEpoch - 1];
    const weightsInputHidden = finalWeights[0];
    const biasesHidden = finalBiases[0];
    const weightsHiddenOutput = finalWeights[1];
    const biasesOutput = finalBiases[1];

    const hiddenActivations = [];
    for (let i = 0; i < hiddenNodes.length; i++) {
        const hiddenNode = nodeData.find(n => n.name === hiddenNodes[i]);
        if (hiddenNode.dead) {
            hiddenActivations[i] = 0;
            continue;
        }
        let z = biasesHidden[i];
        for (let j = 0; j < inputNodes.length; j++) {
            z += inputVector[j] * weightsInputHidden[j][i];
        }
        const activation = relu(z);
        hiddenActivations[i] = activation;
    }

    const outputZs = [];
    for (let i = 0; i < outputNodes.length; i++) {
        let z = biasesOutput[i];
        for (let j = 0; j < hiddenNodes.length; j++) {
            const hiddenNode = nodeData.find(n => n.name === hiddenNodes[j]);
            const hiddenActivation = hiddenNode.dead ? 0 : hiddenActivations[j];
            z += hiddenActivation * weightsHiddenOutput[j][i];
        }
        outputZs[i] = z;
    }

    const outputActivations = softmax(outputZs);
    const predictedLabel = outputActivations.indexOf(Math.max(...outputActivations));
    return {
        predictedLabel,
        hiddenActivations
    };
}

function displayConfusionMatrix(confusionMatrix) {
    let html = '';
    html += `<p><b>Confusion Matrix:</b></p>`;
    html += '<table>';

    const numClasses = confusionMatrix.length;
    html += '<tr><th>Actual \\ Predict</th>';
    for (let i = 0; i < numClasses; i++) {
        html += `<th title="${response_vars[i]}">${i}</th>`;
    }
    html += '</tr>';

    for (let i = 0; i < numClasses; i++) {
        html += `<tr><th title="${response_vars[i]}">${i}</th>`;
        for (let j = 0; j < confusionMatrix[i].length; j++) {
            html += `<td>${confusionMatrix[i][j]}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    document.getElementById('confusion-matrix').innerHTML = html;
}

function computeMetrics(confusionMatrix) {
    const numClasses = confusionMatrix.length;
    let total = 0,
        correct = 0;
    let rowSums = Array(numClasses).fill(0);
    let colSums = Array(numClasses).fill(0);

    for (let i = 0; i < numClasses; i++) {
        for (let j = 0; j < numClasses; j++) {
            total += confusionMatrix[i][j];
            rowSums[i] += confusionMatrix[i][j];
            colSums[j] += confusionMatrix[i][j];
            if (i === j) correct += confusionMatrix[i][j];
        }
    }

    let pod = [],
        far = [],
        csi = [];
    for (let i = 0; i < numClasses; i++) {
        const tp = confusionMatrix[i][i];
        const fn = rowSums[i] - tp;
        const fp = colSums[i] - tp;

        pod[i] = tp / (tp + fn) || 0;
        far[i] = fp / (fp + tp) || 0;
        csi[i] = tp / (tp + fp + fn) || 0;
    }

    const random = rowSums.reduce((acc, r, i) => acc + (r * colSums[i]) / total, 0);
    const hss = (correct - random) / (total - random);
    displayMetrics(pod, far, csi, hss);
}

function displayMetrics(pod, far, csi, hss) {
    let html = '';
    html += `<p><b>Metrics:</b> [HSS ${hss.toFixed(2)}]</p>`;
    html += '<table>';
    html += '<tr><th>Class</th><th>POD</th><th>FAR</th><th>CSI</th></tr>';

    const numClasses = pod.length;
    for (let i = 0; i < numClasses; i++) {
        html += `<tr>
                    <th title="${response_vars[i]}">${i}</th> <!-- Use index with tooltip -->
                    <td>${pod[i].toFixed(2)}</td>
                    <td>${far[i].toFixed(2)}</td>
                    <td>${csi[i].toFixed(2)}</td>
                 </tr>`;
    }

    html += '</table>';
    document.getElementById('metrics').innerHTML = html;
}

function toggleLRPMode() {
    lrpMode = !lrpMode;
    const lrpButton = document.getElementById('lrp-button');
    if (lrpMode) {
        lrpButton.textContent = 'Deactivate LRP';
        runLRP();
    } else {
        if (currentTestCaseType === 'custom') {
            runCustomTestCase(lastScaledInputs, lastOriginalInputs);
        } else {
            lrpButton.textContent = 'Activate LRP';
            if (testCase != null && inputVector != null) {
                runTestCaseWithInput(inputVector, testCase);
            } else {
                runTestCase();
            }
        }
    }
}

function identifyMonosemanticNeurons(weights, threshold = 0.8) {
    let monosemanticIndices = [];
    let dominanceRatios = [];

    let transposedWeights = [];
    transposedWeights = weights[0].map((_, colIndex) =>
        weights.map(row => Math.abs(row[colIndex]))
    );

    for (let j = 0; j < transposedWeights.length; j++) {
        const neuronWeights = transposedWeights[j];
        const maxWeight = Math.max(...neuronWeights);
        const sumWeights = neuronWeights.reduce((sum, weight) => sum + weight, 0);
        const dominanceRatio = maxWeight / sumWeights;
        dominanceRatios.push(dominanceRatio);

        if (dominanceRatio > threshold) {
            monosemanticIndices.push(j);
        }
    }

    return {
        monosemanticIndices,
        dominanceRatios
    };
}

function processModelData(data) {
    isAnimating = false;

    weightsHistory = data.weights_history;
    biasesHistory = data.biases_history;
    activationsHistory = data.activations_history;
    scalingInfo = data.scaling_info;
    totalEpochs = weightsHistory.length;

    lossHistory = data.loss_history;
    valLossHistory = data.val_loss_history;
    accuracyHistory = data.accuracy_history;
    valAccuracyHistory = data.val_accuracy_history;

    allWeights = [];
    allActivations = [];

    weightsHistory.forEach(weights => {
        weights.forEach(layerWeights => {
            allWeights.push(...layerWeights.flat());
        });
    });

    activationsHistory.forEach(activations => {
        activations.forEach(layerActivations => {
            allActivations.push(...layerActivations);
        });
    });

    if (currentEpoch == null) {
        currentEpoch = weightsHistory.length;
    }

    const lastEpochWeights = weightsHistory[currentEpoch - 1];
    const firstHiddenLayerWeights = lastEpochWeights[0];
    const {
        monosemanticIndices,
        dominanceRatios
    } = identifyMonosemanticNeurons(firstHiddenLayerWeights);

    const avgDominanceRatio = d3.mean(dominanceRatios);
    inputNodes = data.predictors;
    input_vars = [...inputNodes];

    const unitsMapping = {};
    inputNodes.forEach((nodeName, index) => {
        unitsMapping[nodeName] = data.units[index];
    });

    const N_HIDDEN = data.N_HIDDEN;
    hiddenNodes = [];
    for (let i = 0; i < N_HIDDEN; i++) {
        hiddenNodes.push(`H${i}`);
    }
    const numOutputs = biasesHistory[0][1].length;
    outputNodes = [];
    for (let i = 0; i < numOutputs; i++) {
        outputNodes.push(`O${i}`);
    }

    layers = [inputNodes, hiddenNodes, outputNodes];
    nodePositions = {};
    const layerSpacing = 300;
    const width = 1000;
    const margin = {
        top: 75,
        bottom: 100
    };
    const minNodeSpacing = 60;
    const maxNodesInLayer = layers.reduce((max, layer) => Math.max(max, layer.length), 0);
    const desiredHeight = margin.top + margin.bottom + (maxNodesInLayer - 1) * minNodeSpacing;
    const height = Math.max(850, desiredHeight);
    const availableHeight = height - margin.top - margin.bottom;

    let nodeSpacing;
    if (maxNodesInLayer > 1) {
        nodeSpacing = availableHeight / (maxNodesInLayer - 1);
    } else {
        nodeSpacing = availableHeight / 2;
    }

    svg = d3.select('#network')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    layers.forEach((layer, layerIndex) => {
        const offsetX = 140;
        const x = layerIndex * layerSpacing + offsetX;
        const layerSize = layer.length;

        const layerHeight = (layerSize - 1) * nodeSpacing;
        const yOffset = (availableHeight - layerHeight) / 2;
        const yStart = margin.top + yOffset;

        layer.forEach((node, nodeIndex) => {
            const y = yStart + nodeIndex * nodeSpacing;
            nodePositions[node] = {
                x,
                y
            };
        });
    });

    dominanceText = svg.append('text')
        .attr('id', 'avg-dominance-text')
        .attr('x', 100)
        .attr('y', height - 30)
        .attr('text-anchor', 'right')
        .attr('font-size', '16px')
        .text(`Avg. Dominance: ${avgDominanceRatio.toFixed(2)}`);

    weightColorScale = d3.scaleDiverging()
        .domain([d3.min(allWeights), 0, d3.max(allWeights)])
        .interpolator(function(t) {
            return d3.interpolateRdBu(1 - t);
        });

    activationColorScale = d3.scaleSequential(d3.interpolateReds)
        .domain([d3.min(allActivations), d3.max(allActivations)]);

    const edges = [];
    inputNodes.forEach((inputNode, i) => {
        hiddenNodes.forEach((hiddenNode, j) => {
            const weight = firstHiddenLayerWeights[i][j];
            edges.push({
                source: inputNode,
                target: hiddenNode,
                weight: weight,
                layer: 'input-hidden'
            });
        });
    });

    const secondHiddenLayerWeights = lastEpochWeights[1];
    hiddenNodes.forEach((hiddenNode, i) => {
        outputNodes.forEach((outputNode, j) => {
            const weight = secondHiddenLayerWeights[i][j];
            edges.push({
                source: hiddenNode,
                target: outputNode,
                weight: weight,
                layer: 'hidden-output'
            });
        });
    });

    const strokeWidthScale = d3.scaleLinear()
        .domain([0, d3.max(allWeights.map(Math.abs))])
        .range([2, 6]);

    const initialX = width / 2;
    const initialY = height / 2;

    edgeSelection = svg.selectAll('.edge')
        .data(edges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('x1', initialX)
        .attr('y1', initialY)
        .attr('x2', initialX)
        .attr('y2', initialY)
        .attr('stroke', d => weightColorScale(d.weight))
        .attr('stroke-width', d => (strokeWidthScale(Math.abs(d.weight))) + 1.0);

    edgeSelection.transition()
        .duration(500)
        .attr('x1', d => nodePositions[d.source].x)
        .attr('y1', d => nodePositions[d.source].y)
        .attr('x2', d => nodePositions[d.target].x)
        .attr('y2', d => nodePositions[d.target].y);

    edgeLabelSelection = svg.selectAll('.edge-label')
        .data(edges)
        .enter()
        .append('text')
        .attr('class', 'edge-label')
        .attr('x', initialX)
        .attr('y', initialY)
        .text(d => d.weight.toFixed(2))
        .attr('font-size', '12px')
        .attr('fill', 'black')
        .attr('font-weight', 'bold')
        .attr('stroke', 'white')
        .attr('stroke-width', 0.5)
        .style('display', showWeights ? 'block' : 'none');

    edgeLabelSelection.transition()
        .duration(500)
        .attr('x', d => {
            if (d.layer === 'input-hidden') {
                return nodePositions[d.source].x * 0.8 + nodePositions[d.target].x * 0.2;
            } else if (d.layer === 'hidden-output') {
                return nodePositions[d.source].x * 0.2 + nodePositions[d.target].x * 0.8;
            } else {
                return (nodePositions[d.source].x + nodePositions[d.target].x) / 2;
            }
        })
        .attr('y', d => {
            if (d.layer === 'input-hidden') {
                return nodePositions[d.source].y * 0.8 + nodePositions[d.target].y * 0.2;
            } else if (d.layer === 'hidden-output') {
                return nodePositions[d.source].y * 0.2 + nodePositions[d.target].y * 0.8;
            } else {
                return ((nodePositions[d.source].y + nodePositions[d.target].y) / 2);
            }
        });

    const nodes = [...inputNodes, ...hiddenNodes, ...outputNodes];
    nodeData = nodes.map(node => {
        return {
            name: node,
            activation: null,
            bias: null,
            monosemantic: hiddenNodes.includes(node) ? monosemanticIndices.includes(hiddenNodes.indexOf(node)) : false,
            dead: false,
            actualValue: null,
            scaledValue: null
        };
    });

    nodeSelection = svg.selectAll('.node')
        .data(nodeData)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', initialX)
        .attr('cy', initialY)
        .attr('r', 20)
        .attr('fill', 'white')
        .style("cursor", "pointer")
        .attr('stroke', d => {
            if (hiddenNodes.includes(d.name) && d.monosemantic) {
                return 'gold';
            } else {
                return 'black';
            }
        })
        .attr('stroke-width', d => {
            if (hiddenNodes.includes(d.name) && d.monosemantic) {
                return 4;
            } else {
                return 1;
            }
        });

    nodeSelection.on('click', function(event, d) {
        if (modeSelect.value === 'testing' && !inputNodes.includes(d.name) && !outputNodes.includes(d.name)) {
            d.dead = !d.dead;
            if (currentTestCaseType === 'custom') {
                runCustomTestCase(lastScaledInputs, lastOriginalInputs);
            } else {
                if (testCase != null && inputVector != null) {
                    runTestCaseWithInput(inputVector, testCase);
                } else {
                    runTestCase();
                }
            }
            computeConfusionMatrix();
        }
    });

    nodeSelection.transition()
        .duration(500)
        .attr('cx', d => nodePositions[d.name].x)
        .attr('cy', d => nodePositions[d.name].y);

    activationValueSelection = svg.selectAll('.activation-value')
        .data(nodeData)
        .enter()
        .filter(d => !inputNodes.includes(d.name))
        .append('text')
        .attr('class', 'activation-value')
        .attr('x', initialX)
        .attr('y', initialY)
        .text('')
        .attr('font-size', '12px')
        .attr('text-anchor', 'middle')
        .attr('fill', 'black')
        .style('pointer-events', 'none');

    activationValueSelection.transition()
        .duration(500)
        .attr('x', d => nodePositions[d.name].x)
        .attr('y', d => nodePositions[d.name].y + 5);

    const labelOffset = 30;

    // Node Labels (Variable Names)
    const nodeLabelSelection = svg.selectAll('.node-label')
        .data(nodeData)
        .enter()
        .filter(d => inputNodes.includes(d.name) || outputNodes.includes(d.name))
        .append('text')
        .attr('class', 'node-label')
        .attr('x', initialX)
        .attr('y', initialY)
        .text(d => {
            if (inputNodes.includes(d.name)) {
                return d.name;
            } else if (outputNodes.includes(d.name)) {
                const index = outputNodes.indexOf(d.name);
                return response_vars[index];
            }
        })
        .attr('font-size', '12px')
        .attr('text-anchor', d => {
            if (inputNodes.includes(d.name)) {
                return 'end';
            } else if (outputNodes.includes(d.name)) {
                return 'start';
            }
        })
        .attr('fill', 'black');

    nodeLabelSelection.transition()
        .duration(500)
        .attr('x', d => {
            if (inputNodes.includes(d.name)) {
                return nodePositions[d.name].x - labelOffset;
            } else if (outputNodes.includes(d.name)) {
                return nodePositions[d.name].x + labelOffset;
            }
        })
        .attr('y', d => {
            if (inputNodes.includes(d.name)) {
                return nodePositions[d.name].y - 8;
            } else if (outputNodes.includes(d.name)) {
                return nodePositions[d.name].y + 8;
            }
        });

    const nodeUnitSelection = svg.selectAll('.node-unit')
        .data(nodeData)
        .enter()
        .filter(d => inputNodes.includes(d.name))
        .append('text')
        .attr('class', 'node-unit')
        .attr('x', initialX)
        .attr('y', initialY)
        .text(d => '(' + unitsMapping[d.name] + ')')
        .attr('font-size', '12px')
        .attr('text-anchor', 'end')
        .attr('fill', 'black');

    nodeUnitSelection.transition()
        .duration(500)
        .attr('x', d => nodePositions[d.name].x - labelOffset)
        .attr('y', d => nodePositions[d.name].y + 5);

    nodeActualValueSelection = svg.selectAll('.node-actual-value')
        .data(nodeData)
        .enter()
        .filter(d => inputNodes.includes(d.name))
        .append('text')
        .attr('class', 'node-actual-value')
        .attr('x', initialX)
        .attr('y', initialY)
        .text('')
        .attr('font-size', '12px')
        .attr('text-anchor', 'end')
        .attr('fill', 'black');

    nodeActualValueSelection.transition()
        .duration(500)
        .attr('x', d => nodePositions[d.name].x - labelOffset)
        .attr('y', d => nodePositions[d.name].y + 20);

    biasLabelSelection = svg.selectAll('.bias-label')
        .data(nodeData)
        .enter()
        .filter(d => !inputNodes.includes(d.name))
        .append('text')
        .attr('class', 'bias-label')
        .attr('x', initialX)
        .attr('y', initialY)
        .text('')
        .attr('font-size', '12px')
        .attr('text-anchor', 'middle')
        .style('pointer-events', 'none');

    biasLabelSelection.transition()
        .duration(500)
        .attr('x', d => nodePositions[d.name].x)
        .attr('y', d => nodePositions[d.name].y + 35);

    layers.forEach((layer, index) => {
        const offsetX = 140;
        const x = index * layerSpacing + offsetX;
        const layerName = index === 0 ? 'Input Layer' : index === 1 ? 'Hidden Layer' : 'Output Layer';
        svg.append('text')
            .attr('x', x)
            .attr('y', 15)
            .text(layerName)
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .attr('text-anchor', 'middle');
    });

    const tooltip = d3.select('body').append('div')
        .attr('id', 'tooltip')
        .style('position', 'absolute')
        .style('opacity', 0)
        .style('pointer-events', 'none')
        .style('background-color', 'white')
        .style('border', '1px solid #ccc')
        .style('padding', '5px');

    edgeSelection.on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke', d3.color(d3.select(this).attr('stroke')).darker(1));
            tooltip.transition().duration(200).style('opacity', .9);
            let content = '';
            if (lrpMode && d.relevance !== undefined) {
                content = 'Relevance: ' + d.relevance.toFixed(2);
            } else if (d.weight !== undefined) {
                content = 'Weight: ' + d.weight.toFixed(2);
            }
            tooltip.html(content)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function(event, d) {
            d3.select(this)
                .attr('stroke', d => {
                    if (lrpMode && d.relevance !== undefined) {
                        return edgeRelevanceColorScale(Math.abs(d.relevance));
                    } else {
                        return weightColorScale(d.weight);
                    }
                });
            tooltip.transition().duration(500).style('opacity', 0);
        });

    nodeSelection.on('mouseover', function(event, d) {
            d3.select(this).attr('stroke-width', 3);
            tooltip.transition().duration(200).style('opacity', .9);
            let content = '';
            if (inputNodes.includes(d.name)) {
                content = 'Input Node: ' + d.name;
                if (d.activation !== null && d.activation !== undefined) {
                    content += '<br/>Value: ' + d.activation.toFixed(2);
                }
            } else if (hiddenNodes.includes(d.name) || outputNodes.includes(d.name)) {
                content = 'Node: ' + d.name;
                if (d.activation !== null && d.activation !== undefined) {
                    content += '<br/>Activation: ' + d.activation.toFixed(2);
                }
                if (d.bias !== null && d.bias !== undefined) {
                    content += '<br/>Bias: ' + d.bias.toFixed(2);
                }
            }
            tooltip.html(content)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function(event, d) {
            d3.select(this).attr('stroke-width', d => {
                if (hiddenNodes.includes(d.name) && d.monosemantic) {
                    return 4;
                } else {
                    return 1;
                }
            });
            tooltip.transition().duration(500).style('opacity', 0);
        });

    isAnimating = false;
    document.getElementById('play-pause-button').addEventListener('click', () => playPauseTraining(svg));
    document.getElementById('step-button').addEventListener('click', () => trainOneStep(svg));
    document.getElementById('restart-button').addEventListener('click', () => restartTrainingVisualization(svg));

    createLossPlot();
    createAccuracyPlot();
    updateVisualization(totalEpochs - 1);
    updatePlots(totalEpochs - 1);

    if (testData.length === 0) {
        loadTestData();
    } else {
        computeConfusionMatrix();
    }
}


function createActivationHeatmap(activationData) {
    const margin = {
        top: 40,
        right: 20,
        bottom: 40,
        left: 50
    };
    const width = 300 - margin.left - margin.right; 
    const height = 300 - margin.top - margin.bottom;
    const n = activationData[0].length;

    const nodeNames = Array.from({
        length: n
    }, (_, i) => `H${i}`);

    const svg = d3.select('#activation-heatmap').append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .style('background-color', '#f9f9f9') 
        .style('display', 'block') 
        .style('margin', '0 auto')
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .attr("font-size", "16px")
        .attr("font-weight", "bold")
        .text("Activation Correlations");

    const data = computeNeuronCorrelation(activationData);

    const xScale = d3.scaleBand()
        .range([0, width])
        .domain(d3.range(n))
        .padding(0.05);

    const yScale = d3.scaleBand()
        .range([0, height])
        .domain(d3.range(n))
        .padding(0.05);

    const colorScale = d3.scaleDiverging()
        .domain([-1, 0, 1])
        .interpolator(function(t) {
            return d3.interpolateRdBu(1 - t);
        });

    const tooltip = d3.select("body")
        .append("div")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "rgba(0, 0, 0, 0.7)")
        .style("color", "white")
        .style("padding", "5px")
        .style("border-radius", "5px")
        .style("font-size", "12px");

    svg.selectAll("rect")
        .data(data.flat())
        .enter()
        .append("rect")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y))
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", d => colorScale(d.value))
        .attr("stroke", "none")
        .on("mouseover", function(event, d) {
            d3.select(this)
                .attr("stroke", "black")
                .attr("stroke-width", 2);
            tooltip.style("visibility", "visible")
                .text(`r [\n${nodeNames[d.x]}, ${nodeNames[d.y]}]: ${d.value.toFixed(2)}`);
        })
        .on("mousemove", function(event) {
            tooltip.style("top", (event.pageY - 10) + "px")
                .style("left", (event.pageX + 10) + "px");
        })
        .on("mouseout", function() {
            d3.select(this)
                .attr("stroke", "none");

            tooltip.style("visibility", "hidden");
        });

    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale).tickFormat(i => nodeNames[i]))
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-0.5em")
        .attr("dy", "0.25em")
        .attr("transform", "rotate(-45)");

    svg.append("g")
        .call(d3.axisLeft(yScale).tickFormat(i => nodeNames[i]));
}

function computeNeuronCorrelation(activations) {
    const n = activations[0].length; 
    const correlationMatrix = [];

    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            const x = activations.map(a => a[i]);
            const y = activations.map(a => a[j]);
            const corr = computeCorrelation(x, y);
            row.push({
                x: j,
                y: i,
                value: corr
            });
        }
        correlationMatrix.push(row);
    }
    return correlationMatrix;
}

function computeCorrelation(x, y) {
    const n = x.length;
    const meanX = d3.mean(x);
    const meanY = d3.mean(y);
    const cov = d3.sum(x.map((xi, i) => (xi - meanX) * (y[i] - meanY))) / n;
    const stdX = Math.sqrt(d3.sum(x.map(xi => (xi - meanX) ** 2)) / n);
    const stdY = Math.sqrt(d3.sum(y.map(yi => (yi - meanY) ** 2)) / n);
    const corr = cov / (stdX * stdY);
    return corr;
}

async function loadTestCaseIntoNN(index) {
    if (testData.length === 0) {
        await loadTestData();
    }

    if (index >= 0 && index < testData.length) {
        testCase = testData[index];
        inputVector = inputNodes.map(nodeName => testCase[nodeName]);
        currentTestCaseType = 'predefined';

        runTestCaseWithInput(inputVector, testCase);
    } else {
        console.warn("Invalid test case index selected.");
    }
}

function createActivationSparsityPlot(activationData) {
    const margin = {
        top: 50,
        right: 50,
        bottom: 50,
        left: 50
    };
    const width = 400;
    const height = 200;
    const epsilon = 0.01;

    const svg = d3.select("#activation-sparsity")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .style("background-color", "#f9f9f9")
        .style('display', 'block')
        .style('margin', '0 auto')
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    svg.append("text")
        .attr("x", width / 2)
        .attr("y", -30)
        .attr("text-anchor", "middle")
        .attr("font-size", "16px")
        .attr("font-weight", "bold")
        .text("Test Sparsity");

    const numNeurons = activationData[0].length;
    const numSamples = activationData.length;

    const plotData = [];
    activationData.forEach((sample, sampleIndex) => {
        sample.forEach((activation, neuronIndex) => {
            if (activation > epsilon) {
                plotData.push({
                    sampleIndex: sampleIndex,
                    neuronIndex: neuronIndex,
                    activation: activation,
                    class: testData[sampleIndex].phase
                });
            }
        });
    });

    const xScale = d3.scaleLinear()
        .domain([0, numSamples - 1])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([0, numNeurons - 1])
        .range([height, 0]);

    const classColors = d3.scaleOrdinal()
        .domain(d3.range(response_vars.length))
        .range(d3.schemeCategory10.slice(0, response_vars.length));
    
    svg.selectAll("text")
        .data(plotData)
        .enter()
        .append("text")
        .attr("x", d => xScale(d.sampleIndex))
        .attr("y", d => yScale(d.neuronIndex))
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .attr("font-size", "12px") // Adjust the font size as needed
        .attr("fill", d => classColors(d.class))
        .text('|');
    

    svg.append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .append("text")
        .attr("x", width / 2)
        .attr("y", 40)
        .attr("fill", "black")
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .text("Test Cases");

    svg.append("g")
        .call(d3.axisLeft(yScale).ticks(numNeurons))
        .append("text")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("transform", "rotate(-90)")
        .attr("fill", "black")
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .text("Neuron Index");

    const verticalLine = svg.append("line")
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("y1", 0)
        .attr("y2", height)
        .style("visibility", "hidden");

        // Dashed vertical line (if needed)
    const verticalLine2 = svg.append("line")
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "gray")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,4") // Dashed line style
        .attr("x1", testClickLineLocation)
        .attr("x2", testClickLineLocation);

    const tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "rgba(0, 0, 0, 0.7)")
        .style("color", "white")
        .style("padding", "8px")
        .style("border-radius", "5px")
        .style("font-size", "12px")
        .style("pointer-events", "none");

    svg.append("rect")
        .attr("width", width)
        .attr("height", height)
        .style("fill", "none")
        .style("pointer-events", "all")
        .style("cursor", "pointer")
        .on("mousemove", function(event) {
            const [mouseX] = d3.pointer(event, this);
            const sampleIndex = Math.round(xScale.invert(mouseX));
            const boundedSampleIndex = Math.max(0, Math.min(numSamples - 1, sampleIndex));

            verticalLine
                .attr("x1", xScale(boundedSampleIndex))
                .attr("x2", xScale(boundedSampleIndex))
                .style("visibility", "visible");

            const activations = activationData[boundedSampleIndex];
            const currentClass = testData[boundedSampleIndex].phase;
            const className = response_vars[currentClass] || `Class ${currentClass}`;

            const tooltipContent = activations
                .map((activation, i) => `H${i}: ${activation.toFixed(2)}`)
                .join("<br>");

            tooltip.html(`Test Case ${boundedSampleIndex}<br>Class: ${className}<br>${tooltipContent}`)
                .style("visibility", "visible")
                .style("top", (event.pageY + 15) + "px")
                .style("left", (event.pageX + 15) + "px");
        })
        .on("mouseout", function() {
            verticalLine.style("visibility", "hidden");
            tooltip.style("visibility", "hidden");
        })
        .on("click", function(event) {
            const [mouseX] = d3.pointer(event, this);
            const sampleIndex = Math.round(xScale.invert(mouseX));
            const boundedSampleIndex = Math.max(0, Math.min(numSamples - 1, sampleIndex));

            testClickLineLocation = xScale(boundedSampleIndex);

            loadTestCaseIntoNN(boundedSampleIndex);
            tooltip.style("visibility", "hidden");
        });
}

function computeAndDisplayActivationPlots(activationData) {
    if (modeSelect.value !== 'testing') return;

    d3.select('#activation-heatmap').selectAll('*').remove();
    d3.select('#activation-sparsity').selectAll('*').remove();

    createActivationHeatmap(activationData);
    createActivationSparsityPlot(activationData);
}

function updateEdgeLabelsVisibility() {
    if (lrpMode) {
        edgeLabelSelection.style('display', 'block');
    } else {
        edgeLabelSelection.style('display', showWeights ? 'block' : 'none');
    }
}

function playPauseTraining(svg) {
    if (firstPlay) {
        firstPlay = false;
        currentEpoch = 1;
    }
    if (isPaused) {
        isPaused = false;
        isAnimating = true;
        document.getElementById("play-pause-button").textContent = "Pause";
        
        if (!d3.select("#animated-edges-style").node()) {
            svg.append("defs").append("style")
                .attr("id", "animated-edges-style")
                .text(`
                    .edge {
                        stroke-dasharray: 5,5; /* 5px dash, 5px gap */
                        animation: dash 2s linear infinite;
                    }
            
                    @keyframes dash {
                        from { stroke-dashoffset: 0; }
                        to { stroke-dashoffset: -10; }
                    }
                `);
        }

        trainingInterval = setInterval(() => {
            if (isPaused) return;
            updateVisualization(currentEpoch);
            currentEpoch++;

            if (currentEpoch >= totalEpochs) {
                clearInterval(trainingInterval);
                isAnimating = false;
                isPaused = true;
                firstPlay = true;
                document.getElementById("play-pause-button").textContent = "Play";
                d3.select("#animated-edges-style").remove();
            }
        }, 100);

    } else {
        isPaused = true;
        isAnimating = false;
        document.getElementById("play-pause-button").textContent = "Play";
        clearInterval(trainingInterval);
        d3.select("#animated-edges-style").remove();
    }
}

function trainOneStep(svg) {
    if (isAnimating) return; 

    if (currentEpoch >= totalEpochs) {
        firstPlay = true;
        d3.select("#animated-edges-style").remove();
        return
    }

    isPaused = true;
    isAnimating = false;
    document.getElementById("play-pause-button").textContent = "Play";

    updateVisualization(currentEpoch);
    currentEpoch++;

}

function restartTrainingVisualization(svg) {
    isAnimating = false;
    isPaused = true;
    currentEpoch = 1;
    document.getElementById("play-pause-button").textContent = "Play";
    d3.select("#animated-edges-style").remove();
    clearInterval(trainingInterval);
    playPauseTraining(svg);
}

function updateVisualization(epochIndex) {
    epochText.innerHTML = `<strong>Epoch</strong>: ${epochIndex + 1} / ${totalEpochs}`;
    nodeActualValueSelection.text('');

    const weights = weightsHistory[epochIndex];
    const biases = biasesHistory[epochIndex];
    const activations = activationsHistory[epochIndex];
    const weightsInputHidden = weights[0];
    const biasesHidden = biases[0];
    const weightsHiddenOutput = weights[1];
    const biasesOutput = biases[1];
    const activationHiddenFirstSample = activations[0];
    const activationOutputFirstSample = activations[1];
    const edgesData = [];

    edgeSelection.each(function(d) {
        if (inputNodes.includes(d.source) && hiddenNodes.includes(d.target)) {
            const i = inputNodes.indexOf(d.source);
            const j = hiddenNodes.indexOf(d.target);
            d.weight = weightsInputHidden[i][j];
        } else if (hiddenNodes.includes(d.source) && outputNodes.includes(d.target)) {
            const i = hiddenNodes.indexOf(d.source);
            const j = outputNodes.indexOf(d.target);
            d.weight = weightsHiddenOutput[i][j];
        }
        edgesData.push(d);
    });

    nodeData.forEach(node => {
        if (hiddenNodes.includes(node.name)) {
            const index = hiddenNodes.indexOf(node.name);
            node.activation = activationHiddenFirstSample[index];
            node.bias = biasesHidden[index];
        } else if (outputNodes.includes(node.name)) {
            const index = outputNodes.indexOf(node.name);
            node.activation = activationOutputFirstSample[index];
            node.bias = biasesOutput[index];
        } else if (inputNodes.includes(node.name)) {
            node.activation = null;
            node.bias = null;
        }
    });

    edgeSelection
        .attr('stroke', d => weightColorScale(d.weight));

    edgeLabelSelection
        .text(d => (d.weight !== null && d.weight !== undefined) ? d.weight.toFixed(2) : '');

    updateEdgeLabelsVisibility();

    nodeSelection
        .attr('fill', d => {
            if (inputNodes.includes(d.name)) {
                return 'white';
            } else {
                return activationColorScale(d.activation);
            }
        });

    activationValueSelection
        .text(d => (d.activation !== null && d.activation !== undefined) ? d.activation.toFixed(2) : '')
        .attr('fill', d => {
            if (inputNodes.includes(d.name)) {
                return 'black';
            } else {
                const bgColor = d3.color(activationColorScale(d.activation));
                const luminance = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
                return luminance > 186 ? 'black' : 'white';
            }
        });

    biasLabelSelection
        .text(d => (d.bias !== null && d.bias !== undefined) ? `${d.bias.toFixed(2)}` : '');

    updatePlots(epochIndex);
}

function runTestCaseCombined({ inputVector = null, testCase = null, scaledInputs = null, originalInputs = null, currentTestCaseType = null }) {
    if (isAnimating) return;
    if (currentEpoch == null) {
        currentEpoch = weightsHistory.length;
    }

    if (currentTestCaseType == null) {
        currentTestCaseType = 'predefined';
    }

    const finalWeights = weightsHistory[currentEpoch - 1];
    const finalBiases = biasesHistory[currentEpoch - 1];

    weightsInputHidden = finalWeights[0];
    biasesHidden = finalBiases[0];
    weightsHiddenOutput = finalWeights[1];
    biasesOutput = finalBiases[1];

    if (currentTestCaseType === 'custom') {
        lastScaledInputs = scaledInputs;
        lastOriginalInputs = originalInputs;
    }

    let realValues = null;

    if (testCase != null) {
        realValues = reverseScaleInputs(testCase);
    }

    if (inputVector == null) {
        if (testCase != null) {
            inputVector = inputNodes.map(nodeName => testCase[nodeName]);
        } else if (scaledInputs != null) {
            inputVector = inputNodes.map(nodeName => scaledInputs[nodeName]);
        } else {
            const selectedIndex = 0;
            testCase = testData[selectedIndex];
            inputVector = inputNodes.map(nodeName => testCase[nodeName]);
            realValues = reverseScaleInputs(testCase);
        }
    }

    const hiddenActivations = [];
    for (let i = 0; i < hiddenNodes.length; i++) {
        const node = nodeData.find(n => n.name === hiddenNodes[i]);
        if (node.dead) {
            hiddenActivations[i] = 0;
            continue;
        }
        let z = biasesHidden[i];
        for (let j = 0; j < inputNodes.length; j++) {
            const inputNode = nodeData.find(n => n.name === inputNodes[j]);
            const inputActivation = inputNode.dead ? 0 : inputVector[j];
            z += inputActivation * weightsInputHidden[j][i];
        }
        const activation = relu(z);
        hiddenActivations[i] = activation;
    }

    const outputZs = [];
    for (let i = 0; i < outputNodes.length; i++) {
        let z = biasesOutput[i];
        for (let j = 0; j < hiddenNodes.length; j++) {
            const hiddenNode = nodeData.find(n => n.name === hiddenNodes[j]);
            const hiddenActivation = hiddenNode.dead ? 0 : hiddenActivations[j];
            z += hiddenActivation * weightsHiddenOutput[j][i];
        }
        outputZs[i] = z;
    }

    const outputActivations = softmax(outputZs);

    nodeData.forEach(node => {
        if (inputNodes.includes(node.name)) {
            const dead = node.dead;
            if (currentTestCaseType === 'custom') {
                node.activation = dead ? 0 : scaledInputs[node.name] || 0;
                node.actualValue = originalInputs[node.name];
                node.scaledValue = scaledInputs[node.name];
            } else {
                node.activation = dead ? 0 : testCase[node.name];
                node.actualValue = realValues != null ? realValues[node.name] : null;
                node.scaledValue = testCase[node.name];
            }
            node.bias = null;
        } else if (hiddenNodes.includes(node.name)) {
            const index = hiddenNodes.indexOf(node.name);
            node.activation = node.dead ? 0 : hiddenActivations[index];
            node.bias = biasesHidden[index];
        } else if (outputNodes.includes(node.name)) {
            const index = outputNodes.indexOf(node.name);
            node.activation = outputActivations[index];
            node.bias = biasesOutput[index];
        } else {
            node.activation = 0;
            node.bias = null;
        }

        if (isNaN(node.activation) || node.activation == null) {
            node.activation = 0;
        }
    });

    nodeActualValueSelection.text(d => {
        if (d.actualValue != null && d.actualValue != undefined) {
            if (currentTestCaseType === 'custom') {
                return 'Actual: ' + d.actualValue;
            } else {
                return '(' + d.actualValue + ')';
            }
        } else {
            return '';
        }
    });

    const validActivations = nodeData
        .filter(d => !inputNodes.includes(d.name) && !isNaN(d.activation) && d.activation != null)
        .map(d => d.activation);

    const maxActivation = d3.max(validActivations) || 1;
    const activationColorScaleTest = d3.scaleSequential(d3.interpolateReds)
        .domain([0, maxActivation]);

    nodeSelection
        .attr('fill', d => {
            if (d.dead) {
                return 'black';
            } else if (inputNodes.includes(d.name)) {
                return 'white';
            } else {
                if (isNaN(d.activation) || d.activation == null) {
                    return 'grey';
                } else {
                    return activationColorScaleTest(d.activation);
                }
            }
        });

    activationValueSelection
        .text(d => (d.activation !== null && d.activation !== undefined && !isNaN(d.activation)) ? d.activation.toFixed(2) : '')
        .attr('fill', d => {
            if (d.dead) {
                return 'white';
            } else if (inputNodes.includes(d.name)) {
                return 'black';
            } else {
                let colorStr = activationColorScaleTest(d.activation);
                if (!colorStr) {
                    return 'black';
                }
                const bgColor = d3.color(colorStr);
                if (!bgColor) {
                    return 'black';
                }
                const luminance = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
                return luminance > 186 ? 'black' : 'white';
            }
        });

    biasLabelSelection
        .text(d => (d.bias !== null && d.bias !== undefined) ? `${d.bias.toFixed(2)}` : '');

    updateEdgeLabelsVisibility();

    if (lrpMode) {
        runLRP();
    } else {
        edgeSelection.each(function(d) {
            if (inputNodes.includes(d.source) && hiddenNodes.includes(d.target)) {
                const i = inputNodes.indexOf(d.source);
                const j = hiddenNodes.indexOf(d.target);
                d.weight = weightsInputHidden[i][j];
            } else if (hiddenNodes.includes(d.source) && outputNodes.includes(d.target)) {
                const i = hiddenNodes.indexOf(d.source);
                const j = outputNodes.indexOf(d.target);
                d.weight = weightsHiddenOutput[i][j];
            }
        });

        edgeSelection
            .attr('stroke', d => weightColorScale(d.weight));

        edgeLabelSelection
            .text(d => (d.weight !== null && d.weight !== undefined) ? d.weight.toFixed(2) : '')
            .style('display', showWeights ? 'block' : 'none')
            .attr('fill', 'black')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold');

        updateEdgeLabelsVisibility();
    }

    computeConfusionMatrix();
}

function runTestCaseWithInput(inputVector, testCase) {
    runTestCaseCombined({
        inputVector: inputVector,
        testCase: testCase,
        currentTestCaseType: 'predefined'
    });
}

function runTestCase() {
    const selectedIndex = 0;
    const testCase = testData[selectedIndex];
    const inputVector = inputNodes.map(nodeName => testCase[nodeName]);
    runTestCaseCombined({
        inputVector: inputVector,
        testCase: testCase,
        currentTestCaseType: 'predefined'
    });
}

function runCustomTestCase(scaledInputs, originalInputs) {
    runTestCaseCombined({
        scaledInputs: scaledInputs,
        originalInputs: originalInputs,
        currentTestCaseType: 'custom'
    });
}

function relu(x) {
    return Math.max(0, x);
}

function softmax(zs) {
    const maxZ = Math.max(...zs);
    const exps = zs.map(z => Math.exp(z - maxZ));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sumExps);
}

function createLossPlot() {
    const margin = {
        top: 20,
        right: 20,
        bottom: 40,
        left: 50
    };
    const width = 450 - margin.left - margin.right; 
    let height = 300 - margin.top - margin.bottom;


    lossSvg = d3.select('#loss-plot').append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .style('background-color', '#f9f9f9')
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    lossXScale = d3.scaleLinear()
        .domain([1, totalEpochs])
        .range([0, width]);

    const allLosses = lossHistory.concat(valLossHistory);
    const maxLoss = d3.max(allLosses);
    const minLoss = d3.min(allLosses);
    lossYScale = d3.scaleLinear()
        .domain([minLoss, maxLoss])
        .range([height, 0]);

    lossLineTrain = d3.line()
        .x((d, i) => lossXScale(i + 1))
        .y(d => lossYScale(d));

    lossLineVal = d3.line()
        .x((d, i) => lossXScale(i + 1))
        .y(d => lossYScale(d));

    lossXAxis = lossSvg.append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .call(d3.axisBottom(lossXScale).ticks(totalEpochs).tickFormat(d3.format("d")));

    lossYAxis = lossSvg.append('g')
        .call(d3.axisLeft(lossYScale));

    lossSvg.append('path')
        .datum(lossHistory)
        .attr('class', 'line-train')
        .attr('fill', 'none')
        .attr('stroke', 'black')
        .attr('stroke-width', 3)
        .attr('d', lossLineTrain);

    lossSvg.append('path')
        .datum(valLossHistory)
        .attr('class', 'line-val')
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 3)
        .attr('d', lossLineVal);

    lossSvg.append('text')
        .attr('x', width / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text('Loss');

    lossSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Epoch');

    lossSvg.append('text')
        .attr('x', -height / 2)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Loss');

    const legend = lossSvg.append('g')
        .attr('transform', 'translate(' + (width - 120) + ',' + 10 + ')');

    legend.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', 'black');

    legend.append('text')
        .attr('x', 15)
        .attr('y', 10)
        .text('Training Loss');

    legend.append('rect')
        .attr('x', 0)
        .attr('y', 15)
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', 'red');

    legend.append('text')
        .attr('x', 15)
        .attr('y', 25)
        .text('Validation Loss');

    const verticalLine = lossSvg.append('line')
        .attr('stroke', 'gray')
        .attr('stroke-width', 1)
        .attr('y1', 0)
        .attr('y2', height)
        .style('visibility', 'hidden');

    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(255, 255, 255, 0.8)')
        .style('border', '1px solid #ccc')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('pointer-events', 'none')
        .style('visibility', 'hidden')
        .style('font-size', '12px');

    lossSvg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .style('fill', 'none')
        .style('pointer-events', 'all')
        .on('mousemove', function(event) {
            const [mouseX] = d3.pointer(event, this);
            let epoch = Math.round(lossXScale.invert(mouseX));
            epoch = Math.max(1, Math.min(totalEpochs, epoch));

            verticalLine
                .attr('x1', lossXScale(epoch))
                .attr('x2', lossXScale(epoch))
                .style('visibility', 'visible');

            const trainLoss = lossHistory[epoch - 1];
            const valLoss = valLossHistory[epoch - 1];

            tooltip.html(
                    `<strong>Epoch ${epoch}</strong><br/>
            Training Loss: ${trainLoss.toFixed(4)}<br/>
            Validation Loss: ${valLoss.toFixed(4)}`
                )
                .style('visibility', 'visible')
                .style('top', (event.pageY + 15) + 'px')
                .style('left', (event.pageX + 15) + 'px');
        })
        .on('mouseout', function() {
            verticalLine.style('visibility', 'hidden');
            tooltip.style('visibility', 'hidden');
        });
}

function createAccuracyPlot() {
    const margin = {
        top: 20,
        right: 20,
        bottom: 40,
        left: 50
    };
    const width = 450 - margin.left - margin.right; 
    let height = 300 - margin.top - margin.bottom;

    accuracySvg = d3.select('#accuracy-plot').append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .style('background-color', '#f9f9f9')
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    accXScale = d3.scaleLinear()
        .domain([1, totalEpochs])
        .range([0, width]);

    const allAccuracies = accuracyHistory.concat(valAccuracyHistory);
    const maxAcc = d3.max(allAccuracies);
    const minAcc = d3.min(allAccuracies);
    accYScale = d3.scaleLinear()
        .domain([minAcc, maxAcc])
        .range([height, 0]);

    accLineTrain = d3.line()
        .x((d, i) => accXScale(i + 1))
        .y(d => accYScale(d));

    accLineVal = d3.line()
        .x((d, i) => accXScale(i + 1))
        .y(d => accYScale(d));

    accXAxis = accuracySvg.append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .call(d3.axisBottom(accXScale).ticks(totalEpochs).tickFormat(d3.format("d")));

    accYAxis = accuracySvg.append('g')
        .call(d3.axisLeft(accYScale));

    accuracySvg.append('path')
        .datum(accuracyHistory)
        .attr('class', 'line-train')
        .attr('fill', 'none')
        .attr('stroke', 'black')
        .attr('stroke-width', 3)
        .attr('d', accLineTrain);

    accuracySvg.append('path')
        .datum(valAccuracyHistory)
        .attr('class', 'line-val')
        .attr('fill', 'none')
        .attr('stroke', 'red')
        .attr('stroke-width', 3)
        .attr('d', accLineVal);

    accuracySvg.append('text')
        .attr('x', width / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text('Accuracy');

    accuracySvg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Epoch');

    accuracySvg.append('text')
        .attr('x', -height / 2)
        .attr('y', -35)
        .attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Accuracy');

    const legend = accuracySvg.append('g')
        .attr('transform', 'translate(' + (width - 150) + ',' + (height - 40) + ')');

    legend.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', 'black');

    legend.append('text')
        .attr('x', 15)
        .attr('y', 10)
        .text('Training Accuracy');

    legend.append('rect')
        .attr('x', 0)
        .attr('y', 15)
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', 'red');

    legend.append('text')
        .attr('x', 15)
        .attr('y', 25)
        .text('Validation Accuracy');

    const verticalLine = accuracySvg.append('line')
        .attr('stroke', 'gray')
        .attr('stroke-width', 1)
        .attr('y1', 0)
        .attr('y2', height)
        .style('visibility', 'hidden');

    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(255, 255, 255, 0.9)')
        .style('border', '1px solid #ccc')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('pointer-events', 'none')
        .style('visibility', 'hidden')
        .style('font-size', '12px')
        .style('box-shadow', '0px 0px 6px rgba(0, 0, 0, 0.1)');

    accuracySvg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .style('fill', 'none')
        .style('pointer-events', 'all')
        .on('mousemove', function(event) {
            const [mouseX] = d3.pointer(event, this);
            let epoch = Math.round(accXScale.invert(mouseX));
            epoch = Math.max(1, Math.min(totalEpochs, epoch));

            verticalLine
                .attr('x1', accXScale(epoch))
                .attr('x2', accXScale(epoch))
                .style('visibility', 'visible');

            const trainAcc = accuracyHistory[epoch - 1];
            const valAcc = valAccuracyHistory[epoch - 1];
            tooltip.html(
                    `<strong>Epoch ${epoch}</strong><br/>
            Training Accuracy: ${(trainAcc * 100).toFixed(2)}%<br/>
            Validation Accuracy: ${(valAcc * 100).toFixed(2)}%`
                )
                .style('visibility', 'visible')
                .style('top', (event.pageY + 15) + 'px')
                .style('left', (event.pageX + 15) + 'px');
        })
        .on('mouseout', function() {
            verticalLine.style('visibility', 'hidden');
            tooltip.style('visibility', 'hidden');
        });
}

function updatePlots(epochIndex) {
    const currentLossTrain = lossHistory.slice(0, epochIndex + 1);
    const currentLossVal = valLossHistory.slice(0, epochIndex + 1);

    lossSvg.select('.line-train')
        .datum(currentLossTrain)
        .attr('d', lossLineTrain);

    lossSvg.select('.line-val')
        .datum(currentLossVal)
        .attr('d', lossLineVal);

    const currentAccTrain = accuracyHistory.slice(0, epochIndex + 1);
    const currentAccVal = valAccuracyHistory.slice(0, epochIndex + 1);

    accuracySvg.select('.line-train')
        .datum(currentAccTrain)
        .attr('d', accLineTrain);

    accuracySvg.select('.line-val')
        .datum(currentAccVal)
        .attr('d', accLineVal);
}

function runLRP() {
    const relevances = {};
    const edgeRelevances = {};

    outputNodes.forEach(nodeName => {
        const node = nodeData.find(n => n.name === nodeName);
        relevances[nodeName] = node.activation;
    });

    const epsilon = 1e-6;

    hiddenNodes.forEach(hiddenNodeName => {
        const hiddenNode = nodeData.find(n => n.name === hiddenNodeName);
        const hiddenActivation = hiddenNode.activation;
        let relevanceSum = 0;

        outputNodes.forEach(outputNodeName => {
            const outputRelevance = relevances[outputNodeName];
            const hiddenIndex = hiddenNodes.indexOf(hiddenNodeName);
            const outputIndex = outputNodes.indexOf(outputNodeName);
            const weight = weightsHiddenOutput[hiddenIndex][outputIndex];
            const z = hiddenActivation * weight;

            let zSum = 0;
            hiddenNodes.forEach(hn => {
                const hnNode = nodeData.find(n => n.name === hn);
                const hnActivation = hnNode.activation;
                const w = weightsHiddenOutput[hiddenNodes.indexOf(hn)][outputIndex];
                zSum += hnActivation * w;
            });
            zSum += epsilon;

            const relevanceContribution = (z / zSum) * outputRelevance;
            relevanceSum += relevanceContribution;
            const edgeKey = `${hiddenNodeName}->${outputNodeName}`;
            edgeRelevances[edgeKey] = relevanceContribution;
        });

        relevances[hiddenNodeName] = relevanceSum;
    });

    inputNodes.forEach(inputNodeName => {
        const inputNode = nodeData.find(n => n.name === inputNodeName);
        const inputActivation = inputNode.activation;
        let relevanceSum = 0;

        hiddenNodes.forEach(hiddenNodeName => {
            const hiddenRelevance = relevances[hiddenNodeName];
            const inputIndex = inputNodes.indexOf(inputNodeName);
            const hiddenIndex = hiddenNodes.indexOf(hiddenNodeName);
            const weight = weightsInputHidden[inputIndex][hiddenIndex];
            const z = inputActivation * weight;

            let zSum = 0;
            inputNodes.forEach(inpNodeName => {
                const inpNode = nodeData.find(n => n.name === inpNodeName);
                const inpActivation = inpNode.activation;
                const w = weightsInputHidden[inputNodes.indexOf(inpNodeName)][hiddenIndex];
                zSum += inpActivation * w;
            });
            zSum += epsilon;

            const relevanceContribution = (z / zSum) * hiddenRelevance;
            relevanceSum += relevanceContribution;
            const edgeKey = `${inputNodeName}->${hiddenNodeName}`;
            edgeRelevances[edgeKey] = relevanceContribution;
        });

        relevances[inputNodeName] = relevanceSum;
    });

    nodeData.forEach(node => {
        node.relevance = relevances[node.name];
    });

    edgeSelection.each(function(d) {
        const edgeKey = `${d.source}->${d.target}`;
        d.relevance = edgeRelevances[edgeKey] || 0;
    });

    updateVisualizationLRP();
}

function updateVisualizationLRP() {
    const allNodeRelevances = nodeData.map(d => d.relevance);
    const nodeRelevanceColorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([0, d3.max(allNodeRelevances)]);

    nodeSelection
        .attr('fill', d => {
            if (d.dead) {
                return 'black';
            } else {
                return nodeRelevanceColorScale(d.relevance);
            }
        });

    activationValueSelection
        .text(d => (d.relevance !== null && d.relevance !== undefined) ? d.relevance.toFixed(2) : '')
        .attr('fill', d => {
            if (d.dead) {
                return 'white';
            } else {
                const bgColor = d3.color(nodeRelevanceColorScale(d.relevance));
                const luminance = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
                return luminance > 186 ? 'black' : 'white';
            }
        });

    const allEdgeRelevances = [];
    edgeSelection.each(function(d) {
        allEdgeRelevances.push(Math.abs(d.relevance));
    });

    edgeRelevanceColorScale = d3.scaleSequential(d3.interpolateOranges)
        .domain([0, d3.max(allEdgeRelevances)]);

    edgeSelection
        .attr('stroke', d => edgeRelevanceColorScale(Math.abs(d.relevance)));

    edgeLabelSelection
        .text(d => (d.relevance !== null && d.relevance !== undefined) ? d.relevance.toFixed(2) : '')
        .style('display', 'block')
        .attr('fill', 'black')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold');

    updateEdgeLabelsVisibility();
}

function scaleInputs(customValues) {
    const scaledInputs = {};
    let valid = true;
    let errorMessage = '';

    inputNodes.forEach((inputName, index) => {
        let value = customValues[inputName];

        if (value === undefined) {
            errorMessage += `Value for ${inputName} is missing.\n`;
            valid = false;
            return;
        }

        if (log_scaled_vars.includes(inputName)) {
            if (value <= 0) {
                errorMessage += `Value for ${inputName} must be positive for log-transformation.\n`;
                valid = false;
                return;
            }
            value = Math.log10(value);
        }

        const mean = scalingInfo.mean[index];
        const scale = scalingInfo.scale[index];
        scaledInputs[inputName] = (value - mean) / scale;
    });

    if (!valid) {
        alert(errorMessage);
        return null;
    }

    return scaledInputs;
}

function reverseScaleInputs(scaledInputs) {
    const realValues = {};

    inputNodes.forEach((inputName, index) => {
        let scaledValue = scaledInputs[inputName];
        const mean = scalingInfo.mean[index];
        const scale = scalingInfo.scale[index];
        let realValue = scaledValue * scale + mean;

        if (log_scaled_vars.includes(inputName)) {
            realValue = Math.pow(10, realValue);
        }

        realValues[inputName] = parseFloat(realValue.toFixed(2));
    });
    return realValues;
}