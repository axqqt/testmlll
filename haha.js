document.addEventListener("DOMContentLoaded", () => {
    const selectLmao = document.getElementsByClassName("lmao");
    console.log(selectLmao); // You can use this if needed for your logic
    
    // Function to extract relevant data (Horsepower and Miles_per_Gallon)
    function extractData(obj) {
      return { x: obj.Horsepower, y: obj.Miles_per_Gallon };
    }
  
    // Function to remove data entries that are null or undefined
    function removeErrors(obj) {
      return obj.x != null && obj.y != null;
    }
  
    // Function to run TensorFlow.js model training and visualization
    async function runTF() {
      try {
        // Fetch JSON data from a local file
        const response = await fetch('./cardata.json'); // Path to the JSON file
        const values = await response.json();
  
        console.log(values); // Log the data for debugging
  
        // Process and clean the data
        values = values.map(extractData).filter(removeErrors);
        
        if (values.length === 0) {
          console.log("No valid data to display");
          return;
        }
  
        // Create the surface for visualization (using tfvis)
        const surface1 = document.getElementById('surface1');
        
        // Plot original data (Horsepower vs MPG)
        tfPlot(values, surface1);
  
        // Shuffle data for training
        tf.util.shuffle(values);
  
        // Separate inputs (x) and labels (y)
        const inputs = values.map(obj => obj.x);
        const labels = values.map(obj => obj.y);
  
        // Convert to 2D tensors (for TensorFlow.js model)
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
        // Normalize the inputs and labels
        const inputMin = inputTensor.min();
        const inputMax = inputTensor.max();
        const labelMin = labelTensor.min();
        const labelMax = labelTensor.max();
  
        const nmInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const nmLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
        // Build a simple neural network model
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
        model.add(tf.layers.dense({ units: 1, useBias: true }));
  
        // Compile the model with a loss function and optimizer
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
  
        // Train the model with the normalized data
        const batchSize = 25;
        const epochs = 50;
        const callbacks = tfvis.show.fitCallbacks(surface1, ['loss'], { callbacks: ['onEpochEnd'] });
        
        // Train the model asynchronously
        await model.fit(nmInputs, nmLabels, { batchSize, epochs, shuffle: true, callbacks });
  
        // Generate predictions
        let unX = tf.linspace(0, 1, 100);
        let unY = model.predict(unX.reshape([100, 1]));
  
        // De-normalize the predictions
        const unNormunX = unX.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormunY = unY.mul(labelMax.sub(labelMin)).add(labelMin);
  
        // Convert predictions to arrays
        unX = unNormunX.dataSync();
        unY = unNormunY.dataSync();
  
        // Prepare the predicted values for plotting
        const predicted = Array.from(unX).map((val, i) => {
          return { x: val, y: unY[i] };
        });
  
        // Plot the original and predicted data
        tfPlot([values, predicted], surface1);
      } catch (error) {
        console.error('Error fetching or processing data:', error);
      }
    }
  
    // Function to render the scatter plot with TensorFlow.js visualization (tfvis)
    function tfPlot(values, surface) {
      tfvis.render.scatterplot(surface,
        { values: values, series: ['Original', 'Predicted'] },
        { xLabel: 'Horsepower', yLabel: 'MPG' });
    }
  
    // Run the TensorFlow.js training and visualization process
    runTF();
});
