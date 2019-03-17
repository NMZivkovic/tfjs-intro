getData = getDataFunction;
createModel =  createModelFunction;
prepareData = prepareDataFunction
trainModel = trainModelFunction;
displayData = displayDataFunction;
evaluateModel = evaluateModelFunction;

async function run() {
  
  const data = await getData();
  displayData(data);
  
	const model = createModel();  
	tfvis.show.modelSummary({name: 'Model Summary'}, model);

	const tensorData = prepareData(data);
	const {inputs, outputs} = tensorData;
		
	await trainModel(model, inputs, outputs, 100);
	console.log('Done Training');
  
  await evaluateModel(model, inputs, outputs);
}

/**
  * @desc retrieves data from defined location
  * @return wine data as json
*/
async function getDataFunction() {
  const wineDataReq = await fetch('https://raw.githubusercontent.com/NMZivkovic/file_hosting/master/wine_quality.json');  
  const wineData = await wineDataReq.json();  
  return wineData;
}


/**
  * @desc plots one 
  * @param array values - array of values
  * @param string name - name of the plot
  * @param string xoutput - x name 
  * @param string youtput - y name
*/
function singlePlot(values, name, xoutput, youtput)
{
  tfvis.render.scatterplot(
    {name: name},
    {values}, 
    {
      xoutput: xoutput,
      youtput: youtput,
      height: 300
    }
  );
}

/**
  * @desc plots one 
  * @param json data - complete json that contains wine quality data 
*/
function displayDataFunction(data){
  let displayData = data.map(d => ({
    x: d.alcohol,
    y: d.quality,
  }));

  singlePlot(displayData, 'Alchocol v Quality', 'Alchocol', 'Quality')

  displayData = data.map(d => ({
    x: d.chlorides,
    y: d.quality,
  }));

  singlePlot(displayData, 'Chlorides v Quality', 'Chlorides', 'Quality')

  displayData = data.map(d => ({
    x: d.citric_acid,
    y: d.quality,
  }));

  singlePlot(displayData, 'Citric Acid v Quality', 'Citric Acid', 'Quality')
}

/**
  * @desc creates tensorflow graph
  * @return model
*/
function createModelFunction() {
  const model = tf.sequential(); 
  model.add(tf.layers.dense({inputShape: [11], units: 50, useBias: true, activation: 'relu'}));
  model.add(tf.layers.dense({units: 30, useBias: true, activation: 'tanh'}));
  model.add(tf.layers.dense({units: 20, useBias: true, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, useBias: true, activation: 'softmax'}));

  return model;
}

/**
  * @desc creates array of input data for every sample
  * @param json data - complete json that contains wine quality data 
  * @return array of input data
*/
function extractInputs(data)
{
  let inputs = []
  inputs = data.map(d => [d.fixed_acidity, d.volatile_acidity, d.citric_acid, d.residual_sugar, d.chlorides, d.free_sulfur_dioxide, d.total_sulfur_dioxide, d.density, d.pH, d.sulphates, d.alcohol])
	return inputs;
}

/**
  * @desc converts data from json format to tensors
  * @param json data - complete json that contains wine quality data 
  * @return tuple of converted data that can be used for training model
*/
function prepareDataFunction(data) {
  
  return tf.tidy(() => {
    tf.util.shuffle(data);
    
    const inputs = extractInputs(data);
    const outputs = data.map(d => d.quality);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const outputTensor = tf.oneHot(tf.tensor1d(outputs, 'int32'), 10);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const outputMax = outputTensor.max();
    const outputMin = outputTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedoutputs = outputTensor.sub(outputMin).div(outputMax.sub(outputMin));

    return {
      inputs: normalizedInputs,
      outputs: normalizedoutputs,
      inputMax,
      inputMin,
      outputMax,
      outputMin,
    }
  });  
}

/**
  * @desc trains model
  * @return trained model
*/
async function trainModelFunction(model, inputs, outputs, epochs) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  const batchSize = 64;
  
  return await model.fit(inputs, outputs, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'accuracy'], 
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

/**
  * @desc evaluates the model
*/
async function evaluateModelFunction(model, inputs, outputs)
{
  const result = await model.evaluate(inputs, outputs, {batchSize: 64});
  console.log('Accuracy is:')
  result[1].print();
}

document.addEventListener('DOMContentLoaded', run);

