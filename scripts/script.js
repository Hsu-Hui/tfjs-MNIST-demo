import {MnistData} from './data.js';

const BATCH_SIZE = 512;
const TRAIN_DATA_SIZE = 5500;
const TEST_DATA_SIZE = 1000;
async function showExamples(data) {
    // Create a container in the visor
    const surface =
      tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
  
    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];
    
    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });
      
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = 'margin: 4px;';
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);
  
      imageTensor.dispose();
    }
  }
  
  async function run() {  
    const data = new MnistData();
    await data.load();
    await showExamples(data);
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  
    await train(model, data);
    await showAccuracy(model, data);
    await showConfusion(model, data);
    //await model.save('downloads://MNIST_Model');
   
  }
  
  document.addEventListener('DOMContentLoaded', run);

function getModel() {
  //使用序列模型
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;  
  
//在卷積神經網絡的第一層指定輸入形狀。 然後指定在這一層發生的卷積操作參數
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // 最大池化層充當一個區域的使用最大值的下採樣，而不是平均。
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // 重複另一個卷積層+最大池化層。
  // 注意在卷積中有更多的過濾器。
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  //將 2D 過濾器的輸出扁平化為一維向量，用於輸入到最後一層。 這是一種準備高維數據到最終分類輸出層的常見做法。
  model.add(tf.layers.flatten());

  // 最後一層是一個密集層，它有 10 個輸出單元，每個單元一個類別。
  // (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // 選擇優化器、損失函數和準確度指標，然後編譯並返回模型
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}
//訓練模型
async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
//預測test data
function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

//準確率
async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}
//混淆矩陣
async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});
  labels.dispose();
}

