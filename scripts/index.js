var model;
var x = 0;
var y = 0;
async function loadmodel(){
  model = await tf.loadLayersModel('https://Hsu-hui.github.io/tfjs-MNIST-demo/model/MNIST_Model.json');
}
loadmodel();
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var clear = document.getElementById('clear');
var execute = document.getElementById('execute');
var result = document.getElementById('result');
canvas.style.backgroundColor = '#000000';
ctx.strokeStyle = "#ffffff";
ctx.lineWidth = 10;

//回傳滑鼠在canvas上的座標
function getMousePos(canvas, evt) {
  var rect = canvas.getBoundingClientRect();
  //getBoundingClientRect 取得物件完整座標資訊，包含寬高等
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY  - rect.top
  };   
};

function mouseMove(evt) {
  var mousePos = getMousePos(canvas, evt);
  //透過getMousePos(canvas, evt)取得滑鼠座標
  ctx.lineTo(mousePos.x, mousePos.y);
  //利用取回的值畫線
  ctx.stroke();
};

canvas.addEventListener('mousedown', function(evt) {
  var mousePos = getMousePos(canvas, evt);
  //按下去即取得第一次的座標
  evt.preventDefault();
  ctx.beginPath();
  ctx.moveTo(mousePos.x, mousePos.y);  
  //每次的點用moveTo去區別開，如果用lineTo會連在一起  
  canvas.addEventListener('mousemove', mouseMove, false);
});

canvas.addEventListener('mouseup', function() {
  canvas.removeEventListener('mousemove', mouseMove, false);
}, false);

//清除畫布
clear.addEventListener('click', function() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});
//辨識數字
execute.addEventListener('click', function() {
  var size=28;
  var Pixels = tf.browser.fromPixels(canvas)
              .resizeNearestNeighbor([28, 28])
              .mean(2)
              .expandDims(2)
              .expandDims()
              .toFloat();
  var prediction = model.predict(Pixels);
  var _result = prediction.arraySync();
  _result = _result[0].indexOf(Math.max.apply(null, _result[0]));
  result.innerHTML ="辨識數字為(The Numbers)"+_result;
  console.log(_result);
  prediction.dispose();
});
