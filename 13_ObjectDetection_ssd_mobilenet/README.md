<h2> Detecção de 20 classes de objetos com SSD e MobileNet </h2>

<h3> Download dos arquivos </h3>

<ul>
  <li>https://github.com/chuanqi305/MobileNet-SSD </li>
</ul>

<p>No arquivo README.md em </p>
<p><b>   Network</p>
<p>MobileNet-SSD</b></p>

<p>Abrir o link de <b>deploy</b> (arquivo MobileNetSSD_deploy.caffemodel) e baixar o arquivo.</p>
<ul>
  <li>https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view </li>
</ul>

<p>Na pasta voc</p>
<ul>
  <li>https://github.com/chuanqi305/MobileNet-SSD/tree/master/voc </li>
</ul>

<p> Baixar <b>MobileNetSSD_deploy.prototxt</b> e não esquecer de excluir o termo <b>.txt</b> do nome.</p>

<h4>Referência</h4>
<ul>
  <li>https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/ </li>
</ul>

<h3> Código </h3>

<h4>SSD - Single Shot Detector </h4>
<p>Arquitetura para detectar objetos</p>
<h4>MobileNet </h4>
<p>Substitui a VGG-16 usada como base original do SSD para aumento da velocidade. </p>

<h4>Prototxt </h4>
<p>Arquitetura do modelo (camadas).</p>    
<h4>Caffemodel </h4>
<p>Pesos do modelo</p>

<h4>300x300</h4>
<p>Entrada comum para SSD e Faster R-CNN</p>
<h4>(104,0, 177,0, 123,0)</h4> 
<p>Valores RGB médios em todos os pixels no conjunto de treinamento, usados para a subtração média </p>

<h4>0.007843</h4> 
<p>Normalization factor (normalization is done via the authors of the MobileNet SSD implementation))</p>
<h4>127.5 </h4> 
<p>Mean subtraction</p>

<h4>Referências</h4>
<ul>
  <li>https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/ </li>
  <li>https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/ </li>
  <li>https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection</li>
</ul>
