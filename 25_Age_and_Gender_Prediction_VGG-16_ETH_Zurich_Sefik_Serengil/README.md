<div style="text-align:center"><a href="https://www.youtube.com/watch?v=1q6w1JqFgkM"><img src="https://i.imgur.com/1lLGygy.jpg"/></a></div>

<h3>OpenCV – Predição de idade e gênero com a arquitetura VGG-16 - ETH Zurich</h3>

<p>Predição de idade e gênero com a arquitetura VGG-16 baseando-se no trabalho de pesquisadores da Swiss Federal Institute of Technology em Zurich (ETH Zurich) a partir da implementação compartilhada por Sefik Ilkin Serengil com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p><b>Estimativa da idade</b></p>
<ul>
<li>https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel</li>
<li>https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt</li>
</ul>

<p><b>Estimativa do gênero</b></p>
<ul>
<li>https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel</li>
<li>https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt</li>
</ul>

<h4>Detector facial da biblioteca Dlib</h4>

<p>Baixar o arquivo <b>mmod_human_face_detector.dat.bz2</b> no link a seguir e extrair na mesma pasta que o código para CNN MMOD está. </p>

<ul><li>http://dlib.net/files/</li></ul>

<h4>Detector facial da biblioteca OpenCV</h4>

<p><b>Método 01</b></p>
<ul><li>https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector</li></ul>

<p>Baixar arquivos <b>weights.meta4</b>, <b>download_weights.py</b>, <b>deploy.prototxt</b> e <b>opencv_face_detector.pbtxt</b></p>
<p>Abrir o arquivo no GitHub</p>
<p>No canto direito clicar em Raw</p>
<p>Na página que abrir clicar no botão secundário do mouse e Salvar Como</p>

<p>Pelo terminal de comando, na pasta em que os arquivos estão, executar</p> 

```
python download_weights.py
```

Dois arquivos são baixados: <b>opencv_face_detector_uint8.pb</b> e <b>res10_300x300_ssd_iter_140000_fp16.caffemodel</b>

<p>Excluir o termo '.txt' do nome dos arquivos
<b>deploy.prototxt</b> e 
<b>opencv_face_detector.pbtxt</b></p>


<p><b>Método 02</b></p>

```
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
```

<p>Baixar opencv_face_detector.pbtxt</p>

<p><b>Tensorflow</b></p>
<p>arquivo = opencv_face_detector_uint8.pb</p>
<p>url = https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb</p>

<p><b>Referência: ProgrammerSought</b></p>
<p>https://www.programmersought.com/article/24774823025/</p>

<p><b>Caffemodel</b></p>
<p>arquivo = deploy.prototxt</p>
<p>url  = https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt</p>
<p>hash = 006BAF926232DF6F6332DEFB9C24F94BB9F3764E</p>
<b>Pesos</b>
<p>arquivo = res10_300x300_ssd_iter_140000.caffemodel</p>
<p>url  = https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel</p>
<p>hash = 15aa726b4d46d9f023526d85537db81cbc8dd566</p>
<p>tamanho = 10.1 MB</p>

<p>Referência: Amro
<p>https://amroamroamro.github.io/mexopencv/opencv/dnn_face_detector.html</p>

<p>Excluir o termo '.txt' do nome dos arquivos <b>deploy.prototxt</b> e <b>opencv_face_detector.pbtxt</b></p>

<p> <b>Referências: </b></p>

<ul>
<li>https://sefiks.com/2020/09/07/age-and-gender-prediction-with-deep-learning-in-opencv/</li>
<li>https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/</li>
<li>https://github.com/escoladeestudantes/opencv/tree/main/10_Baixar_FaceDetection_res10_ssd</li>
<li>https://github.com/escoladeestudantes/dlib/tree/main/01_FaceDetection_HOG_SVM_e_CNN_MMOD</li>
</ul>

