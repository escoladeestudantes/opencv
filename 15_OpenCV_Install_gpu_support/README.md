<h2>Instalação da biblioteca OpenCV com suporte para CUDA e cuDNN (GPUs NVIDIA) no Ubuntu 18.04.5</h2>

<h3>Agradecimento</h3>

<p><b>Adrian Rosebrock </b> do site PyImageSearch.</p>

<ul>
<li>https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/</li>
</ul>

<h3>Passo #1 - Instalação driver da GPU CUDA Toolkit e cuDNN</h3>

<p>Descrito no tutorial da instalação da biblioteca TensorFlow 2.0 com CUDA 10.0 e cuDNN 7.6.4 no Ubuntu 18.04.5</p>

<h3>Passo #2 - Instalação das dependências</h3>

<p>Usadas para a instalação da biblioteca TensorFlow 2.0 com CUDA 10.0 e cuDNN 7.6.4 no Ubuntu 18.04.5</p>

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install build-essential cmake unzip pkg-config
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install python3-dev

```

<h3>Passo #3 - OpenCV</h3>

```
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip
$ mv opencv-4.2.0 opencv
$ mv opencv_contrib-4.2.0 opencv_contrib
```

<h3>Passo #4 - Ambiente virtual</h3>

<p>Realizado no tutorial da instalação da biblioteca TensorFlow 2.0 com CUDA 10.0 e cuDNN 7.6.4 no Ubuntu 18.04.5</p>

```
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ sudo pip install virtualenv virtualenvwrapper
$ sudo rm -rf ~/get-pip.py ~/.cache/pip
$ nano ~/.bashrc
```
<p>Adicionar as linhas:</p>

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```

<p>Salvar o arquivo com <b>Ctrl</b> + <b>x</b> , <b>y</b> , e <b>Enter</b> para voltar ao terminal.</p>

```
$ source ~/.bashrc
mkvirtualenv opencv_cuda -p python3
```

<p>Ambiente virtual de nome <b>opencv_cuda</b> foi criado e nele serão instaladas as bibliotecas.</p>

<h3>Passo #5 - Determinar a arquitetura da GPU</h3>

<p>Acessar o link e procurar a GPU que a sua máquina possui. A série 16xx não aparece, mas é possível encontrar descrito no site da NVIDIA. A seguir também estão os links da 1660, arquitetura Turing, logo, valor de 7.5.</p>

<ul>
<li>https://developer.nvidia.com/cuda-gpus</li>
<li>https://www.nvidia.com/pt-br/geforce/graphics-cards/gtx-1660-ti/</li>
<li>https://forums.developer.nvidia.com/t/compute-capability/110091/</li>
<li>https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability</li>
</ul>

<h3>Passo #6 - Configurar OpenCV com suporte para GPU NVIDIA</h3>

```
$ cd ~/opencv
$ mkdir build
$ cd build

```

<p>Ajustar o valor de <b>ARCH_BIN</b> de acordo com o valor encontrado (sua GPU). Para a GTX 1660: </p>

```
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=7.5 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
	-D BUILD_EXAMPLES=ON ..

```

<p>O caminho de <b>-- install path</b> será utilizado posteriormente.</p>

<p>Verificar a presença de algo semelhante a: </p>


```
...
--   NVIDIA CUDA:                   YES (ver 10.0, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             75
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 7.6.4)
```

<h3>Passo #7 - Compilar OpenCV com suporte para GPU ao módulo "dnn"</h3>

<p>Para processador com 8 núcleos o comando é: </p>

```
make -j8
```

<h3>Passo #8 - Instalar OpenCV com suporte para GPU ao módulo "dnn"</h3>

```
$ sudo make install
$ sudo ldconfig
```
<p>P caminho descrito em <install path</> é utilizado agora: </p>

```
$ ls -l /usr/local/lib/python3.5/site-packages/cv2/python-3.5
```

<p> O terminal mostrará algo como: </p>

```
total 7392
-rw-r--r- 1 root staff 7566984 dez 10 10:54 cv2.cpython-36m-x86_64-linux-gnu.so
```

<p> Os comandos a seguir podem estar com o caminho incompleto, portanto procure por eles! O segundo corresponde ao arquivo mostrado pelo terminal no último comando. </p>

```
$ cd ~/.local/bin/.virtualenvs/opencv_cuda/lib/python3.6/site-packages/
$ ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
```

<p>Se o processo ocorreu corretamente, adicionando as linhas a seguir nos códigos tem-se uso da GPU.  </p>

```
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```
