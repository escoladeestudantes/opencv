import numpy
import os
import ctypes
import cv2
import time
import glob

#Usar
#python 02_FaceDetection_PICO_imagens.py
#

os.system('cc ../../picornt.c -O3 -fPIC -shared -o picornt.lib.so')
pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
os.system('rm picornt.lib.so')

#
#
#

bytes = open('../../cascades/facefinder', 'rb').read()
cascade = numpy.frombuffer(bytes, dtype=numpy.uint8)

slot = numpy.zeros(1, dtype=numpy.int32)
nmemslots = 5
maxslotsize = 1024
memory = numpy.zeros(4*nmemslots*maxslotsize, dtype=numpy.float32)
counts = numpy.zeros(nmemslots, dtype=numpy.int32)

def process_frame(gray):
	#
	maxface = min(gray.shape[0], gray.shape[1])
	minface = maxface//5
	#
	maxndets = 2048
	dets = numpy.zeros(4*maxndets, dtype=numpy.float32)

	ndets = pico.find_objects(
		ctypes.c_void_p(dets.ctypes.data), ctypes.c_int(maxndets),
		ctypes.c_void_p(cascade.ctypes.data), ctypes.c_float(0.0),
		ctypes.c_void_p(gray.ctypes.data), ctypes.c_int(gray.shape[0]), ctypes.c_int(gray.shape[1]), ctypes.c_int(gray.shape[1]),
		ctypes.c_float(1.1), ctypes.c_float(0.1), ctypes.c_float(minface), ctypes.c_float(maxface)
	)

	ndets = pico.update_memory(
		ctypes.c_void_p(slot.ctypes.data),
		ctypes.c_void_p(memory.ctypes.data), ctypes.c_void_p(counts.ctypes.data), ctypes.c_int(nmemslots), ctypes.c_int(maxslotsize),
		ctypes.c_void_p(dets.ctypes.data), ctypes.c_int(ndets), ctypes.c_int(maxndets)
	)

	ndets = pico.cluster_detections(
		ctypes.c_void_p(dets.ctypes.data), ctypes.c_int(ndets)
	)

	return list(dets.reshape(-1, 4))[0:ndets]

#
#
#

path = ''
for infile in glob.glob(os.path.join(path, '*.jpg')):
	imagem = cv2.imread(infile)
	if max(imagem.shape[0], imagem.shape[1]) > 640:
		imagem = cv2.resize(imagem, (0,0), fx=0.5, fy=0.5)
	#
	gray = numpy.ascontiguousarray(imagem[:, :, 1].reshape((imagem.shape[0], imagem.shape[1])))
	#
	t = time.time()
	dets = process_frame(gray) # gray needs to be numpy.uint8 array
	print("* Tempo de processamento: %d [ms]" % int(1000.0*(time.time() - t)))
	for det in dets:
		if det[3] >= 20.0: #Confianca
			print(det[3])
			cv2.circle(imagem, (int(det[1]), int(det[0])), int(det[2]/2.0), (0, 0, 255), 4)
	#
	cv2.imshow('...', imagem)
	cv2.waitKey(0)
cv2.destroyAllWindows()
