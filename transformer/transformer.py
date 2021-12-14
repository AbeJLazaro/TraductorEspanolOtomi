from transformer_architecture import *
from transformer_optimization import *
import torch
from torch import nn
from torch.autograd import Variable
import copy
from math import ceil, floor
from tqdm import tqdm

"""
####################################################################
#################### Funciones importantes #########################
####################################################################
"""
# Función para construir un modelo
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"""
	Ayuda a construir un modelo

	Parámetros
	src_vocab:
	tgt_vocab:
	N: Número de bloques para el encoder y el decoder
	d_model: Dimensión del modelo
	d_ff: Dimensión para el feed-forward
	h: Número de cabezas de atención
	dropout: Probabilidad de dropout

	return un modelo
	"""

	# C es un alias de la función o método copy.deepcopy
	c = copy.deepcopy

	# Se genera el bloque de atención
	attn = MultiHeadedAttention(h, d_model)

	# Se genera un bloque de FFN
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)

	# Se genera un positional encoder
	position = PositionalEncoding(d_model, dropout)

	# El modelo será un EncoderDecoder con
	model = EncoderDecoder(
		# Un encoder con
		Encoder(
			# N capas de codificación 
			EncoderLayer(d_model, c(attn), c(ff), dropout), 
			N),
		# Un decoder con
		Decoder(
			# N capas de decodificación
			DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),
			N),
		# Capa secuencial para embeddings de entrada
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		# Capa secuencial para embeddings de salida
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		# Generador lineal softmax para la salida
		Generator(d_model, tgt_vocab))
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)

	return model

# Función para generar el lote
def batch_gen(X,Y,batch):
	"Generate batch for a src-tgt task."
	num_ex = X.shape[0]
	nbatches = ceil(num_ex/batch)
	for i in range(nbatches):
		src = Variable(X[i:i+batch], requires_grad=False)
		tgt = Variable(Y[i:i+batch], requires_grad=False)

		yield Batch(src, tgt, 0)

"""
####################################################################
################# Funciones de decodificación ######################
####################################################################
"""
# Greedy decode
def greedy_decode(model, src, src_mask, max_len, start_symbol):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
	for i in range(max_len-1):
		out = model.decode(memory, src_mask, 
												Variable(ys), 
												Variable(subsequent_mask(ys.size(1)).type_as(src.data)))                            
		prob = model.generator(out[:, -1])
		#print("prob",prob)
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		#print("next word",next_word)
		ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

# Beam search
def beam_search(model, src, src_mask, max_len, start_symbol, N_b):
	memory = model.encode(src, src_mask)

	ys = [torch.ones(1, 1).fill_(start_symbol).type_as(src.data)]

	for it in range(max_len-1):
		val_new = None 
		idx_new = None
		
		for p in ys:
			out = model.decode(memory, src_mask, 
													Variable(p), 
													Variable(subsequent_mask(p.size(1)).type_as(src.data)))                            
			prob = model.generator(out[:, -1])
			
			values, indexes = torch.topk(prob, N_b)
			
			if val_new is None:
				val_new = values
				idx_new = indexes
			else:
				val_new = torch.cat([val_new, values], dim=1)
				idx_new = torch.cat([idx_new, indexes], dim=1)

		if it < max_len-2:
			val_b, idx_b = torch.topk(val_new, N_b)
		else:
			val_b, idx_b = torch.topk(val_new, 1)

		ys_new = list()
		for index in idx_b[0]:
			aux = idx_new[0][index]
			cot = torch.cat([ys[floor(index/N_b)], torch.ones(1, 1).type_as(src.data).fill_(aux)], dim=1)
			ys_new.append(cot) 
		
		ys = ys_new

	return ys[0]

"""
####################################################################
############ Funciones de entrenamiento y prueba ###################
####################################################################
"""

def train(model,len_src,X,Y,its=100,lr=0,betas=(0.9, 0.98),eps=1e-9,factor=1,warmup=4000,batch_size=1,pd_idx=0,smoothing=0.0,path="/",ls=False):
	# Esta puede ser la clave
	X, Y = torch.tensor(X).cuda(), torch.tensor(Y).cuda()
	if ls:
		criterion = LabelSmoothing(size=len_src, padding_idx=pd_idx, smoothing=smoothing)
	else:
		criterion = nn.CrossEntropyLoss()
	model_opt = NoamOpt(model.src_embed[0].d_model, factor, warmup, torch.optim.Adam(model.parameters(),lr=lr, betas=betas, eps=eps))	

	loss = []
	for epoch in tqdm(range(its)):
		model.train()
		epoch_loss = run_epoch(batch_gen(X,Y, batch_size), model, SimpleLossCompute(model.generator, criterion, opt=model_opt))
		loss.append(epoch_loss)
		if (epoch+1)%50 == 0:
			PATH = path+"/modelo_epoch_{}.h5".format(epoch+1)
			torch.save(model, PATH)

	model.loss = loss

def predict(model, x_input, decode_function, max_len=5, BOS=0):
	model.eval()
	src = Variable(torch.LongTensor(x_input)).cuda()
	#print("src_len",src.shape)
	n = src.shape[1]
	src_mask = Variable(torch.ones(1, 1, n)).cuda()

	return decode_function(model, src, src_mask, max_len=max_len, start_symbol=BOS).detach().reshape(max_len)
