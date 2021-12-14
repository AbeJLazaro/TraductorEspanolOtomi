import numpy as np
from torch import from_numpy
from torch.autograd import Variable
from transformer_architecture import *

# Variables globales
global max_src_in_batch, max_tgt_in_batch

"""
####################################################################
#################### Funciones importantes #########################
####################################################################
"""
# Genera máscaras subsecuentes
def subsequent_mask(size):
	"""
	Hace máscaras subsecuentes

	Parámetros:
	size: Tamaño de características
	"""
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return from_numpy(subsequent_mask) == 0

# Función para correr una época de entrenamiento
def run_epoch(data_iter, model, loss_compute):
	"""
	Entrenamiento estandar

	Parámetros
	data_iter: 
	model: Modelo 
	loss_compute: Función de perdida 
	"""

	# Inicializamos ciertos valores
	total_tokens = 0
	total_loss = 0

	# Iteramos sobre el número de lotes y los lotes
	for batch in data_iter:
		# Hacemos la predicción del lote
		out = model.forward(batch.src, batch.trg, 
												batch.src_mask, batch.trg_mask)

		# Calculamos la perdida y almacenamos información
		loss = loss_compute(out, batch.trg_y, batch.ntokens)
		total_loss += loss
		total_tokens += batch.ntokens
	
	# se retona la perdida total por tokens
	return total_loss / total_tokens

# 
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."

		# Se hace uso de las variables globales
    global max_src_in_batch, max_tgt_in_batch

    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


"""
####################################################################
################ Clases para entrenar el modelo ####################
####################################################################
"""

########## Herramientas ##############
"""
Clase lote
"""
class Batch:
	"""
	Objeto para mantener un lote de data com máscara durante el
	entrenamiento
	"""
	def __init__(self, src, trg=None, pad=0):
		"""
		Constructor de batch

		Parámetros
		src: Datos de entrada del lote
		trg: 
		pad: Si se realiza padding
		"""
		# Se guarda la información de las entradas del lote
		self.src = src
		# Se guardan las mascaras de entrada
		self.src_mask = (src != pad).unsqueeze(-2)

		# Si trg no es None
		if trg is not None:
			# Se guarda el trg menos el último elemento
			self.trg = trg[:, :-1]
			# Se guarda en otra variable el trg menos el primer elemento
			self.trg_y = trg[:, 1:]
			# Se genera la mascara con trg menos el último y pad
			self.trg_mask = self.make_std_mask(self.trg, pad)
			# El número de tokens es el trg_y que son diferentes al padding
			self.ntokens = (self.trg_y != pad).data.sum()
	
	@staticmethod
	def make_std_mask(tgt, pad):
		"""
		Crea una mascara para esconder el padding y futuras 
		palabras
		"""
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask

"""
Clase para calcular la perdida simple
"""
class SimpleLossCompute:
	"""
	Calculo de perdida simple
	"""
	def __init__(self, generator, criterion, opt=None):
		"""
		Constructor 

		Parámetros
		generator: 
		criterion:
		opt: 
		"""
		# Se guardan las partes para obtener la perdida 
		self.generator = generator
		self.criterion = criterion
		self.opt = opt
			
	def __call__(self, x, y, norm):
		"""
		Cuando se llama al objeto

		Parámetros
		x: Valor de entrada
		y: Valor real
		norm: Norma
		"""
		# Se genera la generación(?) de un x
		x = self.generator(x)
		# Se calcula la perdida con el criterio entre la norma
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
													y.contiguous().view(-1)) / norm
		# Se calcula el paso hacia atrás de la perdida
		_ = loss.backward()

		# Si hay optimizador
		if self.opt is not None:
			# Se hace un paso y se modifica el optimizador
			_ =self.opt.step()
			_ =self.opt.optimizer.zero_grad()
		
		# Se retornan los items(?) de la perdida por la norma
		return loss.data.item() * norm
    
class NoamOpt:
	"""
	Implementa un wrapper para la optimización Noam
	"""
	def __init__(self, model_size, factor, warmup, optimizer):
		"""
		Constructor

		model_size: Tamaño del modelo
		factor: 
		warmup: Factor de enfriamiento
		optimizer: Optimizador
		"""

		# Se guardan ciertos datos
		self.optimizer = optimizer
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		# Atributos privados
		self._step = 0
		self._rate = 0
			
	def step(self):
		"""
		Modifica parámetros y califica
		"""
		# Aumenta el contador de pasos
		self._step += 1
		# Calcula la calificación
		rate = self.rate()
		# Agrega la calificación a los parámetros del optimizador
		for p in self.optimizer.param_groups:
				p['lr'] = rate
		# modifica el rate anterior
		self._rate = rate
		# hace un paso en el optimizador
		self.optimizer.step()
			
	def rate(self, step = None):
		"""Implementa lrate (?) """
		# Si no hay step
		if step is None:
				step = self._step
		# Función que devuelve
		return self.factor * \
				(self.model_size ** (-0.5) *
				min(step ** (-0.5), step * self.warmup ** (-1.5)))

class LabelSmoothing(nn.Module):
	"Implement label smoothing."
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None
			
	def forward(self, x, target):
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))