import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

"""
####################################################################
#################### Funciones importantes #########################
####################################################################
"""

# Esta función es importante para hacer varias copias de un módulo
def clones(module, N):
	"""
	Genera N copias de module
	"""
	return nn.ModuleList( [copy.deepcopy(module) for _ in range(N)] )

# Atención
def attention(query, key, value, mask=None, dropout=None):
	"""
	Implementación de la atención

	Parámetros:
	query: Vector query
	key: Vector key
	value: Vector value
	mask: Mascara para aplicar en la atención
	"""

	# Calculamos La atención scalada del producto punto 
	# (Scaled Dot Product Attention)

	d_k = query.size(-1)

	scores = torch.matmul( query, key.transpose(-2,-1) ) / math.sqrt(d_k)

	if mask is not None:
		scores = scores.masked_fill( mask == 0, -1e9)
	
	p_attn = F.softmax( scores, dim = -1 )

	if dropout is not None:
		p_attn = dropout(p_attn)

	return torch.matmul( p_attn, value ), p_attn 



"""
####################################################################
################### Clases de la Arquitectura ######################
####################################################################
"""

"""
Arquitectura Encoder - Decoder
"""
class EncoderDecoder(nn.Module):
	"""
	Arquitectura básica o estandar Encoder-Decoder. Es la base
	para muchos modelos como transformers, RNN, etc.
	"""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		"""
		Constructor de EncoderDecoder

		Parámetros:
		encoder: Codificador, módulo
		decoder: Decodificador, módulo
		src_embed: Embeddings de la entrada
		tgt_embed: Embeddings de la salida
		generator: Capa lineal y softmax de la arquitectura
		"""
		
		# Constructor del padre de EncodeDecoder usando al mismo objeto
		# instanciado
		super(EncoderDecoder, self).__init__()

		# Se guarda internamente el Encoder y el Decoder
		self.encoder = encoder 
		self.decoder = decoder 

		# Se guardan también los embeddings tanto de la entrada
		# como de la salida
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed

		# Capa lineal y softmax de la arquitectura
		self.generator = generator

	# Función para hacer un paso hacia adelante
	def forward(self, src, tgt, src_mask, tgt_mask):
		"""
		Genera un paso hacia delante

		Parámetros:
		src: Entrada (lista o tensor de indices)
		tgt: Salida (lista o tensor de indices)
		src_mask: **************************************
		tgt_mask:

		return salida del paso hacia delante
		"""

		# Aplica algo como que la salida, la predicción
		# Será Y = decoder(encoder(x))

		return self.decode( self.encode( src, src_mask ),
												src_mask,
												tgt,
												tgt_mask)
		
	def encode(self, src, src_mask):
		"""
		Genera la codificación de la entrada

		Parámetros:
		src: Entrada (lista o tensor de indices)
		src_mask: **************************************

		return salida del codificador
		"""

		# Al codificador se le pasan los embeddings de la entrada y
		# ***************
		return self.encoder( self.src_embed(src), src_mask )

	def decode(self, memory, src_mask, tgt, tgt_mask):
		"""
		Genera la decodificación de la salida (memoria) del encoder

		Parámetros:

		memory: Salida del codificador
		src_mask: **************************************
		tgt: Salida (lista o tensor de indices)
		tgt_mask: **************************************
		"""

		# Al decoder se le pasan los embeddings de la salida, 
		# la salida del encoder, **************************************
		return self.decoder( self.tgt_embed(tgt), memory, src_mask, tgt_mask )


########## Parte Final del transformer ##############
"""
Generador, es la última parte del transformer 
"""
class Generator(nn.Module):
	"""
	Define el paso de generación lineal + softmax
	"""

	def __init__(self, d_model, vocab):
		"""
		Constructor del generador lineal

		Parámetros:
		d_model: Dimensión del modelo
		vocab: Tamaño del vocabulario
		"""
		# Constructor de la super clase
		super(Generator, self).__init__()
		# Se define una función lineal con la dimensión
		# de los parámetros (tam_entrada, tam_salida)
		# de la forma que se proyectará un tensor de
		# (n_x, tam_entrada) -> (n_x, tam_salida)
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		"""
		Paso hacia delante del generador
		
		Parámetros:
		x: Entradas para el generador (la salida de los
		decodificadores)

		return probabilidades generadas por softmax
		"""

		# Self.proj realiza la capa linear de la salida
		# log_softmax genera la capa softmax para la salida
		# Se especifica que se realice la capa softmax a lo largo de la
		# ultima dimensión, para que se aplique a las salidas de cada
		# entrada o X

		# El profe puso solo softmax en vez de log_softmax
		#return F.softmax( self.proj(x), dim=-1 )
		return F.log_softmax( self.proj(x), dim=-1 )

########## Parte Primera del transformer ##############
"""
Arquitectura del encoder
"""
class Encoder(nn.Module):
	"""
	Núcleo del Encoder, se compone de N capas iguales
	"""

	def __init__(self, layer, N):
		"""
		Constructor del Encoder

		Parámetros
		layer: Capa o Modulo base del encoder
		N: Número de copias de la capa o modulo base
		"""
		# El encoder se define como Nx capas de codificación
		# En esta clase se implementa la conexión secuencial de
		# los Nx bloques o capas del encoder, uno pasandole su salida
		# al siguiente.

		# Constructor de la superclase de Encoder
		super(Encoder, self).__init__()

		# Pila (stack) de capas identicas
		self.layers = clones(layer, N)

		# Capa de normalización 
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""
		Paso hacia delante. Pasa cada entrada y máscara a través de 
		cada capa.

		Parámetros:
		x: Entrada (se le pasan enbeddings)
		mask: Mascara para las entradas
		"""
		# En esta parte del código se implementa lo que se conectó en
		# la clase EncoderDecoder.encode, donde se hace llamada al encoder
		# Aunque no entiendo muy bien como funciona PyTorch

		# Se conecta cada capa con la siguiente de esta forma
		# La X para la siguiente capa será la salida de la capa
		# anterior
		# La máscara se mantiene
		for layer in self.layers:
			x = layer(x, mask)

		# Se retorna la salida normalizada con la capa normalizadora
		# definida en el constructor
		return self.norm(x)


"""
Capa del encoder
"""
class EncoderLayer(nn.Module):
	"""
	Capa individual para el encoder
	"""

	"""
	En la imágen del transformer se muestra como el Encoder se conforma
	de Nx capas iguales. Estas con cada una de esas capas. 
	Cada capa se conforma de:
		* Bloque multicabezal de atención propia (self_attention)
		* Bloque de suma y normalización
		* Bloque lineal feed forward
		* Bloque de suma y normalización
	"""
	def __init__(self, size, self_attn, feed_forward, dropout):
		"""
		Constructor de cada capa del encoder

		Parámetros
		size: Tamaño del modelo, dimensión de los embeddings
		self_attn: Capa de self_attention
		feed_forward: Capa feed_forward
		dropout: Probabilidad para hacer el dropout
		"""
		# Constructor de la superclase de EncoderLayer
		super(EncoderLayer, self).__init__()

		# * Bloque multicabezal de atención propia (self_attention)
		self.self_attn = self_attn

		# * Bloque lineal feed forward
		self.feed_forward = feed_forward

		# Se describe a arquitectura de cada capa del encoder como 
		# separada en 2 grupos, o subcapas, la primera con atención
		# y suma + normalización, y la segunda con feed_forward y 
		# suma + normalización, por eso se indican dos sub_capas aquí

		# Genera 2 capas de SublayerConnection con el tamaño del modelo y
		# el parámetro del dropout de inicialización (probabilidad)
		self.sublayer = clones( SublayerConnection(size, dropout) , 2 )

		# Se guarda la dimensiónd el modelo(capa)
		self.size = size

	def forward(self, x, mask):
		"""
		Paso hacia delante de cada una de las capas del encoder
		
		Parámetros:
		x: Entradas de la capa
		mask: Mascaras para la capa
		"""
		# Para la primer sub-capa se aplica la atención (pero se pasa como
		# función por que SublayerConnection primero normaliza la entrada
		# x antes de aplicar la función) y ya internamente hace el dropout
		# y hace las conexiones raras de la imágen de los trasnformers
		# donde a la siguiente subcapa se le pasa la suma y normalización
		# de lo que sale de la atención y la entrada
		x = self.sublayer[0]( x, lambda x: self.self_attn( x, x, x, mask ) )

		# Para la segunda subcapa, es lo mismo, pero ahora podemos pasar 
		# la capa feed_forward completa ya que el módulo no necesita
		# un mapeo de parámetros tan complicado
		return self.sublayer[1]( x, self.feed_forward )


########## Parte Segunda del transformer ##############
"""
Arquitectura del decoder
"""
class Decoder(nn.Module):
	"""
	Decodificador con N capas y enmascaramiento
	"""

	def __init__(self, layer, N):
		"""
		Constructor del Decoder

		Parámetros:
		layer: Capa del decoder
		N: Número de capas del decoder
		"""
		# Constructor de la super clase del decoder
		super(Decoder, self).__init__()

		# Se definen las capas del decoder
		self.layer = clones(layer, N)

		# Se define el normalizador para el decoder
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""
		Define el paso hacia delante del decoder
		
		Parámetros:
		x: Entradas-Salidas, embeddings del target
		memory: Memoria o salidas del encoder
		src_mask: Mascaras de la entrada
		tgt_mask: Mascaras de la salida
		"""
		# Se conectan todas las capas de la siguiente forma
		# La salida de la primera capa será la entrada de la segunda.
		for layer in self.layer:
			x = layer(x, memory, src_mask, tgt_mask)

		return self.norm(x)


"""
Capa del decoder
"""
class DecoderLayer(nn.Module):
	"""
	Implementa una capa del decoder
	"""

	"""
	Según la arquitectura del transformer, el decoder se compone 
	de los siguientes grupos:
		1. Subcapa
			* Atención multicabezal enmascarada
			* Suma y normalización
		2. Subcapa
			* Atención multicabezal 
			* Suma y normalización
		3. Subcapa
			* feed forward
			* Suma y normalización
	"""

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		"""
		Constructor de una capa del decodificador

		Parámetros:
		size: Tamaño o dimensión del modelo
		self_attn: Modulo o capa de atención propia
		src_attn: Modulo o capa de atención de la entrada
		feed_forward: Modulo lineal
		dropout: Probabilidad para dropout
		"""
		# Constructor de la superclase 
		super(DecoderLayer, self).__init__()

		# Se define el tamaño del modelo(capa)
		self.size = size 

		# Se define el bloque de atención propia y de atención de entrada
		self.self_attn = self_attn
		self.src_attn = src_attn

		# Se define el bloque feed forward
		self.feed_forward = feed_forward

		# Se generan las 3 subcapas de suma y normalización
		# 3, por los 3 grupos mencionados arriba
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""
		Paso hacia delante de cada capa.
		
		Parámetros:
		x: Entradas o salidas, embeddings del target o capa anterior
		memory: Salida del codificador
		src_mask: Mascara de la entrada
		tgt_mask: Mascara de la salida
		"""

		m = memory 

		# Se definen las conexiones para el primer grupo
		x = self.sublayer[0]( x, lambda x: self.self_attn(x, x, x, tgt_mask) )

		# Se definen las conexiones para el segundo grupo
		# Recordar que la atención de esta capa se hace también con la salida
		# del codificador
		x = self.sublayer[1]( x, lambda x: self.src_attn(x, m, m, src_mask) )

		# Se definen las conexiones para el tercer grupo
		return self.sublayer[2]( x, self.feed_forward )


"""
####################################################################
################## Bloques de la Arquitectura ######################
####################################################################
"""
########## Bloque de Suma y normalización ##############

"""
Capa de normalización
"""
class LayerNorm(nn.Module):
	"""
	Construye la capa de normalización
	"""

	def __init__(self, features, eps=1e-6):
		"""
		Constructor LayerNorm

		Parámetros:
		features: Tamaño del vector
		eps: Diferencia para la división
		"""
		# Constructor superclase de LayerNorm
		super(LayerNorm, self).__init__()

		# Se generan las constantes a_2 y b_2
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))

		# Se guarda el valor del epsilon
		self.eps = eps 
	
	def forward(self, x):
		"""
		Paso hacia delante de la capa de normalización
		
		Parámetros
		x: Entradas (salidas del encoder)
		"""
		# Para normalizar una distribución mean=M, std=S a
		# una distribución normal mean=0, std=1, se resta la 
		# media a la distribución para centrarla en 0 y se 
		# divide entre la std para tener std=1

		# Se suelen ocupar el parámetro a_2 para escalar la
		# media, y el parámetro b_2 para mover la distribución
		# Normalmente se setean en 1 y 0 respectivamente para que
		# no hagan nada

		# El parámetro epsilon trata de suavizar la división para evitar
		# casos en que la std es 0, para eso se supa eps y se evita 
		# dividir entre cero

		# Se calcula la media de las entradas		
		mean = x.mean(-1, keepdim=True)

		# Se calcula la desviación estandar
		std = x.std(-1, keepdim=True)

		# Se retorna el vector normalizado
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

"""
Capa para implementar el dropout
"""
class SublayerConnection(nn.Module):
	"""
	Una conexión residual seguida de una capa de normalización.
	Por simplificación, la normalización se coloca primero, en vez
	de al final
	"""

	def __init__(self, size, dropout):
		"""
		Construye la salida de cada subcapa

		Parámetros:
		size: Tamaño de la capa de normalización
		dropout: Parámetro de probabilidad para hacer las desconexiones
		"""
		# Constructor de la superclase de SublayerConnection
		super(SublayerConnection,self).__init__()

		# Capa de normalización
		self.norm = LayerNorm(size)

		# Capa dropout
		self.dropout = nn.Dropout(dropout)

		"""
		the output of each sub-layer is 
							LayerNorm( x + Sublayer(x) )
		, where is the function implemented by the sub-layer itself. 
		We apply dropout to the output of each sub-layer, before 
		it is added to the sub-layer input and normalized.

		To facilitate these residual connections, all sub-layers in the model, 
		as well as the embedding layers, produce outputs of dimension 
		d_model=512
		"""
	
	def forward(self, x, sublayer):
		"""
		Aplicamos las conexiones residuales a cualquier subcapa con el 
		mismo tamaño

		Parámetros:
		x: Entradas
		sublayer: Subcapa 
		"""

		# Aplica la formula LayerNorm( x + Sublayer(x) )
		# pero inveirte el orden de las funciones de forma
		# x + Sublayer( LayerNorm(x) )
		return x + self.dropout( sublayer( self.norm(x) ) )


################## Bloque Atención #######################
"""
Cabezales multi-atención
"""
class MultiHeadedAttention(nn.Module):
	"""
	Implementación de los bloques de atención multiples
	"""

	def __init__(self, h, d_model, dropout=0.1):
		"""
		Contructor del bloque de Atención

		Parámetros:
		h: Número de capas de atención paralelas
		d_model: Dimensión del modelo
		dropout: Probabilidad de dropout
		"""
		# Constructor de la superclase
		super(MultiHeadedAttention, self).__init__()

		# No se por que, pero deben ser multiplos
		assert d_model % h == 0, "h y d_model no son multiplos"

		# Asumimos que d_v es igual a d_k
		self.d_k = d_model // h

		# Guardamos el valor del número de cabezales
		self.h = h 
		
		# Agregamos 4 capas lineales para tener la variación en las 
		# capas de atención
		self.linears = clones(nn.Linear( d_model, d_model ), 4)

		# Atención
		self.attn = None 

		# Capa de dropout
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask = None):
		"""
		Paso hacia delante de la atención

		Parámetros:
		query: Vector query
		key: Vector key
		value: Vector value
		mask: Mascara para aplicar en la atención
		"""

		# Si se pasa la máscara
		if mask is not None:
			mask = mask.unsqueeze(1)

		# El número de lotes tendrá el tamaño de query en 0
		nbatches = query.size(0)

		# 1.- Hacer la projección lineal en lote desde d_model => h x d_k
		# Aquí se hace la projección lineal de q, k y v
		query, key, value = [ l(x).view( nbatches, -1, self.h, self.d_k ).transpose(1,2)
													for l, x in zip(self.linears, (query, key, value) ) ]

		# 2.- Aplica atención a todos los vectores projectados en lotes
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

		# 3.- Concatenar usando un reshape (view) y aplicar una capa lineal final
		x = x.transpose(1,2).contiguous().view( nbatches, -1, self.h * self.d_k )

		return self.linears[-1](x)


################## Bloque Feed-Forward #######################

"""
Bloque Feed-Forward 
"""
class PositionwiseFeedForward(nn.Module):
	"""
	Implementa la ecuación del feed_forward networks para el transformer
	"""

	"""
	Se implementa una red de dos capas sencillas con una ReLU en medio

	FFN(x) = max(0, xW_1 + b_1)W_x + b_2 
	"""

	def __init__(self, d_model, d_ff, dropout=0.1):
		"""
		Constructor del bloque o capa FFN
		
		Parámetros
		d_model: Dimensión del modelo
		d_ff: Dimensión del feed_forward
		dropout: Probabilidad de dropout
		"""
		# Constructor de la superclase
		super(PositionwiseFeedForward, self).__init__()

		# Se generan las matrices w_1 y w_2 con sus respectivas
		# dimensiones
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)

		# Se genera la capa de dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""
		Paso hacia delante 

		Parámetros:
		x: Entradas
		"""
		
		# Simplemente se hace la multiplicación
		return self.w_2( self.dropout( F.relu( self.w_1( x ) ) ) )
		

################## Bloque Embeddings #######################

"""
Bloque Embeddings
"""
class Embeddings(nn.Module):
	"""
	Implementación de los embeddings
	"""

	def __init__(self, d_model, vocab):
		"""
		Capa de embeddings

		Parámetros
		d_model: Tamaño del modelo
		vocab: Tamaño del vocabulario
		"""
		# Constructor de la superclase
		super(Embeddings, self).__init__()

		# Usamos un modulo ya hecho de embeddings
		self.lut = nn.Embedding( vocab, d_model )

		# Guardamos la dimensión del modelo
		self.d_model = d_model

	def forward(self, x):
		"""
		Paso hacia delante para los embeddings

		Parámetros
		x: Entradas
		"""

		# Enviamos la projección de las entradas en los embeddings
		# pero multiplicados por un factor
		return self.lut(x) * math.sqrt(self.d_model)

"""
Bloque de codificación posicional
"""

class PositionalEncoding(nn.Module):
	"""
	Generación de la codificación posicional
	"""

	def __init__(self, d_model, dropout, max_len=5000):
		"""
		Constructor de los positional encoding

		Parámetros
		d_model: Dimensión del modelo
		dropout: Probabilidad de dropout
		max_len: Longitud Máxima de elementos en un vector de posición
		"""
		# Constructor de la superclase
		super(PositionalEncoding, self).__init__()

		# Generamos la capa de dropout
		self.dropout = nn.Dropout(p=dropout)

		# Calculamos las posiciones codificadas una vez en el espacio
		# logaritmico
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp( torch.arange(0, d_model,2) * -(math.log(10000.0) / d_model) )

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)

		self.register_buffer("pe", pe)

	def forward(self, x):
		"""
		Paso de los positional encoding

		Parámetros
		x: Entradas a las cuales agregarle el positional encoding
		"""
		x = x + Variable( self.pe[:, :x.size(1)], requires_grad=False )

		return self.dropout(x)

