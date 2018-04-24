import numpy as np
import matplotlib.pyplot as plt


def getData():
  arch 		= open('TWTR.csv', 'r')
  adj_close = []
  for j in arch:
    var = list(j.strip().split(","))
    adj_close.append(var[5])
  adj_close.remove('Adj Close')
  arch.close()
  map(float, adj_close)
  return adj_close


#dt:   diferencial de tiempo
#b:    incremento browniano
#W:    camino browniano
def Brownian(seed, N):
  np.random.seed(seed)
  dt 	= 1./N
  b 	= np.random.normal(0., 1., int(N)) * np.sqrt(dt)
  W 	= np.cumsum(b)
  return W, b


def daily_returns(adj_close):
	returns = []
	for i in range(0, len(adj_close) - 1):
		today 			= float(adj_close[i+1])
		yesterday 		= float(adj_close[i])
		daily_return 	= (today - yesterday)/today
		returns.append(daily_return)
	return returns


#S0:		precio inicial
#mu:		media
#sigma:		desviacion
#W:			movimiento browniano
#T:			periodo de tiempo
#N:			total de incrementos
def GBM(S0, mu, sigma, W, T, N):
	t = np.linspace(0., 1., N+1)
	S = []
	S.append(S0)
	for i in range(1, int(N+1)):
		drift 		= (mu - 0.5*sigma**2) * t[i]
		diffusion 	= sigma * W[i-1]
		S_temp 		= S0*np.exp(drift + diffusion)
		S.append(S_temp)
	return S, t


def main(_iter):
	S = []
	mu 		= np.mean(daily_returns(getData()))*252.
	sigma 	= np.std(daily_returns(getData()))*np.sqrt(252.)
	for i in range(1, int(_iter)+1):
		seed 	= i
		W 		= Brownian(seed, N)[0]
		sol 	= GBM(S0, mu, sigma, W, T, N)[0]
		S_temp 	= np.mean(sol)
		S.append(S_temp)
	return S


def EM(S0, mu, sigma, b, T, N, M):
    dt 		= M * (1/N) 
    L 		= N / M
    wi 		= [S0]
    for i in range(0,int(L)):
        Winc = np.sum(b[(M*(i-1)+M):(M*i + M)])
        w_i_new = wi[i]+mu*wi[i]*dt+sigma*wi[i]*Winc
        wi.append(w_i_new)
    return wi, dt

###inicialización de variables###
seed 	= np.random.randint(1, high = 100)
S0 		= float(getData()[0])
T 		= 5.
N 		= 251.
mu 		= np.mean(daily_returns(getData()))*252.
sigma 	= np.std(daily_returns(getData()))*np.sqrt(252.)
W 		= Brownian(seed, N)[0]
M 		= 1
L 		= N/M

###se obtiene una trayectoria random###
soln1 	= GBM(S0, mu, sigma, W, T, N)[0]
_time1 	= GBM(S0, mu, sigma, W, T, N)[1]

plt.plot(_time1, soln1, label='Trayectoria random', ls='--')

###otra trayectoria random###

seed 	= np.random.randint(1, high = 100)
W 		= Brownian(seed, N)[0]

soln2 	= GBM(S0, mu, sigma, W, T, N)[0]
_time2 	= GBM(S0, mu, sigma, W, T, N)[1]
plt.plot(_time2, soln2, label='Trayectoria random', ls='--')

###inicialización de método numérico###
seed 	= np.random.randint(1, high = 100)
W 		= Brownian(seed, N)[0]
b 		= Brownian(seed, N)[1]

em 		= EM(S0, mu, sigma, b, T, N, M)[0]
time 	= np.linspace(0., 1., L+1)

#print('Cantidad de trayectorias')
kk 	= input()
S 	= main(_iter = kk)
print('Valor estimado: ' , np.mean(S))

#actual = getData()
#tt = np.linspace(1, len(actual), len(actual))
#tt = [i/252. for i in tt]

#plt.plot(tt, actual)
plt.plot(time, em, label='Modelo MonteCarlo')
plt.ylabel('Precio Acción, $')
plt.title('Movimiento Browniano')
plt.legend(loc='best')
plt.show()