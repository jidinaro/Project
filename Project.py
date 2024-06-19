import numpy as np
import matplotlib.pyplot as plt

# Parámetros
beta = 0.98
rho = 0.94
sigma_e = 0.015
mu = -0.5 * sigma_e**2
sigma = 2
d0 = 0.17
d1 = 1.2
psi  = 0.5
r = 0.01
num_periods = 300  
# Grid de Y
y_fcst = np.zeros(num_periods)
y_fcst[0] = np.random.uniform(0.1, 2.0) 

# Simulación de Y
for t in range(1, num_periods):
    y_fcst[t] = rho * np.log(y_fcst[t-1]) + (1 - rho) * mu + np.random.normal(0,sigma_e)
    y_fcst[t] = np.exp(y_fcst[t])

# Función de utilidad
def u(c, sigma):
    return (c**(1 - sigma)) / (1 - sigma)

# Penalización por autarquía
def h(y, d0, d1):
    return d0 * y + d1 * y**2

# Grid Value functions, precios y default prob
V = np.zeros(num_periods)
V_A = np.zeros(num_periods)
q = np.zeros(num_periods)
d = np.zeros(num_periods)

# Iteración de Bellman
max_iter = 1000
tol = 0.00001
for it in range(max_iter):
    V_new = np.zeros(num_periods)
    for i in range(num_periods):
        y = y_fcst[i]
        # Valor en autarquía
        V_A[i] = u(h(y, d0, d1), sigma) + beta * (psi  * V[i] + (1 - psi ) * V_A[i])

        # Decisión de default
        V_default = V_A[i]
        V_repay = -np.inf
        for j in range(num_periods):
            b_prime = y_fcst[j] - 1 #Media de Y
            c = y - b_prime + q[j] * b_prime
            if c > 0:
                V_repay = max(V_repay, u(c, sigma) + beta * V[j])

        V_new[i] = max(V_repay, V_default)
        d[i] = 1 if V_repay < V_default else 0

        # Precio de los bonos
        default_prob = np.mean(d)
        q[i] = (1 - default_prob) / (1 + r) if b_prime < 0 else 1 / (1 + r)

    if np.max(np.abs(V_new - V)) < tol:
        break
    V = V_new

# Graficar la función de valor contra el tiempo
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_periods), V, label='Función de Valor \(V\)')
plt.xlabel('Período')
plt.ylabel('Función de Valor')
plt.legend()
plt.title('Función de Valor contra el Tiempo (Simulación)')
plt.show()

# Graficar los precios de los bonos contra el ingreso
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_periods), q, label='Precio de los Bonos \(q\)')
plt.xlabel('Período')
plt.ylabel('Precios de los Bonos')
plt.legend()
plt.title('Precios de los Bonos contra el Tiempo')
plt.show()

# Graficar y
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_periods), y_fcst, label='y')
plt.xlabel('Período')
plt.ylabel('y')
plt.legend()
plt.title('Y contra el Tiempo')
plt.show()
