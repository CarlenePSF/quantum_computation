import numpy as np
import matplotlib.pyplot as plt

# Defina a EDO da queda livre: dv/dt = g - (k/m) * v
def ode(x, y):
    return x


# Método de Euler para resolver a EDO
def euler_method(ode, t0, v0, h, num_steps, g, k, m):
    t_values = [t0]
    v_values = [v0]

    for _ in range(num_steps):
        t = t_values[-1]
        v = v_values[-1]
        dv_dt = ode(t, v, g, k, m)
        v_new = v + h * dv_dt
        t_values.append(t + h)
        v_values.append(v_new)

    return np.array(t_values), np.array(v_values)

# Campo de direções para a EDO da posição
def position_direction_field(ode, t_range, y_range, g, k, m):
    t, y = np.meshgrid(t_range, y_range)
    dy_dt = ode(t, y, g, k, m)
    return t, y, dy_dt

# Parâmetros iniciais
t0 = 0
y0 = 20  # posição inicial (m)
v0 = 0  # velocidade inicial (m/s)
h = 0.1  # Tamanho do passo
num_steps = 100  # Número de passos
g = 9.8  # aceleração devida à gravidade (m/s^2)
k = 0.1  # coeficiente de resistência do ar
m = 1.0  # massa do objeto (kg)

# Aplicar o método de Euler para a EDO da posição
t_values, y_values = euler_method(ode, t0, y0, h, num_steps, g, k, m)

# Calcular a velocidade integrando a aceleração
v_values = np.gradient(y_values, t_values)

# Campo de direções para a EDO da posição
t_range = np.linspace(t0, max(t_values), 20)
y_range = np.linspace(min(y_values), max(y_values), 20)
t_direction, y_direction, dy_dt = position_direction_field(ode, t_range, y_range, g, k, m)

# Plotar a solução da posição, velocidade e campo de direções
plt.figure(figsize=(12, 6))

# Posição
plt.subplot(2, 2, 1)
plt.plot(t_values,
         y_values, label='Posição - Método de Euler', color='blue')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Solução da EDO da Queda Livre - Posição')

# Velocidade
plt.subplot(2, 2, 2)
plt.plot(t_values, v_values, label='Velocidade - Método de Euler', color='green')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Solução da EDO da Queda Livre - Velocidade')

# Campo de direções
plt.subplot(2, 2, 3)
plt.quiver(t_direction, y_direction,
           np.ones_like(dy_dt), dy_dt,
           scale=50, color='red', headwidth=0)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Campo de Direções da EDO da Queda Livre - Posição')

plt.tight_layout()
plt.show()
