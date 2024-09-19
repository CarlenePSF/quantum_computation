import numpy as np
import matplotlib.pyplot as plt


def ode(x, y):
    return x/y


# Método de Euler-Cromer para resolver a EDO
def euler_cromer_method(ode, x0, y0, h, num_steps):
    x_values = [x0]
    y_values = [y0]

    for _ in range(num_steps):
        x = x_values[-1]
        y = y_values[-1]
        dy_dx = ode(x, y)
        y_new = y + h * dy_dx
        x_new = x + h
        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(x_values), np.array(y_values)

# Parâmetros iniciais
x0 = 0.01  # Valor inicial de x
y0 = 0.01  # Valor inicial de y
h = 0.1   # Tamanho do passo
num_steps = 50  # Número de passos

# Aplicar o método de Euler-Cromer
x_values, y_values = euler_cromer_method(
    ode, x0, y0, h, num_steps)

# Plotar a solução
plt.plot(x_values, y_values,
         label='Método de Euler-Cromer')
plt.xlabel('x')
plt.ylabel('y')


# Defina a grade de valores para x e y
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)

# Crie uma grade de direções usando a função ode
x_grid, y_grid = np.meshgrid(x, y)
dx = 1  # Passo para x
dy = dx * ode(x_grid, y_grid)

# Normalize as direções para que o comprimento dos vetores seja 1
length = np.sqrt(dx**2 + dy**2)
dx /= length
dy /= length

# Crie o campo de direção usando quiver plot
plt.quiver(x_grid, y_grid, dx, dy, scale=30,
           color='blue', headwidth=3)

plt.plot(x_values, y_values)

# Adicione rótulos e título ao gráfico
plt.xlabel('x')
plt.ylabel('y')

# Exiba o gráfico
plt.show()
