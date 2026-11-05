import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos = {
    "HorasEstudio": [1, 2, 3, 4, 5, 6, 7, 8],
    "Calificacion": [55, 60, 65, 70, 75, 80, 85, 90]
}

df = pd.DataFrame(datos)

# variable independiente: horas de estudio
# variable dependiente: calificacion del examen
x = df[["HorasEstudio"]]
y = df["Calificacion"]

modelo = LinearRegression()
modelo.fit(x, y)

print("Pendiente:", round(modelo.coef_[0], 2))
print("Intercepto:", round(modelo.intercept_, 2))

nuevo_dato = pd.DataFrame({"HorasEstudio": [9]})
prediccion = modelo.predict(nuevo_dato)
print("Calificacion estimada para 9 horas:", round(prediccion[0], 2))

plt.scatter(x["HorasEstudio"], y, label="Datos reales")
horas_linea = pd.DataFrame({"HorasEstudio": [1, 9]})
plt.plot(horas_linea["HorasEstudio"], modelo.predict(horas_linea),
         color="orange", label="Linea de regresion")
plt.scatter(9, prediccion[0], color="red", label="Prediccion para 9 horas",
            marker="x", s=100)

plt.xlabel("Horas de estudio")
plt.ylabel("Calificacion")
plt.title("Relacion entre horas de estudio y calificacion")
plt.legend()
plt.grid(True)
plt.show()

# pendiente: por cada hora adicional, la calificacion estimada aumenta 5 puntos.
# intercepto: para 0 horas, el modelo estima 50 puntos.
# prediccion: para 9 horas estima 95 puntos, fuera del rango observado de 1 a 8 horas.
