import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as S
from sklearn.linear_model import LinearRegression

datos = {
    "publicidad": [1000, 2000, 3000, 4000, 5000],
    "ventas": [5000, 7000, 9000, 11000, 13000]
}

df = pd.DataFrame(datos)

x = df[["publicidad"]]
y = df["ventas"]

modelo = LinearRegression()
modelo.fit(x, y)

nuevo_dato = pd.DataFrame({"publicidad": [6000]})
prediccion = modelo.predict(nuevo_dato)
    
print("Ventas estimadas:", prediccion[0])
print("Pendientes:", round(modelo.coef_[0], 2))
print("Intercepto:", round(modelo.intercept_, 2))   

plt.scatter(x["publicidad"], y, label="Datos reales")
plt.plot(x["publicidad"], modelo.predict(x), label="Línea de regresión")

plt.scatter(6000, prediccion[0], color="red", label="Predicción", maker="x", s=100)

plt.xlabel("Inversion en publicidad")
plt.ylabel("Ventas")
plt.title("Regresión lineal ")
plt.legend()
plt.grid(True)
plt.show()