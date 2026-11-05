import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# exploracion del dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)
pd.set_option("display.max_columns", None)

print("\nPrimeros 10 registros:")
print(df.head(10))
print("\nDimensiones:", df.shape)
print("\nColumnas:", df.columns)
df.info()
print("\nEstadisticas descriptivas:")
print(df[["horsepower", "mpg"]].describe())
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nRegistros duplicados:", df.duplicated().sum())

# limpieza de datos necesarios para el modelo
df = df.drop_duplicates().copy()
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])
df = df[(df["horsepower"] > 0) & (df["mpg"] > 0)]
print("\nRegistros despues de la limpieza:", df.shape[0])

# x: potencia en hp; y: rendimiento en millas por galon (mpg)
x = df[["horsepower"]]
y = df["mpg"]

modelo = LinearRegression()
modelo.fit(x, y)

print("\nPendiente:", round(modelo.coef_[0], 4))
print("Intercepto:", round(modelo.intercept_, 4))

nuevos_datos = pd.DataFrame({"horsepower": [75, 100, 125, 150, 175]})
predicciones = modelo.predict(nuevos_datos)
resultados = nuevos_datos.copy()
resultados["Rendimiento estimado (mpg)"] = predicciones.round(2)
print("\nPredicciones:")
print(resultados.to_string(index=False))

plt.scatter(x["horsepower"], y, alpha=0.5, label="Datos reales")
potencias_linea = pd.DataFrame({"horsepower": [x["horsepower"].min(), x["horsepower"].max()]})
plt.plot(potencias_linea["horsepower"], modelo.predict(potencias_linea),
         color="orange", label="Linea de regresion")
plt.scatter(nuevos_datos["horsepower"], predicciones, color="red",
            marker="x", s=100, label="Predicciones")
for hp, mpg in zip(nuevos_datos["horsepower"], predicciones):
    plt.annotate(f"{hp} HP: {mpg:.2f} mpg", (hp, mpg),
                 xytext=(5, 8), textcoords="offset points", fontsize=8)

plt.xlabel("Potencia (HP)")
plt.ylabel("Rendimiento (millas por galon, mpg)")
plt.title("Regresion lineal: potencia y rendimiento de combustible")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
