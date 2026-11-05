import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datos = {
    "Edad": [18, 22, 25, 28, 30, 35, 40, 45, 50, 21, 27, 38],
    "Ingresos": [5000, 7000, 9000, 11000, 12000, 15000,
                 18000, 20000, 25000, 6000, 8000, 17000],
    "Compra": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(datos)

print(df.head())
print(df.info())
print(df["Compra"].value_counts())

X = df[["Edad", "Ingresos"]]
y = df["Compra"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

print("Exactitud: ", accuracy_score(y_test, predicciones))
exactitud = accuracy_score(y_test, predicciones)

print("Exactitud:", exactitud)
print("Exactitud:", round(exactitud * 100, 2), "%")

nuevo_cliente = pd.DataFrame({
    "Edad": [32],
    "Ingresos": [14000]
})

resultado = modelo.predict(nuevo_cliente)

if resultado[0] == 1:
    print("El cliente probablemente comprara")
else:
    print("El cliente probablemente no comprara")

nuevosclientes = pd.DataFrame({
    "Edad": [20, 25, 30, 35, 45],
    "Ingresos": [6000, 9000, 12000, 15000, 20000]
})

nuevosclientes["Prediccion"] = modelo.predict(
    nuevosclientes
)

print(nuevosclientes)

colores = nuevosclientes["Prediccion"].map({
    0: "red",
    1: "green"
})

plt.figure(figsize=(10, 6))

plt.scatter(
    nuevosclientes["Edad"],
    nuevosclientes["Ingresos"],
    c=colores,
    s=180,
    edgecolors="black"
)

#etiquetas
for i in range(len(nuevosclientes)):
    resultado = nuevosclientes["Prediccion"].iloc[i]

    etiqueta = "Comprara" if resultado == 1 else "No comprara"

    plt.annotate(
        etiqueta,
        (nuevosclientes["Edad"].iloc[i],
        nuevosclientes["Ingresos"].iloc[i]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.title("Predicciones de compra de clientes")
plt.xlabel("Edad del cliente")
plt.ylabel("Ingresos mensuales ($)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()