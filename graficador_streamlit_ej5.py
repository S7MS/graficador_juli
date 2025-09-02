import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")

st.title("Resolución de Ejercicios de Economía Pública")

# Selector de ejercicio en la barra lateral
ejercicio = st.sidebar.selectbox(
    "Elige el ejercicio que quieres visualizar",
    [
        "Ejercicio 1: Andrés y Beatriz",
        "Ejercicio 2: Función de bienestar social",
        "Ejercicio 3: Diferencia de ingresos",
        "Ejercicio 4: Redistribución voluntaria",
        "Ejercicio 5: Frontera de Utilidad (Blanca y Nieves)"
    ]
)

# --- EJERCICIO 1 ---
if ejercicio == "Ejercicio 1: Andrés y Beatriz":
    st.header("Ejercicio 1: Andrés y Beatriz")
    st.markdown("""
    Se analiza la frontera de posibilidades de utilidad (FPU) entre Andrés y Beatriz, dadas sus funciones de utilidad y una restricción de producción donde el ingreso total es 1.
    """)
    
    st.latex("u(y) = 0.4y")
    st.latex("v(x,y) = 0.4x + [0.2 + 0.8y – y^2]")
    st.latex("x + y = 1")

    # Cálculos
    y = np.linspace(0, 1, 200)
    x = 1 - y
    u = 0.4 * y
    v = 0.4 * x + (0.2 + 0.8 * y - y**2)

    # Puntos óptimos
    # Óptimo para Andrés: maximiza u(y), lo que ocurre cuando 'y' es máximo (y=1)
    y_andres_opt = 1.0
    x_andres_opt = 0.0
    u_andres_opt = 0.4 * y_andres_opt
    v_andres_opt = 0.4 * x_andres_opt + (0.2 + 0.8 * y_andres_opt - y_andres_opt**2)

    # Óptimo para Beatriz: maximiza v(x,y) = v(1-y, y) = 0.4(1-y) + 0.2 + 0.8y - y^2 = 0.6 + 0.4y - y^2
    # Derivando respecto a y: 0.4 - 2y = 0 => y = 0.2
    y_beatriz_opt = 0.2
    x_beatriz_opt = 0.8
    u_beatriz_opt = 0.4 * y_beatriz_opt
    v_beatriz_opt = 0.4 * x_beatriz_opt + (0.2 + 0.8 * y_beatriz_opt - y_beatriz_opt**2)

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(u, v, label='Frontera de Posibilidades de Utilidad (FPU)')
    ax.scatter(u_andres_opt, v_andres_opt, color='blue', zorder=5, label=f'Óptimo para Andrés (y=1)')
    ax.scatter(u_beatriz_opt, v_beatriz_opt, color='red', zorder=5, label=f'Óptimo para Beatriz (y=0.2)')
    ax.set_xlabel("Utilidad de Andrés (u)")
    ax.set_ylabel("Utilidad de Beatriz (v)")
    ax.set_title("Ejercicio 1: Frontera de Posibilidades de Utilidad")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Análisis de los Puntos Óptimos")
    st.markdown(f"""
    - **Distribución óptima para Andrés**: Se logra cuando su ingreso `y` es máximo (`y=1`, `x=0`).
      - Utilidad de Andrés: $u = 0.4 \\times 1 = {u_andres_opt:.2f}$
      - Utilidad de Beatriz: $v = 0.4 \\times 0 + (0.2 + 0.8 \\times 1 - 1^2) = {v_andres_opt:.2f}$
    - **Distribución óptima para Beatriz**: Se logra maximizando su utilidad.
      - Max $v(y) = 0.6 + 0.4y - y^2 \\implies \\frac{{dv}}{{dy}} = 0.4 - 2y = 0 \\implies y=0.2$. Con `y=0.2`, `x=0.8`.
      - Utilidad de Andrés: $u = 0.4 \\times 0.2 = {u_beatriz_opt:.2f}$
      - Utilidad de Beatriz: $v = 0.4 \\times 0.8 + (0.2 + 0.8 \\times 0.2 - 0.2^2) = {v_beatriz_opt:.2f}$
    """)

# --- EJERCICIO 2 ---
elif ejercicio == "Ejercicio 2: Función de bienestar social":
    st.header("Ejercicio 2: Función de Bienestar Social")
    st.markdown("""
    Utilizando la FPU del ejercicio anterior, se busca la distribución que maximiza el bienestar social bajo dos funciones distintas: una utilitarista y una ponderada.
    """)
    alpha = st.sidebar.slider("Parámetro de ponderación (α)", 0.0, 2.0, 2/3, 0.01)

    # FPU del ejercicio 1
    u = np.linspace(0, 0.4, 200)
    v = 0.6 + u - 6.25 * u**2

    # a. Bienestar Utilitarista (W = u + v)
    W_util = u + v
    idx_util = np.argmax(W_util)
    u_opt_util, v_opt_util = u[idx_util], v[idx_util]

    # b. Bienestar Ponderado (W = u + alpha*v)
    W_pond = u + alpha * v
    idx_pond = np.argmax(W_pond)
    u_opt_pond, v_opt_pond = u[idx_pond], v[idx_pond]

    # Gráfico
    fig, ax = plt.subplots()
    ax.plot(u, v, label='FPU')
    # Curvas de indiferencia social
    u_plot = np.linspace(0, 0.5, 100)
    ax.plot(u_plot, W_util[idx_util] - u_plot, 'r--', label=f'Indiferencia Utilitarista (W={W_util[idx_util]:.2f})')
    ax.plot(u_plot, (W_pond[idx_pond] - u_plot) / alpha, 'g--', label=f'Indiferencia Ponderada (W={W_pond[idx_pond]:.2f})')
    ax.scatter(u_opt_util, v_opt_util, color='red', zorder=5, label='Óptimo Utilitarista')
    ax.scatter(u_opt_pond, v_opt_pond, color='green', zorder=5, label=f'Óptimo Ponderado (α={alpha:.2f})')
    ax.set_xlabel("Utilidad de Andrés (u)")
    ax.set_ylabel("Utilidad de Beatriz (v)")
    ax.set_title("Ejercicio 2: Maximización del Bienestar Social")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.7)
    st.pyplot(fig)

    st.subheader("Análisis de los Puntos Óptimos")
    st.markdown(f"""
    - **a. Óptimo Utilitarista ($W=u+v$)**:
      - Max $W(u) = u + (0.6 + u - 6.25u^2) = 0.6 + 2u - 6.25u^2$.
      - $\\frac{{dW}}{{du}} = 2 - 12.5u = 0 \\implies u = 0.16$.
      - **Resultado**: $u={u_opt_util:.2f}, v={v_opt_util:.2f}$. El ingreso de Andrés es $y=u/0.4 = 0.4$, y el de Beatriz es $x=0.6$. No es igualitario.
    - **b. Óptimo Ponderado ($W=u+\\alpha v$) con $\\alpha={alpha:.2f}$**:
      - Max $W(u) = u + \\alpha(0.6 + u - 6.25u^2)$.
      - $\\frac{{dW}}{{du}} = 1 + \\alpha(1 - 12.5u) = 0 \\implies u = \\frac{{1+\\alpha}}{{12.5\\alpha}}$.
      - **Resultado**: $u={u_opt_pond:.2f}, v={v_opt_pond:.2f}$. El ingreso de Andrés es $y=u/0.4 = {u_opt_pond/0.4:.2f}$, y el de Beatriz es $x={1-u_opt_pond/0.4:.2f}$.
    """)

# --- EJERCICIO 3 ---
elif ejercicio == "Ejercicio 3: Diferencia de ingresos":
    st.header("Ejercicio 3: Diferencia de Ingresos")
    st.markdown("""
    Se analiza una función de bienestar social que penaliza la desigualdad: $W(x,y) = 0.5(x+y) - |x-y|$. Asumiendo $y < x$, esto se simplifica a $W(x,y) = 0.5(x+y) - (x-y) = 1.5y - 0.5x$.
    """)
    x_income = st.sidebar.slider("Ingreso de X (más rico)", 1.0, 100.0, 50.0, 0.5)
    y_income = st.sidebar.slider("Ingreso de Y (más pobre)", 1.0, x_income, 30.0, 0.5)

    def calculate_W(x, y):
        return 0.5 * (x + y) - abs(x - y)

    W_base = calculate_W(x_income, y_income)
    st.metric("Bienestar Social Actual (W)", f"{W_base:.2f}")

    st.subheader("Análisis de Cambios en el Ingreso")
    # a. Cambios en el ingreso
    W_x_up = calculate_W(x_income + 1, y_income)
    W_y_up = calculate_W(x_income, y_income + 1)
    W_both_up = calculate_W(x_income + 0.5, y_income + 0.5)
    st.markdown(f"- Si el ingreso de X (rico) aumenta en 1€: $W$ cambia en **{W_x_up - W_base:.2f}**.")
    st.markdown(f"- Si el ingreso de Y (pobre) aumenta en 1€: $W$ cambia en **{W_y_up - W_base:.2f}**.")
    st.markdown(f"- Si ambos ingresos aumentan en 0.5€: $W$ cambia en **{W_both_up - W_base:.2f}**.")
    st.info("Interpretación: El bienestar social es más sensible a los aumentos de renta de la persona más pobre.")

    # b. Transferencia
    st.subheader("Análisis de Transferencia")
    W_transfer = calculate_W(x_income - 1, y_income + 1)
    st.markdown(f"- Si X transfiere 1€ a Y: $W$ cambia en **{W_transfer - W_base:.2f}**.")
    st.info("Interpretación: La redistribución del rico al pobre aumenta drásticamente el bienestar social con esta función.")

    # c. Reparto de 1€ extra
    st.subheader("Reparto de 1€ Adicional")
    st.markdown("""
    ¿Cómo repartir 1€ extra para que el bienestar no cambie? $W_{nuevo} = W_{base}$.
    Sea $\delta$ la porción para X, y $(1-\delta)$ la porción para Y.
    $W(x+\delta, y+1-\delta) = W(x,y)$.
    $0.5(x+\delta+y+1-\delta) - (x+\delta - (y+1-\delta)) = 0.5(x+y) - (x-y)$
    $0.5(x+y+1) - (x-y-1+2\delta) = 0.5(x+y) - (x-y)$
    $0.5 - (-1+2\delta) = 0 \implies 1.5 = 2\delta \implies \delta = 0.75$.
    """)
    st.success("Para que el bienestar no cambie, X (el rico) debería recibir 0.75€ y Y (el pobre) 0.25€ del euro adicional.")

# --- EJERCICIO 4 ---
elif ejercicio == "Ejercicio 4: Redistribución voluntaria":
    st.header("Ejercicio 4: Redistribución Voluntaria")
    st.markdown("""
    Se compara la transferencia caritativa de un matrimonio a una persona necesitada, con la que realizan tras divorciarse, analizando el problema del *free-rider*.
    """)
    y_medina_total = 840
    y_pablo = 60
    
    st.subheader("a. Decisión conjunta (Matrimonio)")
    # u = ln(cM) + 0.5*ln(cP) = ln(yM - TM) + 0.5*ln(yP + TM)
    # Derivando respecto a TM e igualando a 0: -1/(yM-TM) + 0.5/(yP+TM) = 0
    # 0.5(yM-TM) = yP+TM => 0.5*yM - 0.5*TM = yP + TM => 0.5*yM - yP = 1.5*TM
    TM = (0.5 * y_medina_total - y_pablo) / 1.5
    cM = y_medina_total - TM
    cP_a = y_pablo + TM
    st.metric("Transferencia óptima del matrimonio (TM)", f"{TM:.2f} €")
    st.markdown(f"Consumo Medina: {cM:.2f}€, Consumo Pablo: {cP_a:.2f}€")

    st.subheader("b. Decisión individual (Divorciados)")
    y_alba = 420
    y_benjamin = 420
    # Alba maximiza uA = ln(yA-tA) + 0.5*ln(yP+tA+tB) tomando tB como dado
    # dU/dtA = -1/(yA-tA) + 0.5/(yP+tA+tB) = 0 => 2(yP+tA+tB) = yA-tA (Curva de reacción de Alba)
    # Simétricamente, para Benjamín: 2(yP+tA+tB) = yB-tB
    # Como yA=yB, en equilibrio tA=tB=t.
    # 2(yP+2t) = yA-t => 2yP+4t = yA-t => 5t = yA-2yP
    t = (y_alba - 2 * y_pablo) / 5
    tA = t
    tB = t
    TD = tA + tB
    cP_b = y_pablo + TD
    st.metric("Transferencia óptima de Alba (tA)", f"{tA:.2f} €")
    st.metric("Transferencia óptima de Benjamín (tB)", f"{tB:.2f} €")
    st.metric("Transferencia total de divorciados (TD)", f"{TD:.2f} €")
    st.markdown(f"Consumo Pablo: {cP_b:.2f}€")

    st.subheader("c. Comparación y Eficiencia")
    st.markdown(f"El total transferido por el matrimonio ($T_M={TM:.2f}€$) es mayor que el total transferido por los divorciados ($T_D={TD:.2f}€$).")
    st.warning("""
    **Interpretación (Problema del Free-Rider)**: Cuando deciden por separado, tanto Alba como Benjamín tienen un incentivo a contribuir menos, esperando que el otro asuma parte de la "carga" de ayudar a Pablo. Cada uno no internaliza el beneficio completo que su contribución genera para el otro (a través de la utilidad que ambos obtienen del bienestar de Pablo). Esto lleva a una provisión ineficientemente baja del "bien público" (la ayuda a Pablo) en comparación con la situación en la que actúan conjuntamente. La redistribución voluntaria individual no es eficiente.
    """)

# --- EJERCICIO 5 ---
elif ejercicio == "Ejercicio 5: Frontera de Utilidad (Blanca y Nieves)":
    st.header("Ejercicio 5: Frontera de Utilidad con Datos Tabulares")
    st.markdown("""
    Se analiza la FPU a partir de datos de utilidad discretos, y el efecto de una redistribución con costes (distorsión).
    """)
    # Datos del problema
    renta_puntos = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    utilidad_puntos = np.array([0, 10, 19, 27, 34, 40, 45, 49, 52, 54, 55])
    
    # Creamos una función de utilidad continua por interpolación
    u_func = interp1d(renta_puntos, utilidad_puntos, kind='cubic')
    
    renta_total = 1.0
    renta_continua = np.linspace(0, renta_total, 200)

    # a. FPU sin distorsión
    c_blanca_sin = renta_continua
    c_nieves_sin = renta_total - c_blanca_sin
    u_blanca_sin, u_nieves_sin = u_func(c_blanca_sin), u_func(c_nieves_sin)

    # b. Óptimos sin distorsión
    # Utilitarista: max(u1+u2)
    idx_util_sin = np.argmax(u_blanca_sin + u_nieves_sin)
    opt_util_sin = (u_blanca_sin[idx_util_sin], u_nieves_sin[idx_util_sin])
    # Rawlsiano: max(min(u1,u2)) -> donde u1=u2
    idx_rawls_sin = np.argmin(np.abs(u_blanca_sin - u_nieves_sin))
    opt_rawls_sin = (u_blanca_sin[idx_rawls_sin], u_nieves_sin[idx_rawls_sin])

    # c. FPU con distorsión
    # Renta inicial: Blanca 0.2, Nieves 0.8.
    # Por cada 0.1 que recibe Blanca, Nieves pierde 0.2. Coste=0.1.
    # c_blanca = 0.2 + t, c_nieves = 0.8 - 2t. (donde t es la transferencia neta a Blanca)
    t_max = 0.4 # Nieves se queda con 0
    t_vals = np.linspace(0, t_max, 100)
    c_blanca_con = 0.2 + t_vals
    c_nieves_con = 0.8 - 2 * t_vals
    u_blanca_con, u_nieves_con = u_func(c_blanca_con), u_func(c_nieves_con)
    
    # Óptimos con distorsión
    idx_util_con = np.argmax(u_blanca_con + u_nieves_con)
    opt_util_con = (u_blanca_con[idx_util_con], u_nieves_con[idx_util_con])
    idx_rawls_con = np.argmin(np.abs(u_blanca_con - u_nieves_con))
    opt_rawls_con = (u_blanca_con[idx_rawls_con], u_nieves_con[idx_rawls_con])

    # Gráfico
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Sin distorsión
    ax[0].plot(u_blanca_sin, u_nieves_sin, label='FPU sin distorsión')
    ax[0].scatter(*opt_util_sin, color='red', zorder=5, label=f"Óptimo Utilitarista")
    ax[0].scatter(*opt_rawls_sin, color='green', zorder=5, label=f"Óptimo Rawlsiano")
    ax[0].set_title("a) y b) Sin Distorsión")
    ax[0].set_xlabel("Utilidad de Blanca"); ax[0].set_ylabel("Utilidad de Nieves"); ax[0].grid(True); ax[0].legend()

    # Con distorsión
    ax[1].plot(u_blanca_con, u_nieves_con, label='FPU con distorsión', color='orange')
    ax[1].scatter(*opt_util_con, color='red', zorder=5, label=f"Óptimo Utilitarista")
    ax[1].scatter(*opt_rawls_con, color='green', zorder=5, label=f"Óptimo Rawlsiano")
    ax[1].set_title("c) Con Distorsión")
    ax[1].set_xlabel("Utilidad de Blanca"); ax[1].grid(True); ax[1].legend()
    st.pyplot(fig)

    st.subheader("Análisis de los Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sin Distorsión**")
        st.markdown(f"- **Óptimo Utilitarista**: U(Blanca)={opt_util_sin[0]:.1f}, U(Nieves)={opt_util_sin[1]:.1f}. Se logra con una distribución de renta igualitaria (0.5M para cada una).")
        st.markdown(f"- **Óptimo Rawlsiano**: U(Blanca)={opt_rawls_sin[0]:.1f}, U(Nieves)={opt_rawls_sin[1]:.1f}. También se logra con una distribución igualitaria.")
    with col2:
        st.markdown("**Con Distorsión**")
        st.markdown(f"- **Óptimo Utilitarista**: U(Blanca)={opt_util_con[0]:.1f}, U(Nieves)={opt_util_con[1]:.1f}. Se logra transfiriendo {t_vals[idx_util_con]:.2f}M a Blanca.")
        st.markdown(f"- **Óptimo Rawlsiano**: U(Blanca)={opt_rawls_con[0]:.1f}, U(Nieves)={opt_rawls_con[1]:.1f}. Se logra transfiriendo {t_vals[idx_rawls_con]:.2f}M a Blanca.")
    st.info("Interpretación: El coste de la redistribución (la distorsión) 'encoge' la frontera de posibilidades de utilidad. Esto puede llevar a que la política redistributiva óptima sea menos ambiciosa que en un mundo sin costes, ya que parte del bienestar se 'pierde' en el proceso.")
