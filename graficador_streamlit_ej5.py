
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


st.title("Graficador de Ejercicios de Redistribución y el rol del Estado")

    st.header("Parámetros")

    if ejercicio == "Ejercicio 5: Blanca y Nieves":
        total_renta = st.number_input("TOTAL_RENTA (a):", min_value=0.0, value=1.0)
        yB_ini = st.number_input("yB_ini (c):", min_value=0.0, value=0.2)
        yN_ini = st.number_input("yN_ini (c):", min_value=0.0, value=0.8)
        cost_ratio = st.number_input("transfer_cost_ratio:", min_value=0.01, value=2.0)
        crra_sigma = st.number_input("CRRA σ (si aplica):", value=0.5)
        grid_points = st.number_input("Puntos (grid):", min_value=100, max_value=10000, value=1500, step=100)
        utility_model = st.selectbox("Función de utilidad:", ["sqrt", "log1p", "crra"], index=0)

        def u_sqrt(y):
            return np.sqrt(np.maximum(y, 0.0))

        def u_log1p(y):
            return np.log1p(np.maximum(y, 0.0))

        def u_crra(y, sigma):
            y = np.maximum(y, 0.0)
            if np.isclose(sigma, 1.0):
                return np.log(y + 1e-12)
            return (np.power(y, 1.0 - sigma) - 1.0) / (1.0 - sigma)

        def pick_u(model, sigma):
            if model == "sqrt":
                return u_sqrt
            if model == "log1p":
                return u_log1p
            if model == "crra":
                return lambda y: u_crra(y, sigma)
            return u_sqrt

        def frontera_utilidad_sin_distorsion(u, total_renta, grid=1500):
            yB = np.linspace(0.0, total_renta, int(grid))
            yN = total_renta - yB
            return u(yB), u(yN)

        def frontera_utilidad_con_distorsion(u, yB0, yN0, cost_ratio, grid=1500):
            t_min = max(-yB0, -(yN0 / cost_ratio))
            t_max = min(+np.inf, +yN0 / cost_ratio)
            t = np.linspace(t_min, t_max, int(grid))
            yB = yB0 + t
            yN = yN0 - cost_ratio * t
            return u(yB), u(yN)

        def hallar_optimos(uB, uN):
            welfare_util = uB + uN
            welfare_rawls = np.minimum(uB, uN)
            i_u = int(np.argmax(welfare_util))
            i_r = int(np.argmax(welfare_rawls))
            return (uB[i_u], uN[i_u]), (uB[i_r], uN[i_r])

        # Validación y lógica SOLO dentro del bloque Ejercicio 5
        if total_renta < 0:
            st.error("TOTAL_RENTA debe ser no negativa.")
        elif cost_ratio <= 0:
            st.error("transfer_cost_ratio debe ser positiva.")
        elif yB_ini < 0 or yN_ini < 0:
            st.error("yB_ini y yN_ini deben ser no negativas.")
        else:
            u = pick_u(utility_model, crra_sigma)

            # (a) Frontera sin distorsión
            uB_a, uN_a = frontera_utilidad_sin_distorsion(u, total_renta, grid_points)
            # (b) Óptimos en (a)
            (uB_util_a, uN_util_a), (uB_raw_a, uN_raw_a) = hallar_optimos(uB_a, uN_a)
            # (c) Frontera con distorsión
            uB_c, uN_c = frontera_utilidad_con_distorsion(u, yB_ini, yN_ini, cost_ratio, grid_points)

            tabs = st.tabs(["a) Frontera sin distorsión", "b) Óptimos (sin distorsión)", "c) Comparación con distorsión"])

            with tabs[0]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(uB_a, uN_a, linewidth=2)
                m = max(np.max(uB_a), np.max(uN_a))
                ray = np.linspace(0, m, 300)
                ax.plot(ray, ray, linestyle='--')
                ax.set_xlabel("u_B (Blanca)")
                ax.set_ylabel("u_N (Nieves)")
                ax.set_title("Frontera de posibilidades de utilidad (sin distorsión)")
                ax.grid(True)
                st.pyplot(fig)

            with tabs[1]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(uB_a, uN_a, linewidth=2, label="Frontera")
                m = max(np.max(uB_a), np.max(uN_a))
                ray = np.linspace(0, m, 300)
                ax.plot(ray, ray, linestyle='--', label="Rayo de igualdad")
                ax.scatter([uB_util_a], [uN_util_a], marker="o", s=60, label="Óptimo utilitarista")
                ax.scatter([uB_raw_a], [uN_raw_a], marker="s", s=60, label="Óptimo rawlsiano")
                ax.set_xlabel("u_B (Blanca)")
                ax.set_ylabel("u_N (Nieves)")
                ax.set_title("Óptimos (sin distorsión)")
                ax.legend(loc="best")
                ax.grid(True)
                st.pyplot(fig)

            with tabs[2]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(uB_a, uN_a, linewidth=2, label="Frontera (a)")
                ax.plot(uB_c, uN_c, linewidth=2, label="Con distorsión (c)")
                m = max(np.max(uB_a), np.max(uN_a), np.max(uB_c), np.max(uN_c))
                ray = np.linspace(0, m, 300)
                ax.plot(ray, ray, linestyle='--', label="Rayo de igualdad")
                ax.scatter([uB_util_a], [uN_util_a], marker="o", s=60, label="Óptimo utilitarista (a)")
                ax.scatter([uB_raw_a], [uN_raw_a], marker="s", s=60, label="Óptimo rawlsiano (a)")
                ax.set_xlabel("u_B (Blanca)")
                ax.set_ylabel("u_N (Nieves)")
                ax.set_title("Comparación de fronteras y óptimos")
                ax.legend(loc="best")
                ax.grid(True)
                st.pyplot(fig)
            st.pyplot(fig)

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(uB_a, uN_a, linewidth=2, label="Frontera (a)")
            ax.plot(uB_c, uN_c, linewidth=2, label="Con distorsión (c)")
            m = max(np.max(uB_a), np.max(uN_a), np.max(uB_c), np.max(uN_c))
            ray = np.linspace(0, m, 300)
            ax.plot(ray, ray, linestyle='--', label="Rayo de igualdad")
            ax.scatter([uB_util_a], [uN_util_a], marker="o", s=60, label="Óptimo utilitarista (a)")
            ax.scatter([uB_raw_a], [uN_raw_a], marker="s", s=60, label="Óptimo rawlsiano (a)")
            ax.set_xlabel("u_B (Blanca)")
            ax.set_ylabel("u_N (Nieves)")
            ax.set_title("Comparación de fronteras y óptimos")
            ax.legend(loc="best")
            ax.grid(True)
            st.pyplot(fig)

elif ejercicio == "Ejercicio 1: Andrés y Beatriz":
    st.subheader("Ejercicio 1: Andrés y Beatriz")
    st.info("Interfaz y gráficos próximamente disponibles.")

elif ejercicio == "Ejercicio 2: Función de bienestar social":
    st.subheader("Ejercicio 2: Función de bienestar social")
    st.info("Interfaz y gráficos próximamente disponibles.")

elif ejercicio == "Ejercicio 3: Diferencia de ingresos":
    st.subheader("Ejercicio 3: Diferencia de ingresos")
    st.info("Interfaz y gráficos próximamente disponibles.")

elif ejercicio == "Ejercicio 4: Redistribución voluntaria":
    st.subheader("Ejercicio 4: Redistribución voluntaria")
    st.info("Interfaz y gráficos próximamente disponibles.")

def u_sqrt(y):
    return np.sqrt(np.maximum(y, 0.0))

def u_log1p(y):
    return np.log1p(np.maximum(y, 0.0))

def u_crra(y, sigma):
    y = np.maximum(y, 0.0)
    if np.isclose(sigma, 1.0):
        return np.log(y + 1e-12)
    return (np.power(y, 1.0 - sigma) - 1.0) / (1.0 - sigma)

def pick_u(model, sigma):
    if model == "sqrt":
        return u_sqrt
    if model == "log1p":
        return u_log1p
    if model == "crra":
        return lambda y: u_crra(y, sigma)
    return u_sqrt

def frontera_utilidad_sin_distorsion(u, total_renta, grid=1500):
    yB = np.linspace(0.0, total_renta, int(grid))
    yN = total_renta - yB
    return u(yB), u(yN)

def frontera_utilidad_con_distorsion(u, yB0, yN0, cost_ratio, grid=1500):
    t_min = max(-yB0, -(yN0 / cost_ratio))
    t_max = min(+np.inf, +yN0 / cost_ratio)
    t = np.linspace(t_min, t_max, int(grid))
    yB = yB0 + t
    yN = yN0 - cost_ratio * t
    return u(yB), u(yN)

def hallar_optimos(uB, uN):
    welfare_util = uB + uN
    welfare_rawls = np.minimum(uB, uN)
    i_u = int(np.argmax(welfare_util))
    i_r = int(np.argmax(welfare_rawls))
    return (uB[i_u], uN[i_u]), (uB[i_r], uN[i_r])

if total_renta < 0:
    st.error("TOTAL_RENTA debe ser no negativa.")
elif cost_ratio <= 0:
    st.error("transfer_cost_ratio debe ser positiva.")
elif yB_ini < 0 or yN_ini < 0:
    st.error("yB_ini y yN_ini deben ser no negativas.")
else:
    u = pick_u(utility_model, crra_sigma)

    # (a) Frontera sin distorsión
    uB_a, uN_a = frontera_utilidad_sin_distorsion(u, total_renta, grid_points)
    # (b) Óptimos en (a)
    (uB_util_a, uN_util_a), (uB_raw_a, uN_raw_a) = hallar_optimos(uB_a, uN_a)
    # (c) Frontera con distorsión
    uB_c, uN_c = frontera_utilidad_con_distorsion(u, yB_ini, yN_ini, cost_ratio, grid_points)

    tabs = st.tabs(["a) Frontera sin distorsión", "b) Óptimos (sin distorsión)", "c) Comparación con distorsión"])

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(uB_a, uN_a, linewidth=2)
        m = max(np.max(uB_a), np.max(uN_a))
        ray = np.linspace(0, m, 300)
        ax.plot(ray, ray, linestyle='--')
        ax.set_xlabel("u_B (Blanca)")
        ax.set_ylabel("u_N (Nieves)")
        ax.set_title("Frontera de posibilidades de utilidad (sin distorsión)")
        ax.grid(True)
        st.pyplot(fig)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(uB_a, uN_a, linewidth=2, label="Frontera")
        m = max(np.max(uB_a), np.max(uN_a))
        ray = np.linspace(0, m, 300)
        ax.plot(ray, ray, linestyle='--', label="Rayo de igualdad")
        ax.scatter([uB_util_a], [uN_util_a], marker="o", s=60, label="Óptimo utilitarista")
        ax.scatter([uB_raw_a], [uN_raw_a], marker="s", s=60, label="Óptimo rawlsiano")
        ax.set_xlabel("u_B (Blanca)")
        ax.set_ylabel("u_N (Nieves)")
        ax.set_title("Óptimos (sin distorsión)")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)

    with tabs[2]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(uB_a, uN_a, linewidth=2, label="Frontera (a)")
        ax.plot(uB_c, uN_c, linewidth=2, label="Con distorsión (c)")
        m = max(np.max(uB_a), np.max(uN_a), np.max(uB_c), np.max(uN_c))
        ray = np.linspace(0, m, 300)
        ax.plot(ray, ray, linestyle='--', label="Rayo de igualdad")
        ax.scatter([uB_util_a], [uN_util_a], marker="o", s=60, label="Óptimo utilitarista (a)")
        ax.scatter([uB_raw_a], [uN_raw_a], marker="s", s=60, label="Óptimo rawlsiano (a)")
        ax.set_xlabel("u_B (Blanca)")
        ax.set_ylabel("u_N (Nieves)")
        ax.set_title("Comparación de fronteras y óptimos")
        ax.legend(loc="best")
        ax.grid(True)
        st.pyplot(fig)
