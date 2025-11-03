import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- 1. Core Engineering Functions ---

def antoine_pressure(T, A, B, C):
    """
    Calculates saturation pressure (in bar) using Antoine's equation.
    T must be in Celsius.
    log10(P_sat) = A - (B / (T + C))
    """
    return 10**(A - (B / (T + C)))

def calculate_Pxy(T_celsius, antoine_A, antoine_B):
    """
    Calculates P-x-y data for a fixed Temperature.
    antoine_A and antoine_B are lists/tuples: (comp1, comp2)
    """
    P1_sat = antoine_pressure(T_celsius, antoine_A[0], antoine_A[1], antoine_A[2])
    P2_sat = antoine_pressure(T_celsius, antoine_B[0], antoine_B[1], antoine_B[2])
    
    x1 = np.linspace(0, 1, 100)
    P_total = x1 * P1_sat + (1 - x1) * P2_sat
    
    with np.errstate(invalid='ignore'):
        y1 = (x1 * P1_sat) / P_total
    y1[P_total == 0] = x1[P_total == 0] 
    
    return x1, y1, P_total, P1_sat, P2_sat

def calculate_Txy(P_total_bar, antoine_A, antoine_B):
    """
    Calculates T-x-y data for a fixed Pressure.
    This also returns the x-y equilibrium data for McCabe-Thiele.
    """
    x1 = np.linspace(0, 1, 100)
    T_bubble = []
    y1_at_bubble = []
    
    T1_boil = (antoine_A[1] / (antoine_A[0] - np.log10(P_total_bar))) - antoine_A[2]
    T2_boil = (antoine_B[1] / (antoine_B[0] - np.log10(P_total_bar))) - antoine_B[2]
    
    for x in x1:
        def bubble_point_equation(T):
            P1_sat = antoine_pressure(T, antoine_A[0], antoine_A[1], antoine_A[2])
            P2_sat = antoine_pressure(T, antoine_B[0], antoine_B[1], antoine_B[2])
            return x * P1_sat + (1 - x) * P2_sat - P_total_bar
        
        T_guess = x * T1_boil + (1 - x) * T2_boil
        T_solution = fsolve(bubble_point_equation, T_guess)[0]
        
        T_bubble.append(T_solution)
        
        P1_sat_at_T = antoine_pressure(T_solution, antoine_A[0], antoine_A[1], antoine_A[2])
        y1 = (x * P1_sat_at_T) / P_total_bar
        y1_at_bubble.append(y1)
        
    # Return equilibrium x-y data as well
    return x1, np.array(y1_at_bubble), np.array(T_bubble), T1_boil, T2_boil

# --- NEW FUNCTION for McCabe-Thiele ---
def calculate_mccabe_thiele(eq_x, eq_y, xF, xD, xB, R, q):
    """
    Calculates the number of theoretical stages using McCabe-Thiele method.
    """
    
    # 1. Rectifying Operating Line (ROL)
    # y = (R / (R + 1)) * x + (xD / (R + 1))
    def rol(x):
        return (R / (R + 1)) * x + (xD / (R + 1))

    # 2. Find intersection of q-line and ROL
    # q-line equation: y = (q / (q - 1)) * x - (xF / (q - 1))
    # Handle special case for q=1 (saturated liquid)
    if q == 1.0:
        x_intersect = xF
        y_intersect = rol(xF)
    # Handle special case for q=0 (saturated vapor)
    elif q == 0.0:
        x_intersect = xF
        y_intersect = xF # y=x line
        # We need to find the intersection with ROL
        # xF = (R / (R + 1)) * x_intersect + (xD / (R + 1))
        # xF * (R+1) = R*x_intersect + xD
        # x_intersect = (xF*(R+1) - xD) / R
        x_intersect = (xF * (R + 1) - xD) / R
        y_intersect = xF # y = xF for q=0
    else:
        # Solve for intersection
        m_q = q / (q - 1)
        c_q = -xF / (q - 1)
        m_rol = R / (R + 1)
        c_rol = xD / (R + 1)
        
        # m_q*x + c_q = m_rol*x + c_rol
        x_intersect = (c_rol - c_q) / (m_q - m_rol)
        y_intersect = m_rol * x_intersect + c_rol

    # 3. Stripping Operating Line (SOL)
    # Passes through (xB, xB) and (x_intersect, y_intersect)
    m_sol = (y_intersect - xB) / (x_intersect - xB)
    def sol(x):
        return m_sol * (x - xB) + xB
    
    # 4. Step off stages
    stages = 0
    x_step = [xD]
    y_step = [xD]
    
    x_current = xD
    y_current = xD
    
    # Ensure eq_y is monotonically increasing for interpolation
    sort_idx = np.argsort(eq_y)
    eq_y_sorted = eq_y[sort_idx]
    eq_x_sorted = eq_x[sort_idx]
    
    while x_current > xB:
        stages += 1
        
        # Step 1: Horizontal line from op line to equilibrium curve
        # Find x_eq for the current y_current
        # We must interpolate x from y
        y_interp = np.clip(y_current, eq_y_sorted.min(), eq_y_sorted.max())
        x_eq = np.interp(y_interp, eq_y_sorted, eq_x_sorted)
        
        x_step.append(x_eq)
        y_step.append(y_current)
        
        if x_eq <= xB:
            break
            
        # Step 2: Vertical line from equilibrium curve to operating line
        # Use ROL or SOL depending on which side of the feed we are on
        if x_eq > x_intersect:
            y_op = rol(x_eq)
        else:
            y_op = sol(x_eq)
        
        x_step.append(x_eq)
        y_step.append(y_op)
        
        x_current = x_eq
        y_current = y_op
        
    return stages, x_step, y_step, rol, sol, x_intersect, y_intersect

# --- 2. Plotting Functions ---

def plot_Pxy(x1, y1, P_total, P1_sat, P2_sat, T_celsius, comp_names):
    """Creates the P-x-y matplotlib figure."""
    fig, ax = plt.subplots()
    ax.plot(x1, P_total, label='Bubble Line (P vs x1)')
    ax.plot(y1, P_total, label='Dew Line (P vs y1)')
    
    ax.text(1.02, P1_sat, f'$P_1^{{sat}}$={P1_sat:.2f} bar')
    ax.text(-0.15, P2_sat, f'$P_2^{{sat}}$={P2_sat:.2f} bar')
    
    ax.set_xlabel(f'Mole Fraction {comp_names[0]} ($x_1, y_1$)')
    ax.set_ylabel('Total Pressure (bar)')
    ax.set_title(f'P-x-y Diagram at {T_celsius}°C')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 1)
    return fig

def plot_Txy(x1, y1, T_bubble, T1_boil, T2_boil, P_total_bar, comp_names):
    """Creates the T-x-y matplotlib figure."""
    fig, ax = plt.subplots()
    ax.plot(x1, T_bubble, label='Bubble Line (T vs x1)')
    ax.plot(y1, T_bubble, label='Dew Line (T vs y1)')

    ax.text(1.02, T1_boil, f'$T_1^{{boil}}$={T1_boil:.2f}°C')
    ax.text(-0.15, T2_boil, f'$T_2^{{boil}}$={T2_boil:.2f}°C')

    ax.set_xlabel(f'Mole Fraction {comp_names[0]} ($x_1, y_1$)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'T-x-y Diagram at {P_total_bar} bar')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 1)
    return fig

# --- NEW Plotting Function for McCabe-Thiele ---
def plot_mccabe_thiele(eq_x, eq_y, xF, xD, xB, rol, sol, x_int, y_int, x_step, y_step, stages):
    """Creates the McCabe-Thiele plot."""
    fig, ax = plt.subplots()
    
    # 1. y=x line
    ax.plot([0, 1], [0, 1], 'k', label='y=x')
    
    # 2. Equilibrium curve
    ax.plot(eq_x, eq_y, 'b', label='Equilibrium Curve')
    
    # 3. Operating Lines
    x_rol = np.linspace(x_int, xD, 20)
    ax.plot(x_rol, rol(x_rol), 'g--', label='Rectifying Op. Line')
    
    x_sol = np.linspace(xB, x_int, 20)
    ax.plot(x_sol, sol(x_sol), 'r--', label='Stripping Op. Line')
    
    # 4. q-line
    ax.plot([xF, x_int], [xF, y_int], 'm-.', label='q-line')
    
    # 5. Composition lines
    ax.vlines([xB, xF, xD], 0, [xB, xF, xD], colors='gray', linestyles='dotted')
    ax.text(xB, 0.01, f'$x_B$={xB}', ha='center')
    ax.text(xF, 0.01, f'$x_F$={xF}', ha='center')
    ax.text(xD, 0.01, f'$x_D$={xD}', ha='center')
    
    # 6. Steps
    ax.plot(x_step, y_step, 'o-', color='orange', markersize=2, label=f'Stages')
    
    ax.set_xlabel('Liquid Mole Fraction (x)')
    ax.set_ylabel('Vapor Mole Fraction (y)')
    ax.set_title(f'McCabe-Thiele Plot')
    ax.legend(loc='best', fontsize='small')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig, stages

# --- 3. Streamlit Web App Interface ---

st.set_page_config(page_title="VLE & Distillation Calculator", layout="wide") # --- MODIFIED ---
st.title("VLE & McCabe-Thiele Distillation Calculator") # --- MODIFIED ---

# --- Component Selection ---
st.sidebar.header("Select Components")
COMPONENT_DB = {
    "Benzene": [4.20218, 1312.21, 230.563],
    "Toluene": [4.2183, 1428.6, 228.34],
    "Water": [4.6543, 1435.26, 211.516],
    "Ethanol": [4.92531, 1432.02, 200.787],
    "Methanol": [4.95478, 1421.13, 211.565],
    "Acetone": [4.42448, 1312.25, 241.95]
}

comp1_name = st.sidebar.selectbox("Component 1 (More Volatile)", list(COMPONENT_DB.keys()), index=0)
comp2_name = st.sidebar.selectbox("Component 2 (Less Volatile)", list(COMPONENT_DB.keys()), index=1)

# --- NEW: McCabe-Thiele Inputs ---
st.sidebar.header("Distillation Inputs")
xF = st.sidebar.slider("Feed Mole Fraction ($x_F$)", 0.01, 0.99, 0.5)
xD = st.sidebar.slider("Distillate Mole Fraction ($x_D$)", 0.02, 1.0, 0.95)
xB = st.sidebar.slider("Bottoms Mole Fraction ($x_B$)", 0.0, 0.98, 0.05)
R = st.sidebar.number_input("Reflux Ratio (R)", min_value=1.1, value=2.5, step=0.1)
q = st.sidebar.slider("Feed Quality (q)", 0.0, 1.0, 1.0, step=0.1, help="0=Sat. Vapor, 0.5=50% Vapor, 1=Sat. Liquid")

# --- Input Validation ---
if comp1_name == comp2_name:
    st.error("Please select two different components.")
elif not xB < xF < xD:
    st.sidebar.error("Input Error: Compositions must be $x_B < x_F < x_D$.")
else:
    # Get Antoine constants
    antoine_1 = COMPONENT_DB[comp1_name]
    antoine_2 = COMPONENT_DB[comp2_name]
    comp_names = (comp1_name, comp2_name)

    # --- Create THREE columns for the plots --- # --- MODIFIED ---
    col1, col2, col3 = st.columns(3)

    # --- Column 1: P-x-y Diagram ---
    with col1:
        st.header(f"P-x-y Diagram")
        T_input = st.slider("Select Temperature (in °C)", min_value=-20.0, max_value=200.0, value=80.0, step=0.5)
        
        try:
            x1, y1, P, P1sat, P2sat = calculate_Pxy(T_input, antoine_1, antoine_2)
            fig_Pxy = plot_Pxy(x1, y1, P, P1sat, P2sat, T_input, comp_names)
            st.pyplot(fig_Pxy)
            st.markdown(f"* $P_1^{{sat}}$ ({comp1_name}): `{P1sat:.3f}` bar\n* $P_2^{{sat}}$ ({comp2_name}): `{P2sat:.3f}` bar")
        except Exception as e:
            st.error(f"Error in P-x-y plot: {e}")

    # --- Column 2: T-x-y Diagram ---
    with col2:
        st.header(f"T-x-y Diagram")
        P_input = st.slider("Select Pressure (in bar)", min_value=0.1, max_value=5.0, value=1.013, step=0.1)

        try:
            x1_T, y1_T, T_bub, T1boil, T2boil = calculate_Txy(P_input, antoine_1, antoine_2)
            fig_Txy = plot_Txy(x1_T, y1_T, T_bub, T1boil, T2boil, P_input, comp_names)
            st.pyplot(fig_Txy)
            st.markdown(f"* $T_1^{{boil}}$ ({comp1_name}): `{T1boil:.2f}` °C\n* $T_2^{{boil}}$ ({comp2_name}): `{T2boil:.2f}` °C")
        except Exception as e:
            st.error(f"Error in T-x-y plot: {e}")

    # --- NEW: Column 3: McCabe-Thiele Plot ---
    with col3:
        st.header("McCabe-Thiele Stages")
        
        try:
            # We already have the T-x-y data, which contains the equilibrium x-y curve
            # (x1_T and y1_T)
            
            stages, x_s, y_s, rol, sol, x_int, y_int = calculate_mccabe_thiele(
                x1_T, y1_T, xF, xD, xB, R, q
            )
            
            fig_MT, stage_count = plot_mccabe_thiele(
                x1_T, y1_T, xF, xD, xB, rol, sol, x_int, y_int, x_s, y_s, stages
            )
            
            st.pyplot(fig_MT)
            
            
            # Display the result
            st.metric(label="Total Number Theoretical Stages", value=f"{stage_count}")
            st.markdown(f"(Includes reboiler, which counts as one stage)")

        except Exception as e:
            st.error(f"Error in McCabe-Thiele calculation: {e}")

# --- Explanation Expander ---
with st.expander("About this Calculator"):
    st.markdown("""
    This application models VLE and performs distillation calculations.
    
    **1. VLE Calculations:**
    * **Raoult's Law:** $P_i = x_i \cdot P_i^{sat}$
    * **Antoine Equation:** $\log_{10}(P^{sat}) = A - \frac{B}{T + C}$
    
    **2. McCabe-Thiele Distillation:**
    * Calculates the number of **theoretical stages** for a binary distillation.
    * **Operating Lines:** The green (Rectifying) and red (Stripping) lines show the mass balance in each section of the column.
    * **q-line:** The magenta line represents the feed condition.
    * **Stages:** The orange "staircase" represents each equilibrium stage. The count includes the **reboiler** (which acts as one theoretical stage).
    """)