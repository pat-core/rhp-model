""" 
Python-implementation of rotating heat pipe model [Song2003] translated from MatLab by Gemini.

TODO 
----use object-oriented approach instead of global variables
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d 

# Set float precision for display in logs
np.set_printoptions(precision=10)

# ==============================================================================
# 1. GLOBAL VARIABLES (initialize here and set in set_global_variables)
# ==============================================================================

# liquid properties 
kl, hfg, rhol, mul, nul, betal, cpl, Pr = [0.0] * 8
# vapor properties 
rhov, muv, nuv = [0.0] * 3
# wall and rotor 
kw, Ro, Ri, RI, omega, alpha, TC, TE = [0.0] * 8
# discretization parameters
X, DX, Lc, La, Le, Riae = [0.0] * 6
N, Nc, Na, Ne = [0] * 4
mtC_rel_tol, dmt_diff_rel_tol = [0.0] * 2
max_inner_iterations, max_outer_iterations = [0] * 2
NUMZERO, max_restarts, MOD4GEOM = [0.0] * 3

# ==============================================================================
# 2. IMPLEMENTED FUNCTIONS
# ==============================================================================

def set_global_variables(configuration: int) -> int:
    """
    Initializes all physical constants and geometric/discretization arrays.
    """
    print(f"--- Calling set_global_variables({configuration}): Initializing RHP parameters. ---")
    
    # Use the global keyword to modify module-level variables
    global kl, hfg, rhol, mul, nul, betal, cpl, Pr
    global rhov, muv, nuv
    global kw, Ro, Ri, RI, omega, alpha, TC, TE
    global X, DX, Lc, La, Le, Riae
    global N, Nc, Na, Ne, mtC_rel_tol, dmt_diff_rel_tol
    global max_inner_iterations, max_outer_iterations, NUMZERO, max_restarts, MOD4GEOM

    # --- Discretization Parameters ---
    Nc = 23   # number of FVM in condenser
    Na = 10   # number of FVM in adiabatic section
    Ne = 19   # number of FVM in evaporator
    
    NUMZERO = 1e-14
    dmt_diff_rel_tol = 1e-6   # relative tolerance for mass flow difference 
    max_outer_iterations = 50   
    max_inner_iterations = 100
    max_restarts = 38   # restarts within inner loop
    MOD4GEOM = 50
    mtC_rel_tol = 1e-2   # iteration tolerance for mass flow at condenser end corresponding to Tsat

    # --- Operational Parameters ---
    TC = 60.0    # outer wall temperature (C)
    TE = 100.0   # outer wall temperature (E)
    rpm = 20000.0 # revolutions per minute

    # --- Liquid and Vapour Parameters ---
    kl = 0.598     # thermal conductivity water
    kw = 45.0      # thermal conductivity steel
    rhov = 0.768   # density vapour (p=101.300kPa, T=100°C)
    rhol = 998.0   # density water
    muv = 12.4e-6  # viscosity vapour
    mul = 1e-3     # viscosity water (T=20°C)
    betal = (1/3) * 20.7e-5 # thermal expansion
    cpl = 4186.0   # spezific heat capacity
    hfg = 2.257e6  # latent heat

    # --- Geometry ---
    Lc = 0.089
    La = 0.057
    Le = 0.076
    Ric = 0.0160 / 2    # inner radius at condenser (left)
    Riae = 0.0191 / 2   # inner radius at adiabatic section (left), constant from there on
    Ro_cae = 0.0254 / 2 # outer radius

    # --- Derived Parameters (Calculations) ---
    N = Nc + Na + Ne - 2 
    
    # 1. Coordinate Generation (X)
    Xc = np.linspace(0, Lc, Nc)
    Xa = Lc + np.linspace(0, La, Na)
    Xe = Lc + La + np.linspace(0, Le, Ne)
    
    X_flat = np.concatenate([Xc, Xa[1:-1], Xe])
    X = X_flat.reshape(-1, 1) # Column vector (N x 1)
    
    # 2. Finite Volume Lengths (DX)
    DX = np.diff(X, axis=0) # Column vector ((N-1) x 1)
    
    # 3. Viscosity and Prandtl
    nul = mul / rhol    # kin. viscosity liquid
    nuv = muv / rhov    # kin. viscosity vapour
    Pr = nul / (kl / (rhol * cpl)) # Prandtl Number
    
    # 4. Rotational Velocity
    omega = (rpm / 60) * 2 * np.pi
    
    # 5. Taper Angle (alpha)
    alpha = np.zeros((N - 1, 1))
    alpha_const = np.arctan((Riae - Ric) / Lc)
    alpha[:Nc - 1, 0] = alpha_const
    
    # 6. Inner Radius at Nodes (Ri)
    Ri_cum_sum = np.cumsum(DX * np.sin(alpha), axis=0)
    Ri = Ric + np.vstack([np.array([[0.0]]), Ri_cum_sum]) # Column vector (N x 1)
    
    # 7. Outer Radius at Nodes (Ro)
    Ro = np.ones((N, 1)) * Ro_cae # Column vector (N x 1)

    # 8. Inner Radius at FV Midpoints (RI)
    RI = (Ri[:-1] + Ri[1:]) / 2 # Column vector ((N-1) x 1)
    
    # 9. Outer Radius at FV Midpoints (RO)
    RO = (Ro[:-1] + Ro[1:]) / 2 # Column vector ((N-1) x 1)

    set_flag = 1
    return set_flag

# --- CORE PHYSICS HELPER FUNCTIONS ---

def look_up_song(d2D) -> float:
    """Looks up liquid mass as function of film height at evaporator end cap (look_up_song.m)"""
    d2D_data = np.array([0, 0.011, 0.019, 0.025, 0.036])
    ml0 = 0.7e-3
    ml_data = np.array([1, 3, 5, 7, 10]) * ml0
    interpolation_func = interp1d(d2D_data, ml_data, kind='linear', fill_value="extrapolate")
    
    ml = interpolation_func(d2D)
    
    return ml

def liquid_volume(delta: np.ndarray, RI: np.ndarray) -> float:
    """Calculates total liquid volume (liquid_volume.m)"""
    # V = sum(2 * pi * RI_mid * delta_mid * DX)
    delta_mid = (delta[:-1] + delta[1:]) / 2 
    return np.sum(2 * np.pi * RI.flatten() * delta_mid.flatten() * DX.flatten())

def grashof(rim: float, a: float, dTm: float, deltam: float) -> float:
    """Calculates Grashof number (grashof.m)"""
    global rhol, betal, mul, omega
    
    Gr = np.abs( (omega**2)*rim*np.cos(a)*betal*dTm*(deltam**3)/(nul**2) )
    return Gr

def reynolds(d: float, U: float) -> float:
    """Calculates Reynolds number (reynold.m)"""
    global nul
    Re=abs( 4*U*d/nul )
    return Re
    
def restart_values(N: int) -> np.ndarray:
    """Generates restart correction factors K (restart_values.m)"""
    K = np.zeros(N + 1)
    N2 = N // 2
    N21 = N2 + 1

    for k in range(N2):
        K[2 * k ] = (N21 - k) / N21
        K[2 * k + 1] = N21 / (N21 - k)

    return K

def liquid_velocities(d0: float, d1: float, mt0: float, dmt: float, ri0: float, dx: float, a: float, b: float):
    """
    velocity of vapour and liquid top layer (liquid_velocities.m).
    """
    global rhol, rhov
    
    # Vapour velocity
    uv0 = mt0/(rhov*np.pi*ri0**2)

    # Liquid velocity 
    Vt0 = mt0/rhol;
    u_avg0 = Vt0/(2*np.pi*ri0*d0); # average
    uld0 = (b+1)*u_avg0; # vapour interface
    
    return uv0, uld0, u_avg0

def heat_transfer_model_mixed_convection(d0: float, d1: float, Two: float, Tsat: float, dx: float, ri: float, ro: float, a: float):
    """
    Translates heat_transfer_model_mixed_convection.m. 
    Evaporator heat transfer model (Mixed Convection).
    """
    global kl, hfg, Pr, kw
    
    Nuf = 1.0 # Nusselt Number of forced convection
    d = (d0 + d1) / 2
    ln = ri * np.log(ro / ri)
    
    # 1. First run with Twi=Two (Outer Wall Temp)
    Gr = grashof(ri, a, Tsat - Two, d)
    Ra = Gr * Pr
    
    Nun = 0.133 * Ra**0.375 # Nusselt Number of natural convection
    Num = (Nuf**(7/2) + Nun**(7/2))**(2/7) # and mixed
    
    if d > NUMZERO: # there is a liquid layer
        Twi = ((Num * kl / d) * Tsat + (kw / ln) * Two) / (kw / ln + Num * kl / d)
    else: # there is no liquid layer
        Twi = Tsat

    # 2. Second run with Twi=Twi_first run (improved guess)
    Gr = grashof(ri, a, Tsat - Twi, d)
    Ra = Gr * Pr
    Nun = 0.133 * Ra**0.375
    Num = (Nuf**(7/2) + Nun**(7/2))**(2/7)

    if d > NUMZERO:
        Twi = ((Num * kl / d) * Tsat + (kw / ln) * Two) / (kw / ln + Num * kl / d)
    else:
        Twi = Tsat

    # Heat flow and mass flow difference
    qw = kw * (Twi - Two) / ln
    dmt = qw * 2 * np.pi * ri * dx / hfg
    
    return dmt, Twi, qw

def liquid_flow_model_mixed_convection(d0: float, d1: float, mt1: float, dx: float, a: float, ri0: float, uv1: float, uld1: float, Twi: float, Tsat: float):
    """
    Translates liquid_flow_model_mixed_convection.m. 
    Evaporator mass flow model (Mixed Convection).
    """
    global rhol, mul, nul, Pr, rhov, nuv, omega
    
    rim = ri0 + (dx / 2) * np.sin(a)
    ri1 = ri0 + dx * np.sin(a)
    dTm = Tsat - Twi
    deltam = (d0 + d1) / 2
    
    # Cross-sectional area for vapour flow
    Av = np.pi * (rim - deltam)**2 if (rim - deltam) > NUMZERO else 1e-10 
    
    # Estimate liquid velocities for flow regime determination (using b_forced=2)
    b_forced = 2
    # The MatLab code uses mt1+dmt=mt1+0 here, since dmt=0 initially
    _, _, u_avg = liquid_velocities(d0, d1, mt1, 0, ri0, dx, a, b_forced) 
    
    # Characteristic numbers (Gr and Re)
    Gr = grashof(rim, a, dTm, deltam)
    Re = reynolds(deltam, u_avg)
    Ra = Gr * Pr
    GrRe2 = Gr / (Re**2) if Re > NUMZERO else 1e10 # Use a large value if Re=0

    # Velocity profile power law exponent b
    if Ra < 1e9:
        if GrRe2 < 0.1: # forced convection
            b = 2.0
        elif GrRe2 < 10: # mixed convection
            b = 1/3
        else:
            b = 1/4
    else:
        b = 1/7 # natural convection

    # Wall friction [Song -> Afzal & Hussain]
    if Re > NUMZERO:
        K_fric = Gr / (Re**(5/2))
        if K_fric < 1:
            Cfw = 0.5
        else:
            Cfw = 0.5 * K_fric**(3/5)
    else:
        Cfw = 0.0

    # Vapour friction [Daniels]
    Rev = u_avg * Av / nuv # MatLab uses uv1, but that is 0 on the first pass. Using u_avg is safer if it's the average velocity
    if Rev > NUMZERO:
        if Rev < 2000:
            Cfd = 16 / Rev
        else:
            Cfd = 0.0791 / (Rev**0.25)
    else:
        Cfd = 0.0

    # Top velocity from mass flow at x=x1
    if d1 > NUMZERO and ri1 > NUMZERO:
        U1 = mt1 * (b + 1) / (2 * np.pi * ri1 * rhol * d1)
    else:
        U1 = 0.0 
    
    # --- Momentum balance (Quadratic Equation for U0) ---
    # The MatLab code uses constants 0.1e1=1.0, 0.2e1=2.0, 0.3e1=3.0, 0.4e1=4.0, 0.8e1=8.0, 0.16e2=16.0
    
    # Common factors
    A = Cfw * dx * (b**2) + (3.0/2.0 * Cfw * dx - d0 + d1) * b + Cfw * dx / 2.0
    B = 2.0 * (b + 1) * A * Av**2 * rhov
    C = 8.0 * Cfd * rim**2 * rhol * dx * (b + 1.0/2.0) * np.pi**2 * deltam**2
    D = 4.0 * Cfd * rim**2 * rhol * dx * (b + 1.0/2.0) * np.pi**2 * deltam**2

    # QP (Coefficient of U0)
    # The MatLab formula for QP is complex and seems to derive from a cubic/quartic or is simplified.
    # The MatLab code provided for QP:
    # QP = 0.4e1 * ((b + 1) * Av ^ 2 * A * rhov + D) * U1 
    #   / (0.2e1 * (b + 1) * (A + 2.0 * deltam) * Av ^ 2 * rhov + C) 
    
    # Recalculating MatLab's QP and QQ expression for U0
    
    Denominator = 2.0 * (b + 1) * (A + 2.0 * deltam) * Av**2 * rhov + C
    
    # Numerator for P * U0 (linear term)
    Numerator_P = 4.0 * ((b + 1) * Av**2 * A * rhov + D) * U1
    
    # QQ (Constant term)
    # The MatLab formula for QQ:
    # QQ = (-16.0 * rim * Av ^ 2 * omega ^ 2 * ((b + 1) ^ 2) * (b + 1.0/2.0) * rhov * (-d1 + d0) * deltam * np.cos(a) 
    #     - 16.0 * rim * Av ^ 2 * omega ^ 2 * dx * ((b + 1) ^ 2) * (b + 1.0/2.0) * rhov * deltam * np.sin(a) 
    #     + 2.0 * (Av ^ 2 * (A - 2.0 * deltam) * (b + 1) * rhov + D) * U1 ** 2) / Denominator
    
    # Note: There is a discrepancy in the MatLab formula for `A` vs `A - 2.0 * deltam`. 
    # I will stick strictly to the MatLab expression structure using the variables defined:

    # Re-calculating using MatLab's variable names for clarity, despite potential issues
    A_matlab_expr = Cfw * dx * (b ** 2) + (3.0 / 2.0 * Cfw * dx - d0 + d1) * b + Cfw * dx / 2.0
    
    QP_num = 4.0 * ((b + 1) * Av**2 * A_matlab_expr * rhov + D) * U1
    QP_den = 2.0 * (b + 1) * (A_matlab_expr + 2.0 * deltam) * Av**2 * rhov + C

    if QP_den < NUMZERO:
        QP = 0.0
    else:
        QP = QP_num / QP_den

    QQ_num_term1 = -16.0 * rim * Av**2 * omega**2 * (b + 1)**2 * (b + 0.5) * rhov * (d0 - d1) * deltam * np.cos(a)
    QQ_num_term2 = -16.0 * rim * Av**2 * omega**2 * dx * (b + 1)**2 * (b + 0.5) * rhov * deltam * np.sin(a)
    
    # Note on A vs (A - 2*deltam): I will assume the MatLab code intended the expression with -2*deltam for the U1^2 term.
    A_prime_matlab_expr = Cfw * dx * (b ** 2) + (3.0 / 2.0 * Cfw * dx - d0 + d1) * b + Cfw * dx / 2.0 - 2.0 * deltam
    
    QQ_num_term3 = 2.0 * (Av**2 * (b + 1) * rhov * A_prime_matlab_expr + D) * U1**2
    QQ_num = QQ_num_term1 + QQ_num_term2 + QQ_num_term3
    
    if QP_den < NUMZERO:
        QQ = 0.0
    else:
        QQ = QQ_num / QP_den
    
    # Solve Quadratic Equation: U0^2 + QP*U0 + QQ = 0
    # U0 = (-QP / 2) +/- sqrt((QP / 2)^2 - QQ)
    
    DISCRIMI = (QP/2)**2 - QQ
    
    if DISCRIMI >= 0:
        # Prefer higher value (as per MatLab comment)
        U0 = (-QP/2) + np.sqrt(DISCRIMI) 
    else:
        U0 = 0.0
        print('No real solution for U0 in liquid_flow_model_mixed_convection (evaporator). Setting U0=0.')
    
    # Final mass flow rate and difference
    mt0 = U0 * (2 * np.pi * ri0 * rhol * d1) / (b + 1)
    dmt = mt1 - mt0

    return dmt, b

def liquid_flow_model_film_condensation(d0: float, d1: float, mt1: float, dx: float, a: float, ri0: float, uv: float, uld: float):
    """
    Translates liquid_flow_model_film_condensation.m.
    Condenser/Adiabatic mass flow model (Film Condensation).
    """
    global rhol, mul, omega
    
    # Solve for dmt=Z/N, where mt1 = mt0 + dmt
    
    Z1_term1 = mt1
    Z1_term2 = (rhol / mul) * (omega**2) * ri0 * (np.sin(a) - np.cos(a) * (d1 - d0) / dx) * (1/3) * (d0**3) * 2 * np.pi * ri0 * rhol
    Z1 = Z1_term1 - Z1_term2
    
    N1_term1 = 1.0
    N1_term2 = (1/2) * ((d0**2) / (mul * dx)) * (uv * np.cos(a) + uld) * 2 * np.pi * ri0 * rhol
    N1 = N1_term1 - N1_term2
    
    if N1 > NUMZERO:
        dmt = Z1 / N1
    else:
        # Avoid division by zero, setting dmt to an unlikely large value to force a restart
        dmt = 1e10 

    b = 2.0 # Power law exponent for Film Condensation is 2
    
    return dmt, b

def heat_transfer_model_film_condensation(d0: float, d1: float, Two: float, Tsat: float, dx: float, ri: float, ro: float):
    """
    Translates heat_transfer_model_film_condensation.m.
    Condenser heat transfer model (Film Condensation).
    """
    global kl, hfg, kw
    
    # FVM evaluation at midpoint x=(x0+x1)/2
    d = (d0 + d1) / 2
    ln = ri * np.log(ro / ri)
    
    if d > NUMZERO: # there is a liquid layer
        Twi = ((kl / d) * Tsat + (kw / ln) * Two) / (kw / ln + kl / d)
    else: # there is no liquid layer
        Twi = Tsat

    qw = kw * (Twi - Two) / ln
    dmt = qw * 2 * np.pi * ri * dx / hfg
    
    return dmt, Twi, qw

# --- FV ITERATION FUNCTIONS (Translated from MatLab) ---

def iterate_fv_evaporator(d0start: float, d1: float, mt1: float, uv1: float, uld1: float, Tsat: float, k_matlab: int):
    """
    Translates iterate_fv_evaporator.m. Solves the mass balance (dmt_ht = dmt_lf)
    for the upstream liquid film height d0 using Regula Falsi (Mixed Convection model).
    k_matlab: MatLab 1-based index of the downstream node.
    """
    global Ro, Ri, alpha, TE
    global DX, dmt_diff_rel_tol, max_inner_iterations, max_restarts, NUMZERO

    fv_idx = k_matlab - 2 # Python FV index (0 to N-2)
    
    dx = DX[fv_idx].item()
    a = alpha[fv_idx].item()
    
    # ri=Ri(k-1)/2 + Ri(k)/2; mid-point values
    ri = (Ri[fv_idx].item() + Ri[fv_idx+1].item()) / 2 
    ro = (Ro[fv_idx].item() + Ro[fv_idx+1].item()) / 2 
    ri0 = Ri[fv_idx].item() # index of left node (x=x0)
    
    K = restart_values(max_restarts)
    
    # Initialize FVM iteration
    d0 = d0start
    dmt_diff_rel = 1.0
    mt0 = -mt1 # Initial value to ensure loop entry
    restart_count = 0
    iteration_count = 0
    
    # Regula Falsi history
    d0prev = 0.0
    fzprev = 0.0
    dmt = 0.0
    b_lf = 0.0 # Must be defined for liquid_velocities
    Tw = 0.0
    qw = 0.0
    
    while ((dmt_diff_rel > dmt_diff_rel_tol) or (mt0 < 0)) and (restart_count < max_restarts) and (iteration_count < max_inner_iterations):
        iteration_count += 1
        
        [dmt_ht, Tw, qw] = heat_transfer_model_mixed_convection(d0, d1, TE, Tsat, dx, ri, ro, a)
        [dmt_lf, b_lf] = liquid_flow_model_mixed_convection(d0, d1, mt1, dx, a, ri0, uv1, uld1, Tw, Tsat)
        
        dmt_diff_rel = abs(dmt_ht - dmt_lf) / (abs(dmt_ht) + abs(dmt_lf)) if (abs(dmt_ht) + abs(dmt_lf)) > NUMZERO else 1.0
        
        fz = (dmt_ht - dmt_lf) # function to be iterated to zero
        
        if iteration_count == 1: 
            if fz > 0:
                d0next = (4/5) * d0
            else:
                d0next = (5/4) * d0
        else: # Regula Falsi
            if abs(fz - fzprev) < NUMZERO: # Avoid division by zero in Regula Falsi
                 d0next = d0 * 0.99 
            else:
                 d0next = d0 - fz * (d0 - d0prev) / (fz - fzprev)
            
        # shift to next iteration
        fzprev = fz
        d0prev = d0
        
        if (d0next < 0) or np.isnan(d0next):
            restart_count += 1
            if restart_count >= max_restarts: break # Safety break
            
            # Restart condition based on restart factors K
            if d0start == 0:
                d0 = K[restart_count-1] * d1 
            else:
                d0 = K[restart_count-1] * d0start
            dmt = 0.0
        else:
            d0 = d0next
            dmt = dmt_ht # Using heat transfer mass flow for the balance
            mt0 = mt1 - dmt # mt0 + dmt = mt1

    # Final velocity calculation
    [uv0, uld0, _] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf) 
    
    return d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count

def iterate_fv_condenser(d0start: float, d1: float, mt1: float, uv1: float, uld1: float, Tsat: float, k_matlab: int):
    """
    Translates iterate_fv_condenser.m. Solves the mass balance (dmt_ht = dmt_lf)
    for the upstream liquid film height d0 using Regula Falsi (Film Condensation model).
    k_matlab: MatLab 1-based index of the downstream node.
    """
    global Ro, Ri, alpha, TC
    global DX, dmt_diff_rel_tol, max_inner_iterations, max_restarts, MOD4GEOM, NUMZERO

    fv_idx = k_matlab - 2 # Python FV index (0 to N-2)

    dx = DX[fv_idx].item()
    a = alpha[fv_idx].item()
    
    ri = (Ri[fv_idx].item() + Ri[fv_idx+1].item()) / 2 
    ro = (Ro[fv_idx].item() + Ro[fv_idx+1].item()) / 2 
    ri0 = Ri[fv_idx].item() # index of left node (x=x0)
    
    K = restart_values(max_restarts)
    
    # Initialize FVM iteration
    d0 = d0start
    dmt_diff_rel = 1.0
    mt0 = -mt1
    restart_count = 0
    iteration_count = 0
    
    # Regula Falsi history
    d0prev = 0.0
    fzprev = 0.0
    dmt = 0.0
    b_lf = 0.0
    Tw = 0.0
    qw = 0.0
    
    while ((dmt_diff_rel > dmt_diff_rel_tol) or (mt0 < 0)) and (restart_count < max_restarts) and (iteration_count < max_inner_iterations):
        iteration_count += 1
        
        [dmt_lf, b_lf] = liquid_flow_model_film_condensation(d0, d1, mt1, dx, a, ri0, uv1, uld1)
        [dmt_ht, Tw, qw] = heat_transfer_model_film_condensation(d0, d1, TC, Tsat, dx, ri, ro)
        
        dmt_diff_rel = abs(dmt_ht - dmt_lf) / (abs(dmt_ht) + abs(dmt_lf)) if (abs(dmt_ht) + abs(dmt_lf)) > NUMZERO else 1.0
        
        fz = (dmt_ht - dmt_lf)
        
        if iteration_count == 1:
            if fz > 0:
                d0next = (4/5) * d0
            else:
                d0next = (5/4) * d0
        else: # Regula Falsi
            if abs(fz - fzprev) < NUMZERO:
                 d0next = d0 * 0.99
            else:
                 d0next = d0 - fz * (d0 - d0prev) / (fz - fzprev)
            
        # shift to next iteration
        d0prev = d0
        fzprev = fz
        
        if (d0next < 0) or np.isnan(d0next):
            restart_count += 1
            if restart_count >= max_restarts: break
            
            if d0start == 0:
                d0 = K[restart_count-1] * d1
            else:
                d0 = K[restart_count-1] * d0start
            dmt = 0.0
        else:
            if iteration_count % MOD4GEOM == 0: # Geometric convergence acceleration
                d0 = np.sqrt(d0next * d0)
            else:
                d0 = d0next
                
            dmt = dmt_ht
            mt0 = mt1 - dmt

    # Final velocity calculation
    [uv0, uld0, _] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf) 
    
    return d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count

def iterate_fv_adiabatic(d0start: float, d1: float, mt1: float, uv1: float, uld1: float, Tsat: float, k_matlab: int):
    """
    Translates iterate_fv_adiabatic.m. Solves for d0 where dmt_lf = 0
    (zero mass transfer) using Regula Falsi (Film Condensation liquid flow model).
    k_matlab: MatLab 1-based index of the downstream node.
    """
    global Ri, alpha
    global DX, NUMZERO, max_inner_iterations, max_restarts

    fv_idx = k_matlab - 2 # Python FV index (0 to N-2)

    dx = DX[fv_idx].item()
    a = alpha[fv_idx].item()
    ri0 = Ri[fv_idx].item() # index of left node (x=x0)

    K = restart_values(max_restarts)
    
    # Initialize FVM iteration
    d0 = d0start
    iteration_count = 0
    restart_count = 0
    
    dmt_lf = 1.0 # Set to enter iteration
    dmt_ht = 0.0 # Expected dmt_ht in Adiabatic section
    
    # Regula Falsi history
    d0prev = 0.0
    fzprev = 0.0
    dmt = dmt_ht # dmt is forced to 0
    b_lf = 0.0
    Tw = Tsat
    qw = 0.0

    while (abs(dmt_lf) > NUMZERO) and (iteration_count < max_inner_iterations) and (restart_count < max_restarts):
        iteration_count += 1
        
        [dmt_lf, b_lf] = liquid_flow_model_film_condensation(d0, d1, mt1, dx, a, ri0, uv1, uld1)
        
        fz = (dmt_ht - dmt_lf) # function to be iterated to zero
        
        if iteration_count == 1:
            if fz > 0:
                d0next = (4/5) * d0
            else:
                d0next = (5/4) * d0
        else: # Regula Falsi
            if abs(fz - fzprev) < NUMZERO:
                 d0next = d0 * 0.99
            else:
                 d0next = d0 - fz * (d0 - d0prev) / (fz - fzprev)
            
        d0prev = d0
        
        if np.isnan(d0next) or d0next < 0:
            restart_count += 1
            if restart_count >= max_restarts: break
            d0 = K[restart_count-1] * d0start
        else:
            d0 = d0next
            
        fzprev = fz
            
    mt0 = mt1 - dmt_ht # mt0 + dmt_ht = mt1, where dmt_ht = 0
    
    # Final velocity calculation
    [uv0, uld0, _] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf)
    
    # Tw and qw are constant return values in Adiabatic section
    return d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count

def rhp_inner_loop(dEend: float, Tsat: float, delta0: np.ndarray):
    """
    Translates rhp_inner_loop.m.
    Iterates equations for each FV from evaporator end to condenser end.
    
    Returns: QC, QE, mtC, delta, mt, uld, uv, Tw, d0start, knc, FViterations, FVrestarts
    """
    global RI
    global X, N, Nc, Na, DX, max_inner_iterations, NUMZERO, max_restarts

    knc = N           
    QE = 0.0          
    QC = 0.0          

    # N-1 arrays (FV properties)
    FViterations = np.zeros(N - 1, dtype=int)
    FVrestarts = np.zeros(N - 1, dtype=int)
    Tw = np.zeros(N - 1) 
    
    # N arrays (Node properties)
    delta = np.zeros(N) 
    mt = np.zeros(N)  
    uld = np.zeros(N) 
    uv = np.zeros(N)  

    diffd0 = np.diff(delta0)
    
    # Boundary conditions at the Evaporator End (Node N, Python index N-1)
    delta[N-1] = dEend
    mt[N-1] = 0.0
    uld[N-1] = 0.0
    uv[N-1] = 0.0
    
    d0start = 0.0
    mtC = 0.0
    
    i = N - 1 # Start at last node (Evaporator end, Python index N-1)
    k_matlab = N # MatLab node index (N)
    
    while i > 0:
        fv_idx = i - 1
        
        d1 = delta[i]
        mt1 = mt[i]
        uld1 = uld[i]
        uv1 = uv[i]
        
        inner_iteration_count = 0
        restart_count = 0
        qw = 0.0
        Twi = 0.0
        
        # --- Section Check (MatLab k = i+1) ---
        if k_matlab >= Nc + Na: # Evaporator (FV indices Nc+Na-1 to N-2)
            d0start = max(d1 - diffd0[fv_idx], delta0[fv_idx])
            results = iterate_fv_evaporator(d0start, d1, mt1, uv1, uld1, Tsat, k_matlab)
            d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count = results
            Tw[fv_idx] = Twi
            QE += qw * 2 * np.pi * RI[fv_idx] * DX[fv_idx]
            
        elif k_matlab > Nc: # Adiabatic (FV indices Nc-1 to Nc+Na-2)
            d0start = d1
            results = iterate_fv_adiabatic(d0start, d1, mt1, uv1, uld1, Tsat, k_matlab)
            d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count = results
            Tw[fv_idx] = Twi 
            
        else: # Condenser (FV indices 0 to Nc-2)
            # delta(Nc) is delta[Nc-1]. delta0(Nc) is delta0[Nc-1]
            d0start = min(delta0[fv_idx] * delta[Nc-1] / delta0[Nc-1], d1)
            results = iterate_fv_condenser(d0start, d1, mt1, uv1, uld1, Tsat, k_matlab)
            d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count = results
            Tw[fv_idx] = Twi
            QC += qw * 2 * np.pi * RI[fv_idx] * DX[fv_idx]

        # Store results at upstream node i-1 (Python fv_idx)
        FViterations[fv_idx] = inner_iteration_count
        FVrestarts[fv_idx] = restart_count
        delta[fv_idx] = d0
        mt[fv_idx] = mt0
        uld[fv_idx] = uld0
        uv[fv_idx] = uv0
        
        if i == 1:
            mtC = mt0
            
        # --- Check for Non-Convergence and Extrapolation ---
        if ((inner_iteration_count == max_inner_iterations) or (restart_count == max_restarts)):
            if (i > 1) and (i < N - 1): # Between first Condenser node (i=1) and Evaporator End (i=N-1)
                dmt = mt[i+1] - mt[i]
                
                if abs(dmt) > NUMZERO: 
                    # mtC = mt0 - X[k_matlab-1] * dmt / (X[k_matlab] - X[k_matlab-1])
                    mtC = mt0 - X[i] * dmt / (X[i+1] - X[i])
                else: 
                    mtC = mt0
                
            else: 
                mtC = mt0
                
            delta[fv_idx] = 0.0
            knc = k_matlab # Store non-converged FV (MatLab index)
            i = 0          # Stop FV evaluations
            
        i -= 1 # Go to next FV (upstream)
        k_matlab -= 1

    if (knc == N): # Only check if not already broken from non-convergence
        if ((inner_iteration_count < max_inner_iterations) and (restart_count < max_restarts)):
            knc = 0 # Fully converged
        
    return QC, QE, mtC, delta.flatten(), mt.flatten(), uld.flatten(), uv.flatten(), Tw.flatten(), d0start, knc, FViterations, FVrestarts

def rhp_outer_loop(dEend: float, Tsat0: float, delta0: np.ndarray, fileID: object):
    """
    Translates rhp_outer_loop.m
    Iterates over Tsat to find steady state where mtC is close to zero.
    """
    
    # Use global variables
    global hfg, rhol
    global RI, alpha, TC, TE
    global N, Nc, DX, mtC_rel_tol, max_inner_iterations, max_outer_iterations, NUMZERO, max_restarts

    # Calculate Condenser inner wall area for heat flux qc
    Ac = np.sum(2 * np.pi * RI[:Nc-1].flatten() * DX[:Nc-1].flatten())

    # --- Generate Tsat-grid ---
    # LOGN = (1 - logspace(-2, 0, max_outer_iterations)) * 100/99;
    LOGN = (1 - np.logspace(-2, 0, max_outer_iterations)) * (100.0 / 99.0)
    Tsat_v = TC + NUMZERO + LOGN * (Tsat0 - TC)
    
    knc_v = np.zeros(max_outer_iterations, dtype=int)
    mtC_rel_v = np.zeros(max_outer_iterations)
    QCE_rel_v = np.zeros(max_outer_iterations)

    # --- Evaluate Tsat-grid --- 
    for outer_iteration_count in range(max_outer_iterations):
        Tsat = Tsat_v[outer_iteration_count]
        
        results = rhp_inner_loop(dEend, Tsat, delta0)
        QC, QE, mtC, _, mt, _, _, _, _, knc, _, _ = results
        
        QCE = (QC + QE)
        QCE_abs_sum = np.abs(QC) + np.abs(QE)
        QCE_rel = QCE / QCE_abs_sum if QCE_abs_sum > NUMZERO else 0.0
        QCE_rel_v[outer_iteration_count] = QCE_rel
        
        mt_max_current = np.max(mt)
        mtC_rel = mtC / mt_max_current if mt_max_current > NUMZERO else 0.0
        mtC_rel_v[outer_iteration_count] = mtC_rel
        knc_v[outer_iteration_count] = knc

    # --- Determine Tsat_ss ---
    index_converged = np.where(knc_v == 0)[0]
    index_diverged = np.where(knc_v > 0)[0]
    count_converged = len(index_converged)
    Tsat_ss = Tsat0 

    if count_converged > 2:
        x_converged = mtC_rel_v[index_converged]
        y_converged = Tsat_v[index_converged]
        f = interp1d(x_converged, y_converged, kind='cubic', fill_value="extrapolate")
        Tsat_ss = f(mtC_rel_tol).item()
    elif count_converged > 0:
        x_converged = mtC_rel_v[index_converged]
        y_converged = Tsat_v[index_converged]
        f = interp1d(x_converged, y_converged, kind='linear', fill_value="extrapolate")
        Tsat_ss = f(mtC_rel_tol).item()
    else:
        min_knc = np.min(knc_v) if len(knc_v) > 0 else N # If no runs, keep initial guess Tsat0
        Tsat_ss = np.min(Tsat_v[np.where(knc_v == min_knc)[0]]) if len(np.where(knc_v == min_knc)[0]) > 0 else Tsat0

    # --- Final Run (Interpolated Tsat) ---
    results = rhp_inner_loop(dEend, Tsat_ss, delta0)
    QC, QE, mtC, delta, mt, uld, uv, Tw, d0start_final, knc, FViterations, FVrestarts = results

    QCE = (QC + QE)
    QCE_abs_sum = np.abs(QC) + np.abs(QE)
    QCE_rel = QCE / QCE_abs_sum if QCE_abs_sum > NUMZERO else 0.0
    
    mt_max = np.max(mt)
    mtC_rel = mtC / mt_max if mt_max > NUMZERO else 0.0

    # --- Convergence Info Logging ---
    delta_max = np.max(delta) if np.max(delta) > NUMZERO else 1.0
    #log_line_1 = f'mtC_rel={mtC_rel:.12f}    QCE_rel={QCE_rel:.12f}    deltaC/max(delta)={delta[0]/delta_max:.4f}    QC={QC:.2f} QE={QE:.2f}    converged={count_converged}/{max_outer_iterations} \n'
    #fileID.write(log_line_1)  # TODO enable again
    
    valid_iterations = FViterations[FViterations > 0]
    if len(valid_iterations) > 0:
        mean_iter = np.mean(valid_iterations)
        min_iter = np.min(valid_iterations)
        max_iter = np.max(FViterations) 
        log_line_2 = f'inner iterations (final run): mean={mean_iter:.2f}    min={min_iter}    max={max_iter}    (max_inner_iterations={max_inner_iterations}) \n'
        fileID.write(log_line_2)
    else:
         fileID.write('inner iterations (final run): No valid iterations recorded.\n')
    
    valid_restarts = FVrestarts[FViterations > 0] 
    if np.max(FVrestarts) > 0:
        mean_restart = np.mean(valid_restarts) if len(valid_restarts) > 0 else 0
        min_restart = np.min(valid_restarts) if len(valid_restarts) > 0 else 0
        max_restart = np.max(FVrestarts)
        log_line_3 = f'inner restarts (final run): mean={mean_restart:.2f}    min={min_restart}    max={max_restart}    (max_restarts={max_restarts}) \n'
        fileID.write(log_line_3)
    else:
        fileID.write('inner restarts (final run): 0 restarts \n')

    # --- Results Calculation ---
    V = liquid_volume(delta.reshape(-1, 1), RI)
    Q = mt_max * hfg 
    qc = Q / Ac      
    
    log_line_4 = f'Tsat_ss={Tsat_ss:.6f}°C    ml={V*rhol*1000:.6f} g    max(mlt)={mt_max:.6f} kg/s    qc={qc:.6f} W/m^2 \n\n'
    fileID.write(log_line_4)

    # --- Prepare GrRe2 for Plots ---
    GrRe2 = np.zeros(N - 1)
    
    RI_flat = RI.flatten()
    alpha_flat = alpha.flatten()
    
    for k in range(N - 1): 
        # uld, delta, Tw are N-element arrays/N-1 arrays
        Um = (uld[k] + uld[k+1]) / 2
        deltam = (delta[k] + delta[k+1]) / 2
        
        Gr = grashof(RI_flat[k], alpha_flat[k], Tw[k] - Tsat_ss, deltam)
        Re = reynolds(deltam, Um)
        Re2 = Re**2

        if Re2 > NUMZERO: 
            GrRe2[k] = Gr / Re2
        else:
            GrRe2[k] = 0.0
            
    return delta, mt, Tw, GrRe2, V, Tsat_ss, qc, knc, count_converged, Tsat_v, mtC_rel_v, QCE_rel_v, index_converged, index_diverged


# ==============================================================================
# 3. MAIN DRIVER SCRIPT
# ==============================================================================

def run_rhp_model():
    """
    Main execution function
    """
    # 1. INITIAL SETUP
    # plt.close('all') # Not needed in this environment

    set_global_variables(1)

    Nd = 1 # 21 Number of dEend discretization points
    dEend2Dmin = 0.0
    dEend2Dmax = 0.035   # 0.035 for Nd=1 and 0.0058595 for Nd=21

    if N == 0:
        print("Error: Global variables not set. Exiting.")
        return

    L = Lc + La + Le
    meanRi = np.mean(Ri)
    #meanRic = np.mean(Ri[0:Nc-1])   #DK
    #Di = 2*Riae   #DK
    
    # Initial guess for first run (next runs take previous results as guess)
    dEendmin = (2 * meanRi) * dEend2Dmin
    ml = look_up_song(dEend2Dmin)

    delta_konst = ((ml / rhol) / (2 * np.pi * meanRi) - dEendmin * Le / 2) / (Lc / 2 + La + Le / 2)
    
    # Initial film height profile (linear condenser/evaporator, constant adiabatic)
    dc0 = np.linspace(0, delta_konst, Nc)   #DK.reshape(-1, 1)
    da0_size = Na - 2   #because first and last node are in dc/de
    da0 = np.ones(da0_size) * delta_konst
    de0 = np.linspace(delta_konst, dEendmin, Ne)   #DK.reshape(-1, 1)
    delta0 = np.concatenate([dc0, da0, de0])

    if delta0.shape[0] != N:
        print(f"FATAL ERROR: Initial delta vector size mismatch. Expected {N}, got {delta0.shape[0]}.")
        return

    # 2. RUN NUMERICAL SOLUTION (LOOP)
    print(f"-- RUNNING RHP model with N={N} finite volumes --")
    
    log_filename = 'local_iterations.log'
    if os.path.exists(log_filename):
        os.remove(log_filename)
        
    fileID = open(log_filename, 'a')

    dEend2Dinput = np.linspace(dEend2Dmin, dEend2Dmax, Nd)
    
    delta_results = np.zeros((N, Nd))
    mt_results = np.zeros((N, Nd))
    Tw_results = np.zeros((N - 1, Nd))
    GrRe2_results = np.zeros((N - 1, Nd))
    qc_results = np.zeros(Nd)
    V_results = np.zeros(Nd)
    Tsat_results = np.zeros(Nd)
    
    Tsat0 = (TC + TE) / 2 # Initial guess for Tsat

    knc = N
    kk = 0 
    for kk in range(Nd): 
        dEend2D = dEend2Dinput[kk]
        dEend = (2 * meanRi) * dEend2D
        
        print('\n')
        print(f"dEend/D={dEend2D:.6f}")
        fileID.write(f"dEend/D={dEend2D:.6f} \n")

        results = rhp_outer_loop(dEend, Tsat0, delta0.flatten(), fileID)
        
        (delta, mt, Tw, GrRe2, V, Tsat_ss, qc, knc, 
         count_converged, Tsat_v, mtC_rel_v, QCE_rel_v, 
         index_converged, index_diverged) = results
        
        delta_results[:, kk] = delta
        mt_results[:, kk] = mt
        Tw_results[:, kk] = Tw
        GrRe2_results[:, kk] = GrRe2
        V_results[kk] = V
        Tsat_results[kk] = Tsat_ss
        qc_results[kk] = qc
        
        if (knc > 0) or (count_converged < 2):
            print('_')
            print(f"WARNING, solution not converged for DEend/D={dEend2D:.6f}")
            print(f"count_converged={count_converged};")
            if (knc > 1):
                print("DEBUG INFO")
                # knc is the MatLab 1-based index, knc-1 is Python 0-based index
                print(f"knc={knc};")
                print(f"mt1={mt[knc-1]:.10f};") 
                print(f"d1={delta[knc-1]:.10f};") 
                print(f"Tsat={Tsat_ss:.10f};")
            break   # no hope to try higher values of dEend
        else:
            print(f"Tsat={Tsat_ss:.2f}°C    liquid mass m={V*rhol*1000:.2f} g    heat flux qc={qc:.2f} W/m^2")

        delta0 = delta.reshape(-1, 1) 
        Tsat0 = Tsat_ss
        
    fileID.close()
    print(f"\nLog file written to '{log_filename}'")


    # 3. POSTPROCESSING (Plotting based on the last converged run)
    
    last_k = kk
    # If loop broke due to non-convergence, use the previous step's converged data if available
    if (knc > 0 or count_converged < 2) and kk > 0:
        last_k = kk - 1
        
    if 'Tsat_v' in locals() and Tsat_v.size > 0:
        
        # Plot 1: Tsat search results (Convergence plot)
        plt.figure(figsize=(10, 6))
        
        Tsat_con = Tsat_v[index_converged]
        QCE_rel_con = QCE_rel_v[index_converged]
        Tsat_div = Tsat_v[index_diverged]
        QCE_rel_div = QCE_rel_v[index_diverged]
        mtC_rel_con = mtC_rel_v[index_converged]
        mtC_rel_div = mtC_rel_v[index_diverged]
        
        plt.plot(Tsat_con, QCE_rel_con, 'r.', label='Q con')
        plt.plot(Tsat_div, QCE_rel_div, 'ro', fillstyle='none', label='Q div')
        plt.plot(Tsat_con, mtC_rel_con, 'b.', label='mt con')
        plt.plot(Tsat_div, mtC_rel_div, 'bo', fillstyle='none', label='mt div')
        
        plt.plot(Tsat_results[last_k], 0, 'k*', markersize=10, label='Tsat ss')
        
        plt.xlabel(r'$T_{\mathrm{sat}}$')
        plt.ylabel('ERROR')
        plt.title(f'Tsat search for last dEend/D={dEend2Dinput[last_k]:.6f}')
        plt.legend()
        plt.grid(True, linestyle='--')
        
        # Plot 2: Results vs. x/L (Subplots)
        final_delta = delta_results[:, last_k]
        final_mt = mt_results[:, last_k]
        final_Tw = Tw_results[:, last_k]
        final_GrRe2 = GrRe2_results[:, last_k]
        final_Tsat_ss = Tsat_results[last_k]
        
        Xm = (X[:-1] + X[1:]) / 2 
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'RHP Model Results for Last dEend/D={dEend2Dinput[last_k]:.6f}')
        
        # 221: d/D vs x/L
        axes[0, 0].plot(X / L, final_delta / (2 * meanRi), 'k.')
        axes[0, 0].set_xlabel('$x/L$')
        axes[0, 0].set_ylabel(r'$d/D$')
        axes[0, 0].set_title('Liquid Film Height')
        
        # 222: mt vs x/L
        axes[0, 1].plot(X / L, final_mt)
        axes[0, 1].set_xlabel('$x/L$')
        axes[0, 1].set_ylabel('$mt$')
        axes[0, 1].set_title('Mass Flow Rate')

        # 223: Twi-Tsat ss vs x/L
        axes[1, 0].plot(Xm / L, final_Tw - final_Tsat_ss)
        axes[1, 0].set_xlabel('$x/L$')
        axes[1, 0].set_ylabel(r'$T_{\mathrm{wi}} - T_{\mathrm{sat,ss}}$')
        axes[1, 0].set_title('Wall Superheat')
        
        # 224: Gr/Re^2 vs x/L (Evaporator section)
        # N-Ne is the node index corresponding to the start of the Evaporator.
        evap_start_index = N - Ne 
        
        #axes[1, 1].semilogy(Xm[evap_start_index - 1 : N-1].flatten() / L, final_GrRe2[evap_start_index - 1 : N-1])   # TODO enable again
        axes[1, 1].plot(Xm[evap_start_index - 1 : N-1].flatten() / L, final_GrRe2[evap_start_index - 1 : N-1])
        axes[1, 1].set_xlabel('$x/L$')
        axes[1, 1].set_ylabel(r'$Gr/Re^2$')
        axes[1, 1].set_title('Mixed Convection Parameter (Evaporator)')
        axes[1, 1].set_xlim(X[evap_start_index] / L, X[-1] / L)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.show()

if __name__ == "__main__":
    run_rhp_model()

