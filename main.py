import numpy as np
from numpy import linspace, diff, zeros, ones, pi, atan, cumsum, sin, cos 

class RHP:
    def __init__(self, evaporator_temp, condenser_temp):
        """
        Initialize a Reverse Heat Pump model.
        
        Args:
            evaporator_temp (float): Evaporator temperature in Celsius
            condenser_temp (float): Condenser temperature in Celsius
            
        """
        #self.filling_ratio = filling_ratio

        # discretization parameters
        self.Nc=23 
        self.Na=10 
        self.Ne=19   # nodes CAE (89-57-76) Nc=45; Na=28; Ne=38;
        self.NUMZERO=1e-14   # for iterations in adiabatic section
        self.dmt_diff_rel_tol=1e-6 # iteration tol. for rel. difference in mass flow
        self.max_outer_iterations=50  # number Tsat mesh points (outer loop over all FV)
        self.max_inner_iterations=100 # maximum number of iterations for mt (inner iterations for each FV)
        self.max_restarts=38    # restart iterations within inner loop (per FV)
        self.MOD4GEOM=50    # use geometric mean in between to accelerate convergence
        self.mtC_rel_tol=1e-2  # iteration tol. for mt at Condenser end corresponding to Tsat

        self.TC=condenser_temp   # outer wall temperature (C)
        self.TE=evaporator_temp   # outer wall temperature (E)
        self.rpm=20000 # revolutions per minute, determines OMEGA

        # liquid and vapour parameters
        self.kl=0.598   # thermal conductivity water
        self.kw=45   # thermal conductivity steel
        self.rhov=0.768 #0.768; # density vapour (p=101.300kPa, T=100°C)
        self.rhol=998  # density water
        self.muv=12.4e-6   # viscosity vapour (T=100°C, 1bar) http://www.wissenschaft-technik-ethik.de/wasser_eigenschaften.html#kap06
        self.mul=1e-3       # viscosity water (T=20°C)
        self.betal=(1/3)*20.7e-5  # thermal expansion
        self.cpl=4186   # spezific heat capacity
        self.hfg=2.257e6    # latent heat

        # geometry
        self.Lc=0.089 
        self.La=0.057 
        self.Le=0.076   # lengths, determines XC,XA,XE
        self.Ric=0.0160/2    # inner radius at condenser (left), determines RI
        self.Riae=0.0191/2    # inner radius at adiabatic section (left), constant from there on, determines RI
        self.Ro_cae=0.0254/2   # outer radius, determines RO

        # derived parameters
        self.N=self.Nc+self.Na+self.Ne-2   # number of nodes (concatination at CA and AE)
        self.Xc=linspace(0, self.Lc, self.Nc) # determines X
        self.Xa=self.Lc+linspace(0,self.La,self.Na) # determines X
        self.Xe=self.Lc+self.La+linspace(0,self.Le,self.Ne) # determines X
        self.X=np.ravel([self.Xc, self.Xa[1:-1], self.Xe[:]])
        self.DX=diff(self.X)
        self.nul=self.mul/self.rhol   # kin. viscosity liquid
        self.nuv=self.muv/self.rhov   # kin. viscosity vapour
        self.Pr=self.nul/(self.kl/(self.rhol*self.cpl))  # Prandtl Number
        self.omega=(self.rpm/60)*2*pi # rotational velocity
        self.alpha=zeros(self.N-1,1) # taper angle
        self.alpha[0:self.Nc-1]=atan(self.Riae-self.Ric)/self.Lc
        self.Ri=self.Ric+[0, cumsum(self.DX*sin(self.alpha))]
        self.Ro=ones(self.N,1)*self.Ro_cae    # outer radius
        self.RI=self.Ri[0:-1]/2 + self.Ri[1:]/2 # mid-value between nodes
        self.RO=self.Ro[0:-1]/2 + self.Ro[1:]/2
        
    def __str__(self):
        """String representation of the RHP object."""
        return (f"RHP Model:\n"
                f"Filling Ratio: TODO\n"
                f"Evaporator Temperature: {self.evaporator_temp} K\n"
                f"Condenser Temperature: {self.condenser_temp} K\n")
        
    def grashof(rim, a, dTm, deltam):
        #GRASHOF
        # Grashof Number at mid-point of FV
        # rim...inner radius at mid-point of FV
        # a...taper angle
        # dTm...temperature difference (Tsat-Twi)
        # deltam...film thickness at mid-point
        Gr=abs( (self.omega^2)*rim*cos(a)*self.betal*dTm*(deltam^3)/(self.nul^2) )
        return Gr
    
    def heat_transfer_model_film_condensation( d0,d1, Two, Tsat, dx,ri,ro, opt_arg=0 ):
        #FILM_CONDENSATION
        # solve liquid flow model for difference in mass flow (liquid) 
        # evaluation at mid-point x=(x0+x1)/2
        # ri/ro...inner/outer radius at xm
        # Twi/Two...inner/outer wall temperature at xm

        # FVM evaluation at midpoint x=x0/2+x1/2
        d=d0/2+d1/2
        ln=ri*log(ro/ri)

        if (d>0):  # there is a liquid layer
            Twi=((self.kl/d)*Tsat+(self.kw/ln)*Two)/(self.kw/ln+self.kl/d)
        else:   # there is no liquid layer
            Twi=Tsat
    
        qw=kw*(Twi-Two)/ln;   # qw is always determined in contrast to ql 
        dmt=qw*2*pi*ri*dx/hfg
        
        return [dmt, Twi, qw]
    
    def eat_transfer_model_mixed_convection( d0,d1, Two, Tsat, dx,ri,ro, a )
        #MIXED_CONVECTION 
        # solve liquid flow model for difference in mass flow (liquid) 
        # used in evaporator, evaluation at x=xm
        # ri/ro...inner/outer radius at xm
        # Twi/Two...inner/outer wall temperature at xm

        Nuf=1  # Nusselt Number of forced convection
        d=d0/2+d1/2
        ln=ri*log(ro/ri)

        # first run with Twi=Two
        Gr=grashof(ri, a, Tsat-Two, d)  # here use Two
        Ra=Gr*Pr
        Nun=0.133*Ra^0.375;     # Nusselt Number of natural convection..
        Num=(Nuf^(7/2)+Nun^(7/2))^(2/7);    # ..and mixed 
        if (d>0)  # there is a liquid layer
            Twi=((Num*kl/d)*Tsat+(kw/ln)*Two)/(kw/ln+Num*kl/d)
        else   # there is no liquid layer
            Twi=Tsat

        # second run with Twi=Twi_first run
        Gr=grashof(ri, a, Tsat-Twi, d);  3 here use Twi 
        Ra=Gr*Pr;
        Nun=0.133*Ra^0.375;     # Nusselt Number of natural convection..
        Num=(Nuf^(7/2)+Nun^(7/2))^(2/7);    # ..and mixed 
        if (d>0)  # there is a liquid layer
            Twi=((Num*kl/d)*Tsat+(kw/ln)*Two)/(kw/ln+Num*kl/d)
        else   # there is no liquid layer
            Twi=Tsat

        # heat flow and mass flow difference
        qw=kw*(Twi-Two)/ln   # qw is always determined in contrast to ql 
        dmt=qw*2*pi*ri*dx/self.hfg
        
        return [dmt, Twi, qw]
    
    def iterate_fv_adiabatic(d0start, d1, mt1, uv1, uld1, Tsat, k)
        #ADIABATIC
        # k=1..N=Nc+Na+Ne (node numbering, there is one more node than FV)
        global  Ri  alpha # wall
        global  DX NUMZERO max_inner_iterations max_restarts # discretization

        dx=DX(k-1) # index of FV
        a=alpha(k-1) # index of FV
        #ri=Ri(k-1)/2 + Ri(k)/2 # mid-point values (x0/2+x1/2)
        #ro=Ro(k-1)/2 + Ro(k)/2 # mid-point values (x0/2+x1/2)
        ri0=Ri(k-1) # index of left node (x=x0)

        K=restart_values(max_restarts)

        # initialize FVM iteration
        d0=d0start
        iteration_count=0
        restart_count=0

        dmt_lf=1; # CHANGE compared to iterate_CE.m % enter iteration
        dmt_ht=0; # CHANGE compared to iterate_CE.m

        while (abs(dmt_lf)>NUMZERO) and (iteration_count<max_inner_iterations) and (restart_count<max_restarts)
            iteration_count=iteration_count+1

            [dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,uld1 )

            fz=(dmt_ht-dmt_lf)   # function to be iterated to zero

            if iteration_count == 1 # not enough values for regula falsi
                if fz>0
                    d0next=(4/5)*d0 # increase to some value
                else
                    d0next=(5/4)*d0 # decrease to some value
                
            else # determine next value from current and previous ones
                d0next=d0-fz*(d0-d0prev)/(fz-fzprev) # regula falsi
            

            d0prev=d0 # shift to next iteration
            if isnan(d0next) or d0next<0:
                restart_count=restart_count+1
                d0=K(restart_count)*d0start
            else
                d0=d0next
            
            fzprev=fz # function value (that should be eiterated to zero)

         dmt=dmt_ht;   # CHANGE compared to iterate_CE.m
        mt0=mt1-dmt;    # mt0+dmt=mt1

        uv0, uld0, _ = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); # this enters next FV

        Tw=Tsat; # some value which does not distort the diagram
        qw=0; # only for uniform return values

        return [d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count]
       
        
    def iterate_fv_condenser(d0start, d1, mt1, uv1, uld1, Tsat, k)
        #CONDENSER
        # k=1..N=Nc+Na+Ne (node numbering, there is one more node than FV)
        
        dx=DX(k-1) # index of FV
        a=alpha(k-1) # index of FV
        ri=Ri(k-1)/2 + Ri(k)/2 # mid-point values (x0/2+x1/2)
        ro=Ro(k-1)/2 + Ro(k)/2 # mid-point values (x0/2+x1/2)
        ri0=Ri(k-1) # index of left node (x=x0)
        # restart correction factors for delta0
        K=restart_values(max_restarts)     # last value not executed

        #initialize FVM iteration
        d0=d0start
        dmt_diff_rel=1 # set to enter iteration
        mt0=-mt1 # set to enter iteration
        restart_count=0
        iteration_count=0 
        while ((dmt_diff_rel>dmt_diff_rel_tol) or (mt0<0)) and (restart_count<max_restarts) or (iteration_count<max_inner_iterations):
            iteration_count=iteration_count+1

            [dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,uld1 )
            [dmt_ht, Tw, qw]=heat_transfer_model_film_condensation( d0,d1,TC,Tsat, dx,ri,ro )
            dmt_diff_rel=abs(dmt_ht-dmt_lf)/(abs(dmt_ht)+abs(dmt_lf))
            fz=(dmt_ht-dmt_lf)

            if iteration_count == 1: # not enough values for regula falsi
                if fz>0:
                    d0next=(4/5)*d0; # increase to some value
                else
                    d0next=(5/4)*d0; # decrease to some value
                
            else # determine next value from current and previous ones
                d0next=d0-fz*(d0-d0prev)/(fz-fzprev) # regula falsi
            
            # shift to next iteration
            d0prev=d0
            fzprev=fz  # function value (that should be eiterated to zero)

            if (d0next<0) or isnan(d0next):
                restart_count=restart_count+1
                if d0start==0:
                    d0=K(restart_count)*d1:
                else:
                    d0=K(restart_count)*d0start:
                dmt= 0 # in order to guarantee a return value
            else:
                if mod(iteration_count,MOD4GEOM)==0: # metaphysical convergence acceleration (works)
                    d0=sqrt(d0next*d0)
                else:
                    d0=d0next
                dmt=dmt_ht #dmt_lf/2+dmt_ht/2;    % this gets important if ht<>lf (not converged)
                mt0=mt1-dmt    # mt0+dmt=mt1

        [uv0,uld0,~] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); % this enters next FV

        return [d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count]      
    
    
    def iterate_fv_evaporator(d0start, d1, mt1, uv1, uld1, Tsat, k):
        # EVAPORATOR
        # k=1..N=Nc+Na+Ne (node numbering, there is one more node than FV)

        dx=DX(k-1)# index of FV
        a=alpha(k-1)# index of FV
        ri=Ri(k-1)/2 + Ri(k)/2# mid-point values (x0/2+x1/2)
        ro=Ro(k-1)/2 + Ro(k)/2# mid-point values (x0/2+x1/2)
        ri0=Ri(k-1)# index of left node (x=x0)
        # restart correction factors for delta0
        K=restart_values(max_restarts)# last value not executed

        # initialize FVM iteration
        d0=d0start
        dmt_diff_rel=1# set to enter iteration
        mt0=-mt1# set to enter iteration
        restart_count=0#
        iteration_count=0#

        while ((dmt_diff_rel>dmt_diff_rel_tol) or (mt0<0)) and (restart_count<max_restarts) and (iteration_count<max_inner_iterations):
            iteration_count=iteration_count+1
            
            #[dmt_ht, Tw, qw]=heat_transfer_model_film_condensation( d0,d1,TE,Tsat, dx,ri,ro )
            [dmt_ht, Tw, qw]=heat_transfer_model_mixed_convection( d0,d1,TE,Tsat, dx,ri,ro, a )
            #[dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,uld1 )
            [dmt_lf, b_lf]=liquid_flow_model_mixed_convection( d0,d1,mt1, dx,a,ri0, uv1,uld1, Tw, Tsat )
            
            dmt_diff_rel=abs(dmt_ht-dmt_lf)/(abs(dmt_ht)+abs(dmt_lf))
            
            fz=(dmt_ht-dmt_lf);   #log(abs(dmt_ht/dmt_lf)); % function to be iterated to zero
            if iteration_count == 1: # not enough values for regula falsi
                if fz>0:
                    d0next=(4/5)*d0 # increase to some value
                else:
                    d0next=(5/4)*d0 # decrease to some value
            else: # determine next value from current and previous ones
                d0next=d0-fz*(d0-d0prev)/(fz-fzprev)# regula falsi
            
            # shift to next iteration
            fzprev=fz# function value (that should be eiterated to zero)
            d0prev=d0  
            
            if (d0next<0) or isnan(d0next)
                restart_count=restart_count+1
                if d0start==0:
                    d0=K(restart_count)*d1
                else:
                    d0=K(restart_count)*d0start
                dmt=0# in order to guarantee a return value
            else:
                d0=d0next
                dmt=dmt_ht#dmt_lf/2+dmt_ht/2;    % this gets important if ht<>lf (not converged)
                mt0=mt1-dmt# mt0+dmt=mt1

        [uv0,uld0,~] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); # this enters next FV

        return [d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count]

    def liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv,uld, op1=0, op2=2 ):
        #FILM_CONDENSATION
        # solve liquid flow model for difference in mass flow (liquid) 
        # evaluation at begin x=x0 (left) of FV using uv and uld from x1 (right)

        global rhol  mul# liquid
        global omega # wall

        # solve for dmt=Z/N, keep in mind mt1=mt0+dmt

        Z1=mt1-(rhol/mul)*(omega^2)*ri0*(sin(a)-cos(a)*(d1-d0)/dx)*(1/3)*(d0^3)*2*pi*ri0*rhol
        N1=1-(1/2)*((d0^2)/(mul*dx))*(uv*cos(a)+uld)*2*pi*ri0*rhol
        dmt=Z1/N1

        b=2
        
        return [dmt, b]
    
    
    def liquid_flow_model_mixed_convection( d0,d1,mt1, dx,a,ri0, uv1, uld1, Twi, Tsat ):
        #MIXED_CONVECTION 
        # solve liquid flow model for difference in mass flow (liquid) 
        # evaluation at begin (x=x0) of FV using uv and uld from x1 
        # this convection model is assumed for evaporator
        # d0...delta left
        # d1...delta right
        # mt1...mass flow in (right)
        # dx...FV length
        # a...taper angle
        # ri0...inner radius (left)
        # uv1...vapour velocity (right)
        # uld1...liquid velocity at vapour interface (right)
        # right side velocities determine characteristic numbers and thus flow regime

        global rhol  mul nul Pr# liquid
        global rhov nuv# vapour
        global omega# wall

        rim=ri0+(dx/2)*sin(a)
        ri1=ri0+dx*sin(a)
        dTm=Tsat-Twi
        deltam=(d0+d1)/2
        Av=pi*(rim-deltam)^2# cross-sectional area for vapour flow
        dmt=0# initial guess, only needed to estimate liquid velocities (remember: dmt=mt1-mt0)
        b_forced=2# assume forced convection for estimating u_avg
        _,_,u_avg = liquid_velocities(d0, d1, mt1+dmt, dmt, ri0, dx, a, b_forced)

        # characteristic numbers (fixed values at mid-point, iterated values right)
        Gr=grashof(rim, a, dTm, deltam)  
        Re=reynolds(deltam, u_avg)# u_avg is at x1 
        Ra=Gr*Pr
        GrRe2=Gr/(Re^2)
        # velocity profile power law
        if Ra<1e9:
            if GrRe2<0.1:      # forced convection
                b=2
            else:
                if GrRe2<10:     # mixed convection
                    b=1/3
                else:
                    b=1/4
        else:
            b=1/7      # natural convection
        
        # wall friction [Song -> Afzal & Hussain]
        if Re>0:
            K=Gr/(Re^(5/2))
            if K<1:
                Cfw=0.5:
            else
                Cfw=0.5*K^(3/5):
        else
            Cfw=0:

        # vapour friction [Daniels]
        Rev=uv1*Av/nuv
        if Rev>0:
            if Rev<2000:
                Cfd=16/Rev
            else:
                Cfd=0.0791/(Rev^0.25)
        else:
            Cfd=0

        # top velocity from mass flow at x=x1
        if d1>0:
            U1=mt1*(b+1)/(2*pi*ri1*rhol*d1)
        else:
            U1=0# no liquid, no velocity
        
        # momentum bilance
        QP = 0.4e1 * ((b + 1) * Av ^ 2 * ((Cfw * dx * b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + (Cfw * dx) / 0.2e1) * rhov + 0.4e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2) * U1 / (0.2e1 * (b + 1) * ((Cfw * dx * b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + (Cfw * dx) / 0.2e1 + 0.2e1 * deltam) * Av ^ 2 * rhov + 0.8e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2)
        QQ = (-0.16e2 * rim * Av ^ 2 * omega ^ 2 * ((b + 1) ^ 2) * (b + 0.1e1 / 0.2e1) * rhov * (-d1 + d0) * deltam * cos(a) - 0.16e2 * rim * Av ^ 2 * omega ^ 2 * dx * ((b + 1) ^ 2) * (b + 0.1e1 / 0.2e1) * rhov * deltam * sin(a) + 0.2e1 * (Av ^ 2 * (Cfw * dx * (b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + Cfw * dx / 0.2e1 - 0.2e1 * deltam) * (b + 1) * rhov + 0.4e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2) * U1 ^ 2) / (0.2e1 * (b + 1) * (Cfw * dx * (b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + Cfw * dx / 0.2e1 + 0.2e1 * deltam) * Av ^ 2 * rhov + 0.8e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2)


        DISCRIMI=(QP/2)^2-QQ
        if DISCRIMI>=0:
            U0=(-QP/2)+sqrt(DISCRIMI); # prefer higher value (TODO check)
        else
            U0=0
            print('No real solution for U0 in liquid_flow_model_mixed_convection (evaporator).')

        # top velocity from mass flow at x=x1 and mass flow difference
        mt0=U0*(2*pi*ri0*rhol*d1)/(b+1)
        dmt=mt1-mt0

        U0=0.07461# result from film condensation

        # debug info
        tauv = Cfd * rhol ^ 2 * pi ^ 2 * rim ^ 2 * (U0 + U1) ^ 2 * deltam ^ 2 / rhov / ((b + 1) ^ 2) / Av ^ 2 / 0.2e1
        tauw = Cfw * rhol * (U0 + U1) ^ 2 / 0.8e1
        Il = omega ^ 2 * (sin(a) * dx + (-d1 + d0) * cos(a)) * rim * deltam / dx
        Ir = -((U0 + U1) * ((-d1 + d0) * (U0 + U1) * b - 2 * deltam * (U0 - U1)) / dx / (2 * b ^ 2 + 3 * b + 1)) / 0.4e1

        return [dmt, b]
    
    
    def liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b):
        #LIQUID_VELOCITIES 
        # velocity of vapour and liquid top layer
        # at left side of given FVM (forward FD)
        # a...taper angle alpha
        # b...power law exponent
        # uv0...vapour velocity at x=x0
        # uld0...liquid velocity at vapour interface at x=x0
        # u_avg0...average liquid velocity at x=x0

        # vapour velocity
        uv0=mt0/(rhov*pi*ri0^2)# stationary: mass flow forward (liquid)=mass flow backward (vapour)

        # liquid flow model (may contain pole)
        #T0=(rhol/mul)*(sin(a)*cos(a)*(d1-d0)/dx)*(1/2)*d0^2;
        #T1=(d0/mul)*(dmt/dx);
        #uld0=(T0-T1*uv0*cos(a))/(1+T1);

        # liquid velocity
        Vt0=mt0/rhol
        u_avg0=Vt0/(2*pi*ri0*d0) # average
        uld0=(b+1)*u_avg0 # vapour interface
        
        return [uv0, uld0, u_avg0]
    
    def liquid_volume(d,RI):
        #LIQUID_VOLUME
        # d...film heigth in condenser (nodal)
        # RI...inner radius (mid-point)

        deltamid=d(1:end-1)/2 + d(2:end)/2
        V= 2*pi*sum( deltamid.*DX.*RI)

        return V
    
    
    def look_up_song(d2D):
        #LOOK_UP_SONG 
        # liquid mass as function of film height at evaporator end cap
        # [Song 2003]

        d2D_data=[0, 0.011, 0.019, 0.025, 0.036]
        ml0=0.7e-3
        ml_data=[1, 3, 5, 7, 10]*ml0

        ml=interp1(d2D_data, ml_data, d2D)

        return ml
    
    
    def main_iterations():
        [~]=set_global_variables(1)

        Nd=1 # 21 %discretization levels 
        # main operational parameters
        #dEend2D=0.0 # fluid height/diameter at evaporator end (=0 -> minimal fluid loading)
        dEend2Dmin=0.000
        dEend2Dmax=0.0058595# 0.035 (this value is chosen for Nd=1)

        meanRi=mean(Ri)
        meanRic=mean(Ri(1:Nc))

        L=Lc+La+Le # total length
        Di=2*Riae


        # initial guess for first run (next runs take previous results as guess)
        #Tsat_prev=(1/2)*(TC+TE); % initial guess for vapour temperature
        dEendmin=(2*meanRi)*dEend2Dmin# minimal value of fluid height at E (typically zero))
        ml=look_up_song(dEend2Dmin)# look up from [Song2003]
        delta_konst=((ml/rhol)/(2*pi*meanRi)-dEendmin*Le/2)/(Lc/2+La+Le/2)
        dc0=linspace(0,delta_konst,Nc).'
        da0=ones(Na-2,1)*delta_konst# because first and last node are in dc/de
        de0=linspace(delta_konst,dEendmin,Ne).'
        delta0=[dc0; da0; de0]
        #V0=liquid_volume(delta0,RI)
        #m0=rhol*V0

        # run numerical solution (loop: dEend, Tsat, rhp-FVs, FV-dmt)
        disp(['-- RUNNING film condensation/evaporation model with N=',num2str(N),' finite volumes --'])
        fileID = fopen('local_iterations.log','w')

        dEend2Dinput=linspace(dEend2Dmin,dEend2Dmax,Nd)# vector of dEend2D values
        delta_results=zeros(N,Nd)# nodal
        mt_results=zeros(N,Nd)# nodal
        GrRe2_results=zeros(N-1,Nd)# FV
        Tw_results=zeros(N-1,Nd)# FV
        qc_results=zeros(Nd,1)# heat flux through inner condenser wall
        V_results=zeros(Nd,1)# total liquid volume
        Tsat_results=zeros(Nd,1)# total liquid volume
        Tsat0=(TC+TE)/2  # in order to have a maximum for Tsat_range in first run

        for kk=1:Nd: # loop film heigths at evaporator end
            
            # changing parameter
            dEend2D=dEend2Dinput(kk)
            dEend=(2*meanRi)*dEend2D
            disp(' ')
            disp(['dEend/D=', num2str(dEend2D)])
            fprintf(fileID, 'dEend/D=%6.6f \n', dEend2D)

            delta, mt, Tw, GrRe2, V, Tsat_ss, qc, knc, count_converged, Tsat_v, mtC_rel_v, QCE_rel_v, index_converged, index_diverged = rhp_outer_loop(dEend, Tsat0, delta0, fileID)
            
            # store results for one value of dEend
            delta_results(:,kk)=delta
            mt_results(:,kk)=mt
            Tw_results(:,kk)=Tw
            GrRe2_results(:,kk)=GrRe2
            V_results(kk)=V
            Tsat_results(kk)=Tsat_ss
            qc_results(kk)=qc
            fprintf(fileID, '\n')
            
            if (knc>0) || (count_converged<2):
                disp('_')
                disp(['WARNING, solution not converged for DEend/D=', num2str(dEend2D)])
                disp(['count_converged=',num2str(count_converged),';'])
                disp('DEBUG INFO'); disp(' ')
                disp(['knc=',num2str(knc),';'])
                if (knc>1): # copy & paste to debug.m
                    disp(['mt1=',num2str(mt(knc),10),';'])
                    disp(['d1=',num2str(delta(knc),10),';'])
                    disp(['Tsat=',num2str(Tsat_ss,10),';']) # value for final run
                
                break  # no hope to try higher values of dEend
            else:
                #        disp(['deltaC/max(delta)=',num2str(delta(1)/max(delta))])# should be zero
                disp(['Tsat=',num2str(Tsat_ss),'°C   liquid mass m=', num2str(V*rhol*1000),' g','   heat flux qc=',num2str(qc),' W/m^2'])# should be zero
            
            delta0=delta
            Tsat0=Tsat_ss
        
        # postprocessing 
        figure
        plot(Tsat_v(index_converged),    QCE_rel_v(index_converged),   'r.',...
            Tsat_v(index_diverged), QCE_rel_v(index_diverged),'ro',...
            Tsat_v(index_converged),    mtC_rel_v(index_converged),   'b.',...
            Tsat_v(index_diverged), mtC_rel_v(index_diverged),'bo',...
            Tsat_ss,0, 'k*') 
        xlabel('T_{sat}'); ylabel('ERROR'); % (con-/divergence)
        %title('last value for dEend: root finding for Tsat')
        if count_converged>0:
            legend('Q con', 'Q div','mt con', 'mt div','Tsat ss')
        else:
            legend('Q div','mt div','Tsat ss')
        
        return 0
        
        
    def restart_values(N):
            #RESTART_VALUES 
            # generate list of factors

            K=zeros(N+1,1)# last value must be zero, but is not used for iterations
            N2=floor(N/2)
            N21=N2+1

            for k=1:N2:
                K(2*k-1)=(N21-k)/N21
                K(2*k)=N21/(N21-k)

            # some drastic values last
            # K(end-2)=1/100;
            # K(end-1)=100;

            return K
        

    def reynolds(d,U):
        #REYNOLDS 
        # Reynolds Number of liquid flow
        # d(elta)...film thickness
        # U...average liquid velocity
        
        Re=abs( 4*U*d/nul )
        return Re


    def rhp_inner_loop(dEend, Tsat, delta0):
        #RHP_INNER_LOOP 
        #   iterate equations for each FV from evaporator end to condenser end
        
        knc=N# first non-converged FV (from end)
        QE=0# heat flow in Evaporator (<0)
        QC=0# heat flow in Condenser (>0)

        FViterations=zeros(N-1,1)# count iterations (per FV)
        FVrestarts=zeros(N-1,1)# count iterations (per FV)
        Tw=zeros(N-1,1)# inner wall temperature (per FV)
        delta=zeros(N,1)# fluid film height (per node)
        mt=zeros(N,1)# mass flow over node (left to right)
        uld=zeros(N,1)# mass flow over node (left to right)
        uv=zeros(N,1)# mass flow over node (left to right)

        diffd0=diff(delta0)
        delta(N)=dEend# prescribed and thus determining volume of liquid
        mt(N)=0# no flow though end plate
        uld(N)=0# no liquid velocity at end plate
        uv(N)=0# no vapour velocity at end plate
        k=N
        while k>1:   # E -> C instead of for-loop in order to allow for early termination
            d1=delta(k)
            mt1=mt(k)
            uld1=uld(k)
            uv1=uv(k)
            if k>=Nc+Na: # evaporator
                d0start= max(d1-diffd0(k-1), delta0(k-1))# diff=d1-d0
                [d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count] = iterate_fv_evaporator(d0start, d1, mt1, uv1, uld1, Tsat, k)
                Tw(k-1)=Twi# inner wall temp. per FV
                QE=QE+qw*2*pi*RI(k-1)*DX(k-1)
            else if k>Nc: # adiabatic
                d0start=d1
                [d0, mt0, uv0, uld0, Twi, ~, inner_iteration_count, restart_count] = iterate_fv_adiabatic(d0start, d1, mt1, uv1, uld1, Tsat, k)
                Tw(k-1)=Twi; # some value which does not distort the diagram
            else # condenser
                d0start=min(delta0(k-1)*delta(Nc)/delta0(Nc), d1)
            d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count = iterate_fv_condenser(d0start, d1, mt1, uv1, uld1, Tsat, k)
                Tw(k-1)=Twi# inner wall temp. per FV
                QC=QC+qw*2*pi*RI(k-1)*DX(k-1)
            
            FViterations(k-1)=inner_iteration_count
            FVrestarts(k-1)=restart_count
            delta(k-1)=d0
            mt(k-1)=mt0
            uld(k-1)=uld0
            uv(k-1)=uv0# for next FV
            
            # pick mass flow at condenser end (should be zero)
            if k==2:
                mtC=mt0
            
            # if inner iterations do not converge..
            if ((inner_iteration_count==max_inner_iterations) or (restart_count==max_restarts)):
                if (k>2) and (k<N):
                    dmt=mt(k+1)-mt(k)# use converged values (prev.FV) for extrapolation
                    if (abs(dmt)>NUMZERO): # prevent diff by zero
                        mtC=mt0-X(k)*dmt/(X(k+1)-X(k))# extrapolate mt at Condenser plate
                    else: # if there were div0
                        mtC=mt0
                    
                else: # first or last FV
                    mtC=mt0
                
                delta(k-1)=0# for diagram beauty only (non-converged)
                knc=k# store non-converged FV
                k=1# stop FV evaluations from this point
            
            k=k-1
        

        if ((inner_iteration_count<max_inner_iterations) and (restart_count<max_restarts)):
            knc=0

        return [QC, QE, mtC, delta, mt, uld, uv, Tw, d0start, knc, FViterations, FVrestarts]


    def rhp_outer_loop(dEend, Tsat0, delta0, fileID):
        #RHP_OUTER_LOOP
        # iterate over Tsat until physical plausible state (no flow through wall)
        # knc...first non-converged FV (from end) in last inner iteration
        # count_converged...number of completely converged inner iterations
        
        Ac=sum(2*pi*RI(1:Nc-1).*DX(1:Nc-1))# for heat flux qc at condenser

        # generate Tsat-grid (minimum to maximal possible value)
            %Tsat_v=linspace(TC+NUMZERO,Tsat_ss-NUMZERO, max_outer_iterations).'# Tsat_ss from previous run bounds meaningful range
            LOGN=(1-logspace(-2,0,max_outer_iterations))*100/99
            Tsat_v=TC+NUMZERO+LOGN*(Tsat0-TC)
            knc_v=zeros(max_outer_iterations,1)# store index of non-converged FV for each Tsat (allows to extract converged results)
            mtC_rel_v=zeros(max_outer_iterations,1)# for heat flow "through condenser wall" as criteria
            QCE_rel_v=zeros(max_outer_iterations,1)# for heat bilance as termination criteria

        # evaluate Tsat-grid    
        for outer_iteration_count=1:max_outer_iterations: # evaluate heat pipe (all FVs) to find Tsat
            Tsat=Tsat_v(outer_iteration_count)
            QC, QE, mtC, _, mt, _, _, _, _, knc, _, _ = rhp_inner_loop(dEend, Tsat, delta0)
            QCE=(QC+QE)
            QCE_rel=QCE/(abs(QC)+abs(QE))
            QCE_rel_v(outer_iteration_count)=QCE_rel# for heat bilance as termination criteria
            mtC_rel=mtC/max(mt)
            mtC_rel_v(outer_iteration_count)=mtC_rel# for heat flow "through condenser wall" as criteria
            knc_v(outer_iteration_count)=knc
        
        # determine Tsat
        # inter/extrapolate steady state Tsat from converged results 
        index_converged=find(knc_v==0)
        index_diverged=find(knc_v>0)
        count_converged=length(index_converged)
        if count_converged>2:
            Tsat_ss=interp1(mtC_rel_v(index_converged), Tsat_v(index_converged), mtC_rel_tol, 'spline')# slightly positive zero (for non-positive value problems on last FV)
        else if count_converged>0:
            Tsat_ss=interp1(mtC_rel_v(index_converged), Tsat_v(index_converged), mtC_rel_tol, 'linear','extrap')# slightly positive zero (for non-positive value problems on last FV)
    else:
            Tsat_ss=min(Tsat_v(knc_v==min(knc_v)))# in order to have some values, take the run where k_non_converged is minimal

        # final run (interpolated Tsat)
        QC, QE, mtC, delta, mt, uld, _, Tw, _, knc, FViterations, FVrestarts = rhp_inner_loop(dEend, Tsat_ss, delta0)
        QCE=(QC+QE)
        QCE_rel=QCE/(abs(QC)+abs(QE))
        mtC_rel=mtC/max(mt)

        # convergence info
        fprintf(fileID, 'mtC_rel=%1.12f   QCE_rel=%1.12f   deltaC/max(delta)=%4.2f   QC=%4.2f QE=%4.2f   converged=%d/%d \n', mtC_rel, QCE_rel, delta(1)/max(delta), QC, QE, count_converged, max_outer_iterations)
        fprintf(fileID, 'inner iterations (final run): mean=%4.2f   min=%d   max=%d    (max_inner_iterations=%d) \n', mean(FViterations(FViterations>0)), min(FViterations(FViterations>0)), max(FViterations), max_inner_iterations)
        if (max(FVrestarts)>0):
            fprintf(fileID, 'inner restarts (final run): mean=%4.2f   min=%d   max=%d   (max_restarts=%d) \n', mean(FVrestarts(FViterations>0)), min(FVrestarts(FViterations>0)), max(FVrestarts), max_restarts)
        else:
            fprintf(fileID, 'inner restarts (final run): 0 restarts \n')

        # results
        V=liquid_volume(delta, RI)
        mt_max=max(mt)
        Q=mt_max*hfg# maximum flow occurs in adiabatic section end determines transfered heat
        qc=Q/Ac# inner condenser wall
        fprintf(fileID, 'Tsat_ss=%3.6f°C   ml=%3.6f g   max(mlt)=%3.6f kg/s   qc=%3.6f W/m^2 \n\n', Tsat_ss, V*rhol*1000, mt_max, qc)

        # prepare plots to compare with Song
        GrRe2=zeros(N-1,1);
        for k=1:N-1 % per FV
            Um=(1/2)*(uld(k)+uld(k+1))
            deltam=(1/2)*(delta(k)+delta(k+1))
            Gr=grashof(RI(k),alpha(k),Tw(k)-Tsat_ss, deltam)
            Re2=(reynolds(deltam,Um))^2
            if Re2>0:
                GrRe2(k)=Gr/Re2
            else
                GrRe2(k)=0
            


song_rhp = RHP(evaporator_temp=100, condenser_temp=60)