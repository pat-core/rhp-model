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
        
    def calculate_heat_transfer(self):
        """Calculate heat transfer rate in the evaporator and condenser."""
        # Implementation to be added
        pass
    
    def calculate_cop(self):
        """Calculate Coefficient of Performance."""
        # Implementation to be added
        pass
    
    def calculate_pressure_drop(self):
        """Calculate pressure drop in the system."""
        # Implementation to be added
        pass
    
    def run_simulation(self):
        """Run the complete RHP simulation."""
        self.calculate_heat_transfer()
        self.calculate_cop()
        self.calculate_pressure_drop()
        return {
            'heat_transfer_rate': self.heat_transfer_rate,
            'cop': self.cop,
            'mass_flow_rate': self.mass_flow_rate,
            'pressure_evap': self.pressure_evap,
            'pressure_cond': self.pressure_cond
        }
    
    def __str__(self):
        """String representation of the RHP object."""
        return (f"RHP Model:\n"
                f"Working Fluid: {self.working_fluid}\n"
                f"Evaporator Temperature: {self.evaporator_temp} K\n"
                f"Condenser Temperature: {self.condenser_temp} K\n"
                f"COP: {self.cop}")


song_rhp = RHP(evaporator_temp=100, condenser_temp=60)