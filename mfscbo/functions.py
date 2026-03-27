import subprocess
import torch
import numpy as np  
import os
import time


class Hartmann6:
    """Hartmann6 function for different fidelities."""

    def __init__(self, fidelity, fidelity_high, torchargs):
        self.bounds = torch.tensor([[0.0] * 6, [1.0] * 6], **torchargs)
        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], **torchargs)
        self.delta = torch.tensor([0.01, -0.01, -0.1, 0.1], **torchargs)
        self.s = fidelity
        self.S = fidelity_high
        self.torchargs = torchargs


        self.A = torch.tensor(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ],
            **torchargs
        )

        self.P = 0.0001* torch.tensor(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ],
            **torchargs
        )


    def __call__(self, x):
        """Evaluate the Hartmann6 function.

        Args:
            x (torch.Tensor): Points to evaluate of shape (n_pts, dim)

        Returns:
            torch.Tensor: Evaluated points of shape (n_pts, 1)
        """

        assert torch.all(x >= 0.0) and torch.all(x <= 1.0) and torch.all(torch.isfinite(x)), f"All input points must be in [0, 1]^{self.dim} and finite."

        x = x.to(**self.torchargs)

        inner_sum = torch.sum(self.A * (x.unsqueeze(1) - self.P) ** 2, dim=2)
        coeff_fidelity = self.alpha + (self.S - self.s) * self.delta
        value = -torch.sum(coeff_fidelity * torch.exp(-inner_sum), dim=1)

        return value.unsqueeze(-1)


class Borehole8 :
    """Borehole8 function for different fidelities.
    """

    def __init__(self, fidelity, torchargs):
        self.bounds = torch.tensor([[0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0],
                                    [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0]], **torchargs)
        self.s = fidelity
        self.torchargs = torchargs
    
    def __call__(self, x):

        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x)), f"All input points must be in the bounds and finite."

        x = x.to(**self.torchargs)

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        x7 = x[:, 6]
        x8 = x[:, 7]

        if self.s == 1.0 :
            return 2*torch.pi * x3 * (x4 - x6) / (torch.log(x2/x1) * (1 + 2*x7*x3/(torch.log(x2/x1)*x1**2*x8) + x3/x5))
        elif self.s == 0.0 :
            return 5*x3*(x4 - x6) / (torch.log(x2/x1) * (1.5 + 2*x7*x3/(torch.log(x2/x1)*x1**2*x8) + x3/x5))



class Rastrigin :
    """Rastrigin function for different fidelities in dimension d.
    """

    def __init__(self, dim, fidelity, fidelity_high, torchargs):
        self.bounds = torch.tensor([[-5] * dim, [5] * dim], **torchargs)
        self.dim = dim
        self.s = fidelity
        self.S = fidelity_high
        self.torchargs = torchargs

    def __call__(self, x):

        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x)), f"All input points must be in [-5.12, 5.12]^{self.dim} and finite."

        x = x.to(**self.torchargs)

        A = (0.2 + 0.8 * self.s / self.S)*10.0
        
        value = 10 * self.dim * A + torch.sum(x**2 - A * torch.cos(2*torch.pi*x), dim=1)
        noise = (self.S - self.s) * 0.1 * torch.sin(5 * torch.pi * x).sum(dim=1)

        return (value + noise)
    

class Ackley : 
    """Ackley function for 2 different fidelities in dimension d.
    """
    
    def __init__(self, dim, fidelity, torchargs):
        self.bounds = torch.tensor([[-5] * dim, [10] * dim], **torchargs)
        self.dim = dim
        self.s = fidelity
        self.torchargs = torchargs

    def _high_fid(self, x) :
        a = 20.0
        b = 0.2
        c = 2 * torch.pi

        sum1 = torch.sum(x**2, dim=1)
        sum2 = torch.sum(torch.cos(c * x), dim=1)

        term1 = -a * torch.exp(-b * torch.sqrt(sum1 / self.dim)) 
        term2 = -torch.exp(sum2 / self.dim) 

        value = term1 + term2 + a + torch.exp(torch.tensor(1.0, **self.torchargs))
        
        return value

    def _low_fid(self, x) :
        
        sum1 = torch.sum(1.1*x**2, dim=1)
        sum2 = torch.sum(torch.cos(2*torch.pi * x), dim=1)

        term1 = -22 * torch.exp(-0.2 * torch.sqrt(sum1 / self.dim)) 
        term2 = -0.9*torch.exp(sum2 / self.dim) 

        value = term1 + term2 + 20 + torch.exp(torch.tensor(1.0, **self.torchargs))
        
        return value
    
    def __call__(self, x):
        
        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x)), f"All input points must be in [-32, 32]^{self.dim} and finite."
        
        x = x.to(**self.torchargs)

        if self.s == 2 :
            return self._high_fid(x)
        elif self.s == 1 :
            return self._low_fid(x)

   
class Rosenbrock :
    """Rosenbrock function for 3 differents fidelities in dimension d.
    """
    
    def __init__(self, dim, fidelity, torchargs):
        self.bounds = torch.tensor([[-2.0] * dim, [2.0] * dim], **torchargs)
        self.dim = dim
        self.s = fidelity
        self.torchargs = torchargs

        assert fidelity in {1, 2, 3}, f"Fidelity must be 1, 2 or 3. Here fidelity = {fidelity}."
    
    def _high_fid(self, x) :
        sum1 = torch.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, dim=1)
        return sum1
    
    def _mid_fid(self, x) :
        sum1 = torch.sum(50.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (-2 - x[:, :-1])**2, dim=1)
        sum2 = - torch.sum(0.5* x, dim=1)
        return sum1 + sum2

    def _low_fid(self, x) :
        sum1 = torch.sum(x, dim=1)
        return (self._high_fid(x) -4 - 0.5 * sum1 )/ (10 + 0.25 * sum1)

    def __call__(self, x):

        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x)), f"All input points must be in [-5, 10]^{self.dim} and finite."

        x = x.to(**self.torchargs)

        if self.s == 3 :
            return self._high_fid(x)
        elif self.s == 2 :
            return self._mid_fid(x)
        elif self.s == 1 :
            return self._low_fid(x)






class Solar : 
    """Solar benchmark from https://github.com/bbopt/solar. Need the compilation before using !"""

    def __init__(self, id_pb, exe, fidelity, directory, torchargs, seed=0, rep=1) :
        """Initialize the Solar benchmark.

        Args:
            id_pb (str) : problem identity. Should be '2' or '7'.
            exe (str) : path to executable file of solar.
            fidelity (float): fidelity level. Between [0,1].
            directory (str): directory where to run solar and save results.
            torchargs (torch.Tensor): torch arguments.
            seed (int, optional): Seed for stochastic outputs. Defaults to 0.
            rep (int, optional): Repetition. Defaults to 1.
        """

        self.exe = exe
        self.s = fidelity
        self.directory = directory
        self.torchargs = torchargs
        self.seed = seed
        self.rep = rep
        self.id_pb = id_pb

        assert self.id_pb == '2' or self.id_pb == '7', f"Problem identity should be 2 or 7, here {self.id_pb}."
        if self.id_pb == '2' :
            #[1,40]x[1,40]x[20,250]x[1,30]x[1,30]x[1,89]x[0,20]x[1,20]x[793,995]x[0.01,5]x[0.005, 1]x[0.005,0.1]
            self.bounds = torch.tensor([[1, 1, 20, 1, 1, 1, 0, 1, 793, 0.01, 0.005, 0.005], [40, 40, 250, 30, 30, 89, 20, 20, 995, 5, 0.1, 0.1]], **torchargs)
            self.number_obj = 1
            self.number_const = 12
            #fix discrete variables
            self.x6 = 2650
            self.x11 = 36
        elif self.id_pb == '7' :
            #[1.0, 30]x[1.0, 30]x[793.0, 995.5]x[0.01, 5.0]x[0.005, 0.1]x[0.0055, 0.1]
            self.bounds = torch.tensor([[1.0, 1.0, 793.0, 0.01, 0.005, 0.0055], [30.0, 30.0, 995.5, 5.0, 0.1, 0.1]], **torchargs)
            self.number_obj = 1
            self.number_const = 6
            #fix discrete variables
            self.x4 = 40

        self._create_directory()

    def _create_directory(self) :
        """Creates the directory if it doesn't exist."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
    
    def _run_solar(self, x_txt, verbose=False):
        """Run the solar executable with the given input file.

        Args:
            x_txt (str): Path to the input file.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Returns:
            str: The output of the solar executable.
        """

        cmd = [
            self.exe,
            str(self.id_pb),
            str(self.directory + "/" + x_txt),
            f'-seed={self.seed}',
            f'-fid={self.s}',
            f'-rep={self.rep}',
            f'-o={str(self.directory + "/results.txt")}'
        ]
        
        if verbose:
            cmd.append('-v')
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error running solar: {e}")
            print("Stderr:", e.stderr)
            return None

    def _write_input(self, x : torch.Tensor) :
        """Writes the input tensor to a text file.
        Args:
            x (torch.Tensor): Input tensor to write to the file.
        """
        x_str = x.cpu().numpy().tolist()
        with open(self.directory + "/x.txt", "w") as text_file:
            text_file.write(" ".join(map(str, x_str)))
    
    def _get_solar_outputs(self, p,m):
        """
        Reads the 'results.txt' file and extracts objective and constraint values.

        Args:
            p (int): Number of objective values to extract
            m (int): Number of constraint values to extract
        
        Returns:
            tuple: A tuple containing two numpy arrays:
                - objectives (np.ndarray): Array of objective values of shape (p,)
                - constraints (np.ndarray): Array of constraint values of shape (m,)
        """
        text_file = open(self.directory + "/results.txt", "r")
        values = text_file.read().split(' ')
        values_in_float = [float(v) for v in values if v.strip() != '']
        values_in_float = np.array(values_in_float)
        _ = text_file.close()

        objectives = values_in_float[0:p]
        constraints = values_in_float[p:p+m]
        return torch.tensor(objectives, **self.torchargs), torch.tensor(constraints, **self.torchargs)

    def __call__(self, x):

        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x))

        obj_value = torch.zeros(x.shape[0], **self.torchargs)
        constr_value = torch.zeros((x.shape[0], self.number_const), **self.torchargs)

        for i in range(x.shape[0]) :
            #write x in a x.txt file
            if self.id_pb == '2':
                xi_prime = torch.zeros(x.shape[1]+2, **self.torchargs)
                xi_prime[0:5] = x[i,0:5].clone()
                xi_prime[5] = self.x6
                xi_prime[6:10] = x[i, 5:9].clone()
                xi_prime[10] = self.x11
                xi_prime[11:] = x[i, 9:].clone()
            elif self.id_pb == '7':
                xi_prime = torch.zeros(x.shape[1]+1, **self.torchargs)
                xi_prime[0:3] = x[i, 0:3].clone()
                xi_prime[3] = self.x4
                xi_prime[4:] = x[i, 3:].clone()
            
            self._write_input(xi_prime)

            #run_solar
            self._run_solar('x.txt')
            obj, constr = self._get_solar_outputs(self.number_obj, self.number_const)

            obj_value[i] = obj
            constr_value[i,:] = constr
        
        return obj_value, constr_value



class Airfoils :
    """Airfoils benchmark by using the XFOIL software (https://web.mit.edu/drela/Public/web/xfoil/)
    and PARSEC.
    Parameter \Deltay_{TE} is fixed equal to 0. So doesn't appear in the optimization bounds.
    """


    def __init__(self, fidelity, directory, torchargs):
        self.fidelity = fidelity
        self.directory = directory
        self.torchargs = torchargs
        self.dim = 10
        bounds_inf = [0.005, 0.25, 0.05, -1, 0.35, -0.12, 0.3, -0.02, -8, 4]
        bounds_sup = [0.06, 0.5, 0.15, -0.4, 0.5, -0.04, 1, 0.02, -3, 8]
        self.bounds = torch.tensor([bounds_inf, bounds_sup], **self.torchargs)

        self.MACH  = [0.5, 0.55, 0.5, 0.55]
        self.ALPHA = [2, 2.2, 2.5, 2]
        self.RE = [6.3e6, 6.3e6, 6.3e6, 6.3e6]
        self.W = [0.4, 0.2, 0.2, 0.2]

        self._create_directory()

    def _create_directory(self) :
        """Creates the directory if it doesn't exist."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)


    def _cmatrix(self, x) :
        """Computes C matrices for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: C matrix of shape [batch_size, 6, 6]
        """

        c = torch.zeros((x.shape[0], 6, 6), **self.torchargs)
        c[:, 0, :] = 1.0
        for i in range(6) :
            c[:, 1, i] = x**((2*i+1)/2)
            c[:, 2, i] = (2*i+1)/2 
            c[:, 3, i] = (2*i+1)/2 * x**((2*i-1)/2)
        c[:, 4, 0] = -1/4 * x**(-3/2)
        c[:, 4, 1] = 3/4 * x**(-1/2)
        c[:, 4, 2] = 15/4 * x**(1/2)
        c[:, 4, 3] = 35/4 * x**(3/2)
        c[:, 4, 4] = 63/4 * x**(5/2)
        c[:, 4, 5] = 99/4 * x**(7/2)
        c[:, 5, 0] = 1.0
        return c


    def _cupper(self, x) :
        """Computes C upper matrix for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: C upper matrix of shape [batch_size, 6, 6]
        """
        x2 = x[:, 1].clone()
        return self._cmatrix(x2)
    
    def _clower(self, x) :
        """Computes C lower matrix for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: C lower matrix of shape [batch_size, 6, 6]
        """
        x5 = x[:, 4].clone()
        return self._cmatrix(x5)
    
    def _bupper(self, x) :
        """Computes b upper vector for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: b upper vector of shape [batch_size, 6]
        """
        x1 = x[:, 0].clone()
        x3 = x[:, 2].clone()
        x4 = x[:, 3].clone()
        x8 = x[:, 7].clone()
        x9 = x[:, 8].clone()
        #in radians
        x10 = x[:, 9].clone() * torch.pi / 180
        x11 = x[:, 10].clone() * torch.pi / 180

        bupper = torch.zeros((x.shape[0], 6), **self.torchargs)
        bupper[:, 0] = x8 +x9/2
        bupper[:, 1] = x3
        bupper[:, 2] = torch.tan(x10 - x11/2)
        bupper[:, 4] = x4
        bupper[:, 5] = torch.sqrt(2*x1)
        return bupper

    def _blower(self, x) :
        """Computes b lower vector for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: b lower vector of shape [batch_size, 6]
        """
        x1 = x[:, 0].clone()
        x6 = x[:, 5].clone()
        x7 = x[:, 6].clone()
        x8 = x[:, 7].clone()
        x9 = x[:, 8].clone()
        #in radians
        x10 = x[:, 9].clone() * torch.pi / 180
        x11 = x[:, 10].clone() * torch.pi / 180

        blower = torch.zeros((x.shape[0], 6), **self.torchargs)
        blower[:, 0] = x8 - x9/2
        blower[:, 1] = x6 
        blower[:, 2] = torch.tan(x10 + x11/2)
        blower[:, 4] = x7
        blower[:, 5] = -torch.sqrt(2*x1)
        return blower
    
    def _au(self, x) :
        """Computes a upper vector for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: a upper vector of shape [batch_size, 6]
        """
        cupper = self._cupper(x)
        bupper = self._bupper(x)
        au = torch.linalg.solve(cupper, bupper)
        return au
    
    def _al(self, x) :
        """Computes a lower vector for PARSEC parameterization.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 11]

        Returns:
            torch.Tensor: a lower vector of shape [batch_size, 6]
        """
        clower = self._clower(x)
        blower = self._blower(x)
        al = torch.linalg.solve(clower, blower)
        return al

    def ya_upper(self, xa, x) :
        """Computes y upper values for PARSEC parameterization.

        Args:
            xa (torch.Tensor): x values of shape [1, n_pts]
            x (torch.Tensor): airfoil parameters of shape [batch_size, 11]
        
        Returns:
            torch.Tensor: y upper values of shape [batch_size, n_pts]
        """
        
        au = self._au(x)
        ya_u = torch.zeros((x.shape[0], xa.shape[1]), **self.torchargs)
        for i in range(6) :
            ya_u += au[:, i].unsqueeze(-1) * xa**(i+1-1/2)
        return ya_u
    
    def ya_lower(self, xa, x) :
        """Computes y lower values for PARSEC parameterization.

        Args:
            xa (torch.Tensor): x values of shape [1, n_pts]
            x (torch.Tensor): airfoil parameters of shape [batch_size, 11]

        Returns:
            torch.Tensor: y lower values of shape [batch_size, n_pts]
        """

        al = self._al(x)
        ya_l = torch.zeros((x.shape[0], xa.shape[1]), **self.torchargs)
        for i in range(6) :
            ya_l += al[:, i].unsqueeze(-1) * xa**(i+1-1/2)
        return ya_l
    

    def _delete_old_files(self, *filenames):
        """
        Delete one or more files if they exist.
        
        Args:
            *filenames: Variable length argument list of filenames to delete.
        """

        for file in filenames:
            if os.path.exists(self.directory + "/" + file):
                try:
                    os.remove(self.directory + "/" + file)
                except Exception as e:
                    print(f"Error deleting {self.directory + '/' + file}: {e}")


    def _write_airfoil_dat(self, filename, xa_u, ya_u, xa_l, ya_l, name="custom"):
        """
        Write airfoil data to a .dat file.

        Args:
            filename (str): Name of the output .dat file.
            xa_u (torch.Tensor): x-coordinates for the upper surface.
            ya_u (torch.Tensor): y-coordinates for the upper surface.
            xa_l (torch.Tensor): x-coordinates for the lower surface.
            ya_l (torch.Tensor): y-coordinates for the lower surface.
            name (str): Name of the airfoil (used as a header).
        """
        with open(self.directory + "/" + filename, "w") as f:
            f.write(name + "\n")

            # Upper surface
            for x, y in zip(xa_u, ya_u):
                f.write(f"{x:.6f} {y:.6f}\n")

            # Lower surface
            for x, y in zip(xa_l, ya_l):
                f.write(f"{x:.6f} {y:.6f}\n")

    def _write_xfoil_input(self,
                           filename,
                           airfoil,
                           alpha,
                           Re,
                           Mach,
                           polar_file,      
                           cp_file=None     
                        ):
        """
        Write the XFOIL input file.

        Args:
            filename (str): Name of the XFOIL input file to create.
            airfoil (str): DAT file for current PARSEC geometry.
            alpha (float): Angle of attack in degrees.
            Re (float): Reynolds number.
            Mach (float): Mach number.
            polar_file (str): Path to save polar results.
            cp_file (str, optional): Path to save Cp results. If None, Cp output is not generated.
        """
        with open(self.directory + "/" + filename, "w") as f:
            # Load airfoil geometry
            f.write(f"LOAD {self.directory + '/' + airfoil}\n")
            f.write("PANE\n\n")  # Panel the geometry

            # Operating conditions
            f.write("OPER\n")
            f.write(f"VISC {Re}\n")      # Activate viscous BL if Re > 0
            f.write(f"MACH {Mach}\n")    # Compressibility
            f.write("ITER 200\n")      # Max iterations

            # Save polar results
            f.write("PACC\n")
            f.write(f"{self.directory + '/' + polar_file}\n\n")

            # Set angle of attack
            f.write(f"ALFA {alpha}\n\n")

            # Optional Cp output
            if cp_file:
                f.write(f"CPWR {cp_file}\n")

            # Stop writing polar
            #f.write("PACC\n")
            f.write("QUIT\n")

    def _run_xfoil(self, xfoil_exe, input_file):
        """Run XFOIL with the specified input file.

        Args:
            xfoil_exe (str): Path to the XFOIL executable.
            input_file (str): Path to the XFOIL input file.
        """
        res = subprocess.run(
            [xfoil_exe],
            stdin=open(self.directory + "/" + input_file),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=".",
        )
        return res

    def _read_polar(self, filename="polar.txt"):
        """Read the polar data from a file.

        Args:
            filename (str, optional): Path to the polar file. Defaults to "polar.txt".

        Returns:
            tuple: Angle of attack, lift coefficient, drag coefficient, moment coefficient.
        """
 
        try :
            data = np.loadtxt(self.directory + "/" + filename, skiprows=12)
            if data.size == 0 :
                return 1e2, 1e2, 1e2, 1e2
            alpha, cl, cd, cm = data[-7], data[-6], data[-5], data[-3]
            return alpha, cl, cd, cm
        except Exception as e:
            raise RuntimeError(f"Error reading polar file '{self.directory + '/' + filename}': {e}")

    def _compute_cost(self, i, xa_u, ya_u, xa_l, ya_l, Mach, Alpha, Re, W) :
        """Compute the cost and moment.

        Args:
            i (int): Airfoil index.
            xa_u (np.ndarray): Upper surface x-coordinates.
            ya_u (np.ndarray): Upper surface y-coordinates.
            xa_l (np.ndarray): Lower surface x-coordinates.
            ya_l (np.ndarray): Lower surface y-coordinates.
            Mach (float): Mach number.
            Alpha (float): Angle of attack.
            Re (float): Reynolds number.
            W (float): Weight for the cost function .

        Returns:
            tuple: Cost and moment coefficient.
        """
        all_files = [f"airfoil_{i}.dat", f"xfoil_input_{i}.txt", f"polar_{i}.txt", ":00.bl"]
        self._delete_old_files(*all_files)

        #write airfoil dat file
        self._write_airfoil_dat(f"airfoil_{i}.dat", xa_u.cpu().numpy(), ya_u.cpu().numpy(), xa_l.cpu().numpy(), ya_l.cpu().numpy())

        #write xfoil input file
        self._write_xfoil_input(f"xfoil_input_{i}.txt", f"airfoil_{i}.dat", Alpha, Re, Mach, f"polar_{i}.txt")

        #run xfoil
        try :
            res = self._run_xfoil("/opt/xfoil/xfoil", f"xfoil_input_{i}.txt")
            if res.returncode != 0 :
                print(f"Warning: XFOIL crashed, return code {res.returncode}. Setting cost to 1e2.")
                
                cost = 1e2
                cm = 1e2
                return torch.tensor(cost, **self.torchargs), torch.tensor(cm, **self.torchargs)
        except Exception as e:
            print(f"Error running XFOIL: {e}. Setting cost to 1e2.")
            cost = 1e2
            cm = 1e2
            return torch.tensor(cost, **self.torchargs), torch.tensor(cm, **self.torchargs)

        #read polar file
        alpha, cl, cd, cm = self._read_polar(f"polar_{i}.txt")

        time.sleep(0.01)


        #compute cost (here we want to maximize cl/cd)
        cost = - W * cl/cd
    
        return torch.tensor(cost, **self.torchargs), torch.tensor(cm, **self.torchargs)

    def __call__(self, x) :
        """Evaluate the airfoil shape for given PARSEC parameters.

        Args:
            x (torch.Tensor): Airfoil parameters (without \Delta y_te) of shape [batch_size, 10]
        
        Returns:
            tuple: Cost and moment coefficient.
        """
        

        assert torch.all(x >= self.bounds[0]) and torch.all(x <= self.bounds[1]) and torch.all(torch.isfinite(x)), f"All input points must be in the bounds and finite."
        x = x.to(**self.torchargs)

        #add \Delta y_te = 0 to x
        x_full = torch.zeros((x.shape[0], 11), **self.torchargs)
        x_full[:, :8] = x[:, :8].clone()
        x_full[:, 8] = 0.0
        x_full[:, 9:] = x[:, 8:].clone()

        #compute upper and lower surfaces coordinates
        end_te = 0.6
        end_le = 0.4
        xa_TE_upper = np.linspace(1, end_te, 200, endpoint=False)
        xa_LE_upper = np.linspace(end_le,0, 200, endpoint=False)
        xa_center_upper = np.linspace(end_te,end_le, 100, endpoint=False)
        xa_upper = torch.tensor(np.concatenate([xa_TE_upper, xa_center_upper, xa_LE_upper]), **self.torchargs).unsqueeze(0)

        xa_TE_lower = np.linspace(end_te,1, 200, endpoint=True)
        xa_LE_lower = np.linspace(0, end_le, 200, endpoint=False)
        xa_center_lower = np.linspace(end_le,end_te, 100, endpoint=False)
        xa_lower = torch.tensor(np.concatenate([xa_LE_lower, xa_center_lower, xa_TE_lower]), **self.torchargs).unsqueeze(0)
        ya_u = self.ya_upper(xa_upper, x_full)
        ya_l = self.ya_lower(xa_lower, x_full)
   
        cost = torch.zeros(x.shape[0], **self.torchargs)
        constraint = torch.zeros(x.shape[0], **self.torchargs)
        for i in range(x.shape[0]) :
            cost[i], constraint[i] = self._compute_cost(0, xa_upper[0, :], ya_u[i, :], xa_lower[0,:], ya_l[i, :], self.MACH[0], self.ALPHA[0], self.RE[0], self.W[0])
        if self.fidelity == 0 :
            return cost, constraint
        else :
            for i in range(x.shape[0]) :
                cost_sum = 0
                for j in range(1, len(self.MACH)):
                    cost_ij, _ = self._compute_cost(j, xa_upper[0, :], ya_u[i, :], xa_lower[0,:], ya_l[i, :], self.MACH[j], self.ALPHA[j], self.RE[j], self.W[j])
                    cost_sum += cost_ij.clone()
                cost[i] = cost_sum + cost[i]
            return cost, constraint

