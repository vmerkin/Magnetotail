from numpy import linspace,sqrt,exp,tanh,where,cosh,pi,log,array,isscalar
from matplotlib.pyplot import plot,show
import sys


# Everything is normalized
# B/B0; x/L; z/L; Psi/(B0*L)
# flux tube volume V normalized to L/B0

class equilibrium():
    F    = lambda self,x: 1.+self.a/cosh(self.e2*x)**2
    beta = lambda self,x: exp(self.e1*self.g(x))
    g    = lambda self,x: x+self.a/self.e2*(1.+tanh(self.e2*(x-self.x0)))
    gp   = lambda self,x: self.F(x-self.x0) # dg/dx
    gpp = lambda self,x: -2.*self.a*self.e2*tanh(self.e2*(x-self.x0))/cosh(self.e2*(x-self.x0))**2
    Bx   = lambda self,x,z: -1./self.beta(x)*tanh(z/self.beta(x))
    Bz   = lambda self,x,z: self.e1*self.gp(x)*(1.-z/self.beta(x)*tanh(z/self.beta(x)))
    Psi  = lambda self,x,z: log(self.beta(x)*cosh(z/self.beta(x)))

    def __init__(self,a,e1,e2,x0,gamma=5./3.):
        self.a  = a
        self.e1 = e1
        self.e2 = e2
        self.gamma = gamma
        self.x0 = x0

    def y(self,x,x1):
        # We avoid the singularity at x=x1 (u=0)
        # and thus express y through x directly
        #        return sqrt(exp(self.u)-1.)

        # The function u (SS10) is simply 2*log(beta(x1)/beta(x)),
        # where x1 is the point where we investigate stability.
        # Funcition y below is sqrt( e^u-1) however, we express y
        # directly via g(x) to avoid the singularity at 0, which would
        # arise if we expressed y via u.

        #y is only defined for x<=x1. Cut the input x off at x1:
        x = array(x)
        ind = (x<=x1)
        if ind.any():
#            return [x[ind],sqrt( exp(2*self.e1*(self.g(x1)-self.g(x[ind])))-1. )]
            return [x[ind],sqrt( (self.beta(x1)/self.beta(x[ind]))**2-1. )]
        else:
            sys.exit('All points where y is requested are tailward of x1.')

    def V(self,x1):
        # flux tube volume (note, normalized to L/B0)
        # x1 -- point where we assess stability
        f = lambda x: 1/self.gp(x)
        I = self.I(x1,f)
        return  I[0],pi/self.e1*self.beta(x1)*I[1]


    def Cd(self,x1):
        # flux tube volume (note, normalized to L/B0)
        # x1 -- point where we assess stability
        f = lambda x: 1/self.gp(x)
        I = self.I(x1,f)
        return  I[0],self.gp(x1)*I[1]

    def Bi(self,x1):
        # The Bici integral
        f = lambda x: self.gpp(x)/self.gp(x)**3
        I = self.I(x1,f)
        return I[0],-self.gp(x1)/self.e1/self.Cd(x1)[1]*I[1]

    def Q(self,x1):
        f1 = lambda x: self.gpp(x)/self.gp(x)**3
        f2 = lambda x: 1/self.gp(x)

        xx1,I1 = self.I(x1,f1)
        xx2,I2 = self.I(x1,f2)

        if not all(xx1==xx2): 
            sys.exit('Error in equilibrium Q function: coordinate arrays returned from two I integrals are not the same.')

        # Note, the y coordiante coming out of here should be
        # identical to the one used inside the I integral, i.e., it
        # should be a uniform linspace array.  We could, in pricinpe,
        # pass the y array from the I integral as an output, but for
        # backward compatibiility, I did not want to change the
        # function calling sequence. For this reason, I'm just
        # calculating the y array here.
        xy,y = self.y(xx1,x1)

        if not all(xx1==xy):
            sys.exit('Error in equilibrium Q function: coordinate arrays returned from the I integral and the y function are not the same.')

        F1 = -1./self.e1*I1/I2
        F2 = self.beta(x1)/self.beta(xx1)/self.gp(xx1)/y/(y**2+1)/I2
        return(xx1,1+F1+F2)

    def I(self,x1,f,N=1.e6,tol=1.e-10,small=1.e-8):
        import scipy.integrate as integrate
        from scipy.interpolate import interp1d

        # 1./pi*int_0^\inf {f(x)*du/sqrt(e^u-1)}

        # Here we avoid the singularity at the Bz peak by tranforming
        # to a new variable y = sqrt(exp(phi)-1)

        # in the previous version, I used to define ymax as y where
        # 1(y**2+1) <= tol which corresponds to (for large y):
        # ymax = sqrt(1/tol)   
        # this would require tol=1.e-8

        # this time I decided to derive ymax by assuming that the
        # integral has converged to within the specified
        # tolerance. The integral is on the order of 1 always , so we
        # require dy/ymax**2 = tol*1., i.e., 1/(N*ymax) = tol.  This,
        # however, gives a much less restriction on ymax (it is much
        # smaller), so I set tol=1.e-10 which makes it equivalent to
        # the previous formulation.
        ymax = 1/N/tol
        xmax = self.__find_xmax(x1,ymax)
        x = linspace(xmax,x1,N)
        yatx = self.y(x,x1)[1]
        fyx = interp1d(yatx,x)

        # presumably because of roundoff error, defining y without
        # -small below results in y outside of the interpolation
        # domain for fyx below.
        y = linspace(0,ymax-small,N)
        xaty = fyx(y)
        return(xaty,integrate.cumtrapz(2./pi*f(xaty)/(y**2+1),y,initial=0))

    def calc1(self,Lx,N=1.e6,tol=1.e-8):
        self.N  = N

        # Here we avoid the singularity at the Bz peak by tranforming to a new variable y = sqrt(exp(phi)-1)
        self.x = linspace(0,Lx,N)
        x = self.x
        #    y = sqrt(exp(2*e1*(x+a/e2*tanh(e2*x)))-1)
        y = self.yf(x)
        ymaxind = where(1./(y**2+1)>=tol)
        fyx = interp1d(y[ymaxind],x[ymaxind])
        yint = linspace(0,y[ymaxind][-1],N)
        self.xint = fyx(yint)
        xint = self.xint
        #    bf = (1.+a)/(1.+a/cosh(e2*xint)**2)
        bicif = 2./pi*self.bf1(xint)/(yint**2+1)
        self.bici = integrate.cumtrapz(bicif,yint,initial=0)
        return(self.bici)

    def __find_xmax(self,xstart,ymax,Dx=100.):
        import scipy.optimize

        # VERY IMPORTANT!!!!!!  I ran into this nasty bug. xstart is
        # intended to be float.  if a numpy array is passed by
        # accident, even if xstart.size=1 (e.g., array([1.])), the
        # stuff below will modify the array that is being passed on
        # exit!!!!!!!
        # therefore, check fo the type of xstart explicitely:
        if not isscalar(xstart): sys.exit('xstart type is not float. Exiting...')
        xguess = xstart
        while self.y(xguess-Dx,xstart)[1]<=ymax:
            xguess-=Dx
            continue
        return scipy.optimize.brentq(lambda x: self.y(x,xstart)[1]-ymax,xguess-Dx,xguess)
