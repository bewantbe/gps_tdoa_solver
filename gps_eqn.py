# Solve gps equation

import numpy as np
_a = lambda x: np.array(x)
_ia = lambda x: np.array(x, dtype=int)

# Use JAX for derivatives
import jax                # JAX >= 0.4.26
import jax.numpy as jnp
from jax import grad
from jax import jacfwd, jacrev

# use xnp to control globol numerical lib
xnp = np

_ja = lambda x: xnp.array(x)

import logging
logging.getLogger("jax").setLevel(logging.ERROR)
# old
# Ref. https://github.com/google/jax/issues/10070
#from absl import logging
#logging.set_verbosity(logging.ERROR)

# set JAX to use CPU only
jax.config.update('jax_default_device', jax.devices('cpu')[0])

logging.getLogger("jax").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
plt.ion()

# parameters (known): m = number of satellites, p = positions, t = time
# p1, p2, ..., pm, ct1, ct2, ..., ctm

# to solve
# q = [rx, ry, rz, ct0]

def gen_sound_data_5p():
    """Generate example microphone configuration with sound source position."""
    # sound source position (unit: m)
    r_true = np.array([0.1, 0.06, -0.2])
    # temperature (unit: degree Celsius)
    temp = 20
    # speed of sound (unit: m/s)
    # Ref. [Air - Speed of Sound vs. Temperature]
    #      (https://www.engineeringtoolbox.com/air-speed-sound-d_603.html)
    # Ref. https://sengpielaudio.com/calculator-airpressure.htm
    c = 20.05 * (273.16 + temp) ** 0.5
    #print(f'Sound speed {c:.2f} m/s, at temperature {temp} C.')

    m = 5            # number of microphones

    p = np.zeros((m, 3))     # microphone positions           (unit: m)
    ct = np.zeros(m)         # relative time delay of arrival (unit: s)

    # exact positions and time delays
    p[0] = _a([0.0, 0.0, -0.05])
    ct[0] = np.linalg.norm(r_true - p[0])
    p[1] = _a([+0.25/2, +0.18/2, 0])
    ct[1] = np.linalg.norm(r_true - p[1]) - ct[0]
    p[2] = _a([+0.25/2, -0.18/2, 0])
    ct[2] = np.linalg.norm(r_true - p[2]) - ct[0]
    p[3] = _a([-0.25/2, +0.18/2, 0])
    ct[3] = np.linalg.norm(r_true - p[3]) - ct[0]
    p[4] = _a([-0.25/2, -0.18/2, 0])
    ct[4] = np.linalg.norm(r_true - p[4]) - ct[0]

    ct0 = ct[0]
    ct[0] = ct[0] - ct0

    return p, ct, r_true, ct0

def add_noise_sound_data(p, ct, err_pos, err_ct):
    # add measurement noise
    p_measure  = p.copy()
    ct_measure = ct.copy()
    for j in range(len(ct)):
        p_measure[j]  = p[j]  + err_pos * np.random.randn(3)
        ct_measure[j] = ct[j] + err_ct  * np.random.randn()

    return p_measure, ct_measure

def gen_gps_data_d4():
    # receiver position (unit: m)
    r_true = np.array([0.0, 0.0, 6371000])
    c = 299792458.0

    p = np.zeros((4, 3))     # satellite positions            (unit: m)
    ct = np.zeros(4)         # relative time delay of arrival (unit: s)

    R = 20180e3 + 6370e3
    randu = lambda : 2 * np.random.rand() - 1

    # exact positions and time delays
    p[0] = _a([randu() * R, randu() * R, R])
    p[1] = _a([randu() * R, randu() * R, R])
    p[2] = _a([randu() * R, randu() * R, R])
    p[3] = _a([randu() * R, randu() * R, R])
    p = p / np.linalg.norm(p, axis=1)[:, np.newaxis] * R

    ct[0] = np.linalg.norm(r_true - p[0])
    ct[1] = np.linalg.norm(r_true - p[1]) - ct[0]
    ct[2] = np.linalg.norm(r_true - p[2]) - ct[0]
    ct[3] = np.linalg.norm(r_true - p[3]) - ct[0]

    ct0 = ct[0]
    ct[0] = ct[0] - ct0

    # noise level
    err_pos = 1.0
    err_ct = c * 1e-9

    # add measurement noise
    for j in range(4):
        p[j]  = p[j]  + err_pos * np.random.randn(3)
        ct[j] = ct[j] + err_ct  * np.random.randn()

    return p, ct, r_true, ct0

def gen_sound_data_2p():
    r_true = np.array([0.1, 1.0, 0.0])

    m = 2
    p = np.zeros((m, 3))     # microphone positions           (unit: m)
    ct = np.zeros(m)         # relative time delay of arrival (unit: s)

    # exact positions and time delays
    p[0] = _a([+0.02, 0.0, 0.0])
    p[1] = _a([-0.02, 0.0, 0.0])

    ct[0] = np.linalg.norm(r_true - p[0])
    ct[1] = np.linalg.norm(r_true - p[1]) - ct[0]
    ct0 = ct[0]
    ct[0] = ct[0] - ct0

    return p, ct, r_true, ct0

def GPS_F(p, ct, r, ct0):
    m = len(ct)
    v = xnp.array([
        xnp.linalg.norm(r - p[j]) ** 2 - (ct0 + ct[j]) ** 2
        for j in range(m)
    ])
    return v

def GPS_dF_dr(p, ct, r, ct0):
    m = len(ct)
    dF = xnp.array([
        2 * (r - p[j])
        for j in range(m)
    ])
    return dF

def GPS_dF_dt(p, ct, r, ct0):
    m = len(ct)
    dF = xnp.array([
        -2 * (ct0 + ct[j])
        for j in range(m)
    ])
    return dF

def GPS_dF_dq(p, ct, r, ct0):
    J = xnp.hstack([GPS_dF_dr(p, ct, r, ct0),
                    GPS_dF_dt(p, ct, r, ct0)[:, xnp.newaxis]])
    return J

def NewtonIter(F, dF_dx, x_init, max_iter=20, tol=1e-7):
    """Newton's method for solving F(x) = 0 in least square sense."""
    x = x_init
    #print('x0', x)
    for i in range(max_iter):
        J = dF_dx(x)
        F_val = F(x)
        delta = xnp.linalg.lstsq(J, -F_val, rcond=None)[0].flatten()
        x = x + delta
        #print('x', x)
        if xnp.linalg.norm(delta) < tol:
            break
    #print('n_iter =', i)

    return x

def NewtonIterGPS(p, ct, r_n, ct0_n):
    #q0 = xnp.hstack([r_n, ct0_n])
    q0 = np.hstack([r_n, ct0_n])
    F  = lambda q: GPS_F(p, ct, q[:3], q[3])
    dF = lambda q: GPS_dF_dq(p, ct, q[:3], q[3])
    q = NewtonIter(F, dF, q0)
    return q[:3], q[3]

def NewtonIterGPSConstraint(p, ct, r_n, ct0_n, p_c, n_c):
    """ allow constraint(s) in the form (r - p_c) * n_c = 0 """
    p_c_2d = xnp.atleast_2d(p_c)
    n_c_2d = xnp.atleast_2d(n_c)
    num_constraint = len(n_c_2d)
    F  = lambda q: xnp.hstack([
            GPS_F(p, ct, q[:3], q[3]),
            [xnp.dot(q[:3] - p_c_2d[i], n_c_2d[i])
                for i in range(num_constraint)]
        ])
    n_c_2d_e = xnp.hstack([n_c_2d, xnp.zeros((n_c_2d.shape[0], 1))])
    dF = lambda q: xnp.vstack([
            GPS_dF_dq(p, ct, q[:3], q[3]),
            n_c_2d_e
        ])
    q0 = np.hstack([r_n, ct0_n])
    q = NewtonIter(F, dF, q0)
    return q[:3], q[3]

def DirectGPSSolver(p, ct):
    """
    Ref. An Algebraic Solution of the GPS Equations
    """
    m = len(ct)
    A = xnp.hstack([p, ct[:, np.newaxis]])
    i0 = xnp.ones(m)
    r_vec = _ja([
        (xnp.dot(p[j], p[j]) - ct[j]**2) / 2
        for j in range(m)
    ])
    if 0:
        B = xnp.dot(xnp.linalg.inv(xnp.dot(A.T, A)), A.T)
        u_vec = B @ i0
        v_vec = B @ r_vec
    else:
        # hope for better numerical stability
        # But still, for rank-deficient A, the solution is usuall incorrect
        u_vec = xnp.linalg.lstsq(A, i0, rcond=None)[0].flatten()
        v_vec = xnp.linalg.lstsq(A, r_vec, rcond=None)[0].flatten()
    E = xnp.dot(u_vec[:3], u_vec[:3]) - u_vec[3] * u_vec[3]
    F = xnp.dot(u_vec[:3], v_vec[:3]) - u_vec[3] * v_vec[3] - 1
    G = xnp.dot(v_vec[:3], v_vec[:3]) - v_vec[3] * v_vec[3]
    #lambdas = xnp.roots(_ja([E, 2*F, G]), strip_zeros = False)
    lambdas = xnp.roots(_ja([E, 2*F, G]))
    y_all = u_vec[:, xnp.newaxis] * lambdas[xnp.newaxis, :] + v_vec[:, xnp.newaxis]

    # pick the real solution
    y1 = xnp.real(y_all[:, 0])
    y2 = xnp.real(y_all[:, 1])
    l1 = xnp.linalg.norm(GPS_F(p, ct, y1[:3], y1[3]))
    l2 = xnp.linalg.norm(GPS_F(p, ct, y2[:3], y2[3]))
    if l1 < l2:
        y = y1
    else:
        y = y2

    return y[:3], y[3]

def GetK(p, ct, r, ct0):
    m = len(ct)
    A = -2 * xnp.hstack([r[np.newaxis, :] - p, ct0 + ct[:, xnp.newaxis]])
    dF_dp = xnp.vstack([
        xnp.hstack([xnp.zeros((1, 4*j)), A[j:j+1, :], xnp.zeros((1, 4*(m-j-1)))])
        for j in range(m)
    ])
    dF_dq = GPS_dF_dq(p, ct, r, ct0)
    K = - xnp.linalg.lstsq(dF_dq, dF_dp, rcond=None)[0]
    return K

def GetK_Constraint(p, ct, r, ct0, n_c):
    m = len(ct)
    n_c_2d = xnp.atleast_2d(n_c)
    n_c_2d_e = xnp.hstack([n_c_2d, xnp.zeros((n_c_2d.shape[0], 1))])
    A = -2 * xnp.hstack([r[np.newaxis, :] - p, ct0 + ct[:, xnp.newaxis]])
    dF_dp = xnp.vstack([
        xnp.hstack([xnp.zeros((1, 4*j)), A[j:j+1, :], xnp.zeros((1, 4*(m-j-1)))])
        for j in range(m)
    ])
    dF_dp = xnp.vstack([dF_dp, xnp.zeros((len(n_c_2d), 4*m))])
    dF_dq = GPS_dF_dq(p, ct, r, ct0)
    dF_dq = xnp.vstack([dF_dq, n_c_2d_e])
    K = - xnp.linalg.lstsq(dF_dq, dF_dp, rcond=None)[0]
    return K

def GetErrorEclipsed(p, ct, r, ct0, err_pos, err_ct, n_c = None):
    m = len(ct)
    if n_c is None:
        K = GetK(p, ct, r, ct0)
    else:
        K = GetK_Constraint(p, ct, r, ct0, n_c)
    # error matrix of p and ct
    Lambda = xnp.diag(
        xnp.tile(
            xnp.hstack([err_pos**2 * xnp.ones(3),
                        err_ct**2 * xnp.ones(1)]),
        m)
    )
    Omega = xnp.dot(K, xnp.dot(Lambda, K.T))
    return Omega

def SolveWithErrorInfo(p, ct, err_pos, err_ct, p_c, n_c):
    r_init, ct0_init = DirectGPSSolver(p, ct)
    r_init = xnp.hstack([r_init[:2], p_c[2]])
    r_n, ct0_n = NewtonIterGPSConstraint(p, ct, r_init, ct0_init, p_c, n_c)
    # Error
    Omega = GetErrorEclipsed(p, ct, r_n, ct0_n, err_pos, err_ct, n_c)
    #gdop = np.sqrt(np.trace(Omega[:3, :3]))
    hdop = np.sqrt(np.trace(Omega[:2, :2]))
    cov_xy = Omega[:2, :2]
    #print('h0 =', r_init[2])
    #print('hn =', r_n[2])
    return r_n[:2], hdop, cov_xy

def verify_dF(p, ct, r_true, ct0):
    # For verification of derivatives

    # Get jacobian of GPS_F with respect to r
    dF_dr_auto = jacrev(GPS_F, argnums=2)
    dF_dt_auto = jacrev(GPS_F, argnums=3)

    print(dF_dr_auto(p, ct, r_true, ct0))
    print(dF_dt_auto(p, ct, r_true, ct0))

    print(GPS_dF_dr(p, ct, r_true, ct0))
    print(GPS_dF_dt(p, ct, r_true, ct0))

def verify_K(p, ct, r_true, ct0):
    # Verify dF / dp and dF / dct
    dF_dp_auto = jacrev(GPS_F, argnums=0)
    print(dF_dp_auto(p, ct, r_true, ct0))
    dF_dct_auto = jacrev(GPS_F, argnums=1)
    print(dF_dct_auto(p, ct, r_true, ct0))
    j = 1
    print(-2*(r_true-p[j]), -2*(ct[j]+ct0))

def SolverStat(p_orig, ct_orig, r_true, ct0, err_pos, err_ct, p_c, n_c):
    if n_c is None:
        solver='Newton'
    else:
        solver='NewtonConstraint'
    n_trial = 100
    arr_pos = np.zeros((n_trial,3))
    for i in range(n_trial):
        p, ct = add_noise_sound_data(p_orig, ct_orig, err_pos, err_ct)
        if solver == 'Newton':
            r_n, ct0_n = NewtonIterGPS(p, ct, r_true, ct0)
        elif solver == 'NewtonConstraint':
            r_n, ct0_n = NewtonIterGPSConstraint(p, ct, r_true, ct0, p_c, n_c)
        arr_pos[i] = r_n
    
    s_cov = np.cov(arr_pos.T)
    #print('s_cov\n', s_cov)
    gdop = np.sqrt(np.trace(s_cov))
    #print('GDOP stat:', gdop)
    # plot arr_pos in 3D with matplotlib
    plt.figure(10)
    plt.cla()
    ax = plt.axes(projection='3d')
    ax.scatter3D(p_orig[:,0], p_orig[:,1], p_orig[:,2], c='r')
    ax.scatter3D(arr_pos[:,0], arr_pos[:,1], arr_pos[:,2], c='b')
    return ax, s_cov, gdop

def DrawCovEclipse(ax, r, ct0, Omega):
    # Covariance matrix
    #cov = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    cov = Omega[:3, :3]
    #print('cov\n', cov)

    # Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)

    # Create a grid of points
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones_like(theta), np.cos(phi))

    # O = P V * V P^T
    # X(at 1) -> Z(at O)
    #    Z = P V X
    #   E(X X^T) = I
    #   E(Z Z^T) = P V X X^T V P^T = O

    # Transform the points to the ellipse
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j], y[i,j], z[i,j] = np.dot(eigvecs,
                                            [x[i,j], y[i,j], z[i,j]] * np.sqrt(eigvals)) \
                                     + r

    # Create a 3D plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.2)
    #plt.show()

def DrawCovEclipse2D(ax, r_2d, cov_xy):
    eigvals, eigvecs = np.linalg.eig(cov_xy)

    theta = np.linspace(0, 2*np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)

    # Transform the points to the ellipse
    for i in range(len(x)):
        x[i], y[i] = np.dot(eigvecs, [x[i], y[i]] * np.sqrt(eigvals)) \
                    + r_2d

    plt.plot(x, y)

def TestGPSlike():
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    p, ct, r_true, ct0 = gen_gps_data_d4()
    print('r_true', r_true)

    # solve
    r_n, ct0_n = NewtonIterGPS(p, ct, r_true, ct0)
    print('Newton solver:')
    print('r_n', r_n, '  diff =', r_n - r_true)
    print(f'ct0_n {ct0_n:.2f}  diff = {ct0_n - ct0:.2f}')

    r_direct, ct0_direct = DirectGPSSolver(p, ct)
    print('Direct solver:')
    print('r_direct', r_direct)
    print(f'ct0_direct {ct0_direct:0.2f}')

    Omega = GetErrorEclipsed(p, ct, r_true, ct0, 1.0, 1.0e-9 * 3e8)
    print(f'GDOP = {np.sqrt(np.trace(Omega[:3, :3])):.2f}')

    np.set_printoptions(formatter=None)

def TestSoundSource():
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    p_orig, ct_orig, r_true, ct0 = gen_sound_data_5p()
    # noise level
    err_pos = 0.2e-3                 # default 0.2mm
    sample_rate = 250e3              # default 250kHz
    err_ct = 340 * 2.0e-0 / sample_rate   # default 2 samples
    p, ct = add_noise_sound_data(p_orig, ct_orig, err_pos, err_ct)
    #print(p)
    #print(ct)
    print('r_true', r_true)
    print(f'ct0 {ct0:0.4f}')

    #print('GPS_F =', GPS_F(p, ct, r_true, ct0))

    if 0:
        verify_dF(p, ct, r_true, ct0)

    r_n, ct0_n = NewtonIterGPS(p, ct, r_true, ct0)
    print('Newton solver:')
    print('r_n', r_n)
    print(f'ct0_n {ct0_n:0.4f}')

    print('Direct solver:')
    r_direct, ct0_direct = DirectGPSSolver(p, ct)
    print('r_direct', r_direct)
    print(f'ct0_direct {ct0_direct:0.4f}')

    print('Newton solver with constraint:')
    p_c = _a([0, 0, r_true[2]])
    coef_constraint = 0.2  # 0.05 mild constraint, 0.2 smooth strong constraint
    n_c = _a([0, 0, 1]) * coef_constraint
    r_nc, ct0_nc = NewtonIterGPSConstraint(p, ct, r_true, ct0, p_c, n_c)
    print('r_nc', r_nc, '  diff =', r_nc - r_true)
    print(f'ct0_nc {ct0_nc:0.4f}')

    if 0:
        verify_K(p, ct, r_true, ct0)

    if 1:
        ax, s_cov, gdop = SolverStat(p_orig, ct_orig, r_true, ct0, err_pos, err_ct, p_c, n_c)
        print(f'GDOP stat: {gdop:.4f}')

    if 1:
        Omega = GetErrorEclipsed(p_orig, ct_orig, r_true, ct0, err_pos, err_ct, n_c)
        print(f'GDOP = {np.sqrt(np.trace(Omega[:3, :3])):.4f}')
        ax.axis('equal')
        DrawCovEclipse(ax, r_true, ct0, Omega)

    if 1:
        r_2d, hdop, cov_xy = SolveWithErrorInfo(p, ct, err_pos, err_ct, p_c, n_c)
        print('r_2d', r_2d, '  diff =', r_2d - r_true[:2])
        print(f'hdop {hdop:.4f}')
        #print('cov_xy\n', cov_xy)
        
        # Create a 2D plot
        plt.figure(20)
        plt.cla()
        plt.axis('equal')
        plt.plot(p[:,0], p[:,1], 'ro')
        plt.plot(r_true[0], r_true[1], 'go')
        DrawCovEclipse2D(None, r_2d, cov_xy)
        #plt.show()

    np.set_printoptions(formatter=None)

    globals().update(locals())  # easier for debug

def TestSoundSource2D():
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # gen_sound_data_2p
    p_orig, ct_orig, r_true, ct0 = gen_sound_data_2p()
    print('r_true', r_true)
    err_pos = 1e-3
    err_ct = 340 * 1.0 / 48e3
    #p, ct = add_noise_sound_data(p_orig, ct_orig, 0.2e-3, 340 * 2.0e-0 / 48e3)
    p, ct = add_noise_sound_data(p_orig, ct_orig, err_pos, err_ct)
    # constraint on line passing (-1,1,0) to (1,1,0)
    p_c = _a([[0, 1, 0], [0, 1, 0]])
    n_c = _a([[0, 1, 0], [0, 0, 1]]) * 1e-3
    # solve position
    r_n, ct0_n = NewtonIterGPSConstraint(p, ct, r_true, ct0, p_c, n_c)
    print('r_n', r_n)

    # plot
    ax, s_cov, gdop = SolverStat(p_orig, ct_orig, r_true, ct0, err_pos, err_ct, p_c, n_c)
    ax.axis('equal')
    print(f'GDOP stat: {gdop:.4f}')

    Omega = GetErrorEclipsed(p_orig, ct_orig, r_true, ct0, err_pos, err_ct, n_c)
    print(f'GDOP = {np.sqrt(np.trace(Omega[:3, :3])):.4f}')
    DrawCovEclipse(ax, r_true, ct0, Omega)

    np.set_printoptions(formatter=None)

# for time delay estimation
def gcc_phat(x1, x2, window = None):
    # the PHAse Transform-weighted Generalized Cross-Correlation (GCC-PHAT) algorithm
    # for signals x1 and x2
    # x1(t) = s(t) + n1(t)          # as reference
    # x2(t) = s(t - d) + n2(t)
    # return the time delay estimation `d` in unit of sample
    # Ref. https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    # Ref. https://ieeexplore.ieee.org/document/5670137
    #      Analysis of the GCC-PHAT technique for multiple sources
    # Ref. https://www.rd.ntt/cs/team_project/icl/signal/iwaenc03/cdrom/data/0013.pdf
    #      CONSIDERING THE SECOND PEAK IN THE GCC FUNCTION FOR MULTI-SOURCE
    #        TDOA ESTIMATION WITH A MICROPHONE ARRAY
    # Ref. https://speechprocessingbook.aalto.fi/Enhancement/tdoa.html
    # 11.8.3. Time-Delay of Arrival (TDoA) and Direction of Arrival (DoA) Estimation
    # Ref. https://www.mathworks.com/help/phased/ref/gccdoaandtoa.html
    #      Matlab - GCC DOA and TOA
    
    if len(x1.shape) >= 2:
        x1 = x1.flatten()
    if len(x2.shape) >= 2:
        x2 = x2.flatten()
    n = len(x1)
    if window is None:
        window = np.ones(n)
    xf1 = np.fft.fft(x1 * window)
    xf2 = np.fft.fft(x2 * window)
    cxf1 = np.conj(xf1)

    weight = 1                         # correlation, most stable
    #weight = 1 / np.abs(cxf1 * xf2)   # phase correlation weighting
    #ff = np.arange(n) / n * 48000
    #weight = ((np.abs(ff - 4800) < 50) | (np.abs(ff - (48000-4800)) < 50)) + 0.1

    # Ref. Time delay estimation by generalized cross correlation methods
    #      https://ieeexplore.ieee.org/document/1164314
    cross_spectrum = cxf1 * xf2 * weight
    #crosscorrelation = np.fft.ifft(crossspectrum / np.abs(crossspectrum))
    cross_correlation = np.real(np.fft.ifft(cross_spectrum))
    i_max, val_max = find_peak_interp2(cross_correlation)
    return cross_correlation, i_max, val_max

def find_peak_interp2(x):
    # Find the peak in data points in x, interpolate it with quadratic function
    # and return the interpolated peak position.
    # Will auto wrap around the boundary.
    n = len(x)
    i_max = np.argmax(x)
    # The peak around peak should look like quadratic curve
    # a*x^2 + b*x + c = y
    # a - b + c = x1
    #         c = x2
    # a + b + c = x3
    x1 = x[(i_max - 1) % n]
    x2 = x[i_max]
    x3 = x[(i_max + 1) % n]
    c = x2
    a = (x3+x1)/2 - x2
    b = (x3-x1)/2
    if a > 0:
        return None, None
    if a == 0:
        return i_max, x2
    pos = - b / (2 * a)
    val = (a * pos + b) * pos + c
    i_interp = i_max + pos
    if i_interp > n // 2:
        i_interp = i_interp - n
    return i_interp, val

def test_find_peak_interp2():
    a = -0.12
    b = 0.5
    c = 0.3
    fn = lambda x: a * x ** 2 + b * x + c
    xx = np.arange(10)
    x_peak = - b / (2*a)
    v_peak = fn(x_peak)
    #print(x_peak)
    imax, val = find_peak_interp2(fn(xx))
    assert np.abs(imax - x_peak) < 1e-14
    assert np.abs(val - v_peak) < 1e-14

    a = -0.12
    b = 1.5
    c = 0.3
    fn = lambda x: a * x ** 2 + b * x + c
    xx = np.arange(10)
    x_peak = - b / (2*a)
    v_peak = fn(x_peak)
    x_peak = x_peak - 10
    #print('ans')
    #print(x_peak)
    #print(v_peak)
    imax, val = find_peak_interp2(fn(xx))
    #print('interp')
    #print(imax)
    #print(val)
    assert np.abs(imax - x_peak) < 1e-14
    assert np.abs(val - v_peak) < 1e-14

def generate_delayed_signal(n_measure, f0, d_shift):
    # generate signal
    n = int(n_measure / 2)
    amp_noise = 1.0       # phase jitter (simulated narrow-band)
    phase_inc = (np.ones(n) + amp_noise * np.random.randn(n)) * f0
    s1 = np.sin(2 * np.pi * np.cumsum(phase_inc))

    # generate noise
    n1 = 1.5 * np.random.randn(n_measure)
    n2 = 1.5 * np.random.randn(n_measure)

    # shift
    x1 = np.hstack([s1, np.zeros(len(n1) - len(s1))]) + n1
    x2 = np.hstack([np.zeros(d_shift),
                    s1,
                    np.zeros(len(n1) - len(s1) - d_shift)]) \
         + n2

    return x1, x2

def generate_delayed_signal_phy(d_shift):
    fs = 48000            # sampling frequency
    f0 = 0.1*fs           # signal central frequency
    n = int(0.10*fs)      # signal length
    tt = np.arange(n) / fs
    x1, x2 = generate_delayed_signal(n, f0/fs, d_shift)
    return tt, x1, x2

def test_GCC_PHAT_one(d_shift = 10, b_show = True):
    tt, x1, x2 = generate_delayed_signal_phy(d_shift)
    gcov, i_max, _ = gcc_phat(x1, x2)

    if not b_show:
        return i_max

    print(f'delay estimation (unit: sample): {i_max:.2f},   diff = {i_max - d_shift:.2f}')

    plt.figure(50)
    plt.cla()
    plt.plot(tt, x1, tt, x2)

    plt.figure(52)
    plt.cla()
    plt.plot(gcov)

    plt.figure(53)
    plt.cla()
    rg = np.arange(-20, 20)
    plt.plot(rg, np.hstack([gcov[rg[0]:], gcov[0:rg[-1]+1]]))

    plt.show()

def test_GCC_PHAT_batch():
    n_trial = 1000
    i_max_s = np.zeros(n_trial)
    d_shift = 10
    for i in range(n_trial):
        i_max_s[i] = test_GCC_PHAT_one(d_shift, False) - d_shift

    #i_max_s = _a(list(filter(lambda x: np.abs(x)<2, i_max_s)))

    print(f'mean = {np.mean(i_max_s):.3f}, std = {np.std(i_max_s):.3f}')

    # perf of weightings
    # n_trial = 1000
    # weight   std        std(filtered)
    # const. : 0.23       0.22
    # phase  : 397.87     0.31
    # band   : 125.09     0.22

    plt.figure(30)
    plt.cla()
    plt.hist(i_max_s, bins=20)

def DOA_2MIC(x_au, sr):
    n_len  = x_au.shape[0]
    n_ch   = x_au.shape[1]
    sz_wnd = 1024
    sz_hop = 512   # 50% overlap
    # loop over each window
    for j in range(int((n_len - sz_wnd) / sz_hop) + 1):
        x = x_au[j*sz_hop: j*sz_hop + sz_wnd, :]
        x1 = x[:, 0]
        x2 = x[:, 1]
        gcov, i_peak, val_peak = gcc_phat(x1, x2)
        t = j*sz_hop/sr
        rms_db = 20*np.log10(np.sqrt(np.mean(x1**2)))
        print(f'segmentation:{j:4} = {t:.3f}s, rms {rms_db:.2f}dB, peak at {i_peak: .2f}, val = {val_peak:.2f}')

def test_DOA_simu():
    tt, x1, x2 = generate_delayed_signal_phy(10)
    x_au = np.vstack([x1, x2]).T   # (len, channel)
    DOA_2MIC(x_au, 48000)

def test_DOA_wav():
    from scipy.io import wavfile
    wav_path = 'tictic.wav'
    sr, x_au = wavfile.read(wav_path)
    DOA_2MIC(x_au / 32768, sr)

def test_AEC():
    # Ref. https://pypi.org/project/speexdsp/
    # pip install speexdsp

    from speexdsp import EchoCanceller
    from scipy.io import wavfile
    wav_path = 'tictic.wav'
    sr, x_au = wavfile.read(wav_path)

    frame_size = 256
    echo_canceller = EchoCanceller.create(frame_size, 2048, sr)

    out_data
    in_data = echo_canceller.process(in_data, out_data)

    

    # Method 1
    # fitting room impulse response
    # Method 2
    # https://www.bvmengineering.ac.in/misc/docs/published-20papers/etel/etel/405036.pdf
    # adaptive filter
    # Method 3
    # LMS
    # Method 4
    # RLS
    # Method 5
    # NLMS
    # Method 6
    # AEC
    # Method 7
    # NLMS
    # Method 8
    # AEC

    # Algo 1: L1 minimization
    # import sklearn
    #

if __name__ == '__main__':
    #TestSoundSource()
    #TestGPSlike()
    #test_find_peak_interp2()
    #test_GCC_PHAT_one()
    #test_GCC_PHAT_batch()
    #TestSoundSource2D()
    #test_DOA_simu()
    test_DOA_wav()