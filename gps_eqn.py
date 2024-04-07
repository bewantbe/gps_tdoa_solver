# Solve gps equation

# Use JAX for derivatives
import jax                # JAX >= 0.4.26
import jax.numpy as jnp
from jax import grad
from jax import jacfwd, jacrev

_ja = lambda x: jnp.array(x)

import logging
logging.getLogger("jax").setLevel(logging.ERROR)
# old
# Ref. https://github.com/google/jax/issues/10070
#from absl import logging
#logging.set_verbosity(logging.ERROR)

# set JAX to use CPU only
jax.config.update('jax_default_device', jax.devices('cpu')[0])

logging.getLogger("jax").setLevel(logging.WARNING)

import numpy as np
_a = lambda x: np.array(x)
_ia = lambda x: np.array(x, dtype=int)

# parameters (known): m = number of satellites, p = positions, t = time
# p1, p2, ..., pm, ct1, ct2, ..., ctm

# to solve
# q = [rx, ry, rz, ct0]

def gen_sound_data_5p():
    """Generate example microphone configuration with sound source position."""
    # sound source position (unit: m)
    r_true = np.array([0.1, 0.2, -0.5])
    # temperature (unit: degree Celsius)
    temp = 20
    # speed of sound (unit: m/s)
    # Ref. [Air - Speed of Sound vs. Temperature]
    #      (https://www.engineeringtoolbox.com/air-speed-sound-d_603.html)
    # Ref. https://sengpielaudio.com/calculator-airpressure.htm
    c = 20.05 * (273.16 + temp) ** 0.5
    print(f'Sound speed {c:.2f} m/s, at temperature {temp} C.')

    m = 5            # number of microphones

    p = np.zeros((m, 3))     # microphone positions           (unit: m)
    ct = np.zeros(m)         # relative time delay of arrival (unit: s)

    # exact positions and time delays
    p[0] = _a([0, 0, 0])
    ct[0] = np.linalg.norm(r_true - p[0])
    p[1] = _a([0.1, 0, 0])
    ct[1] = np.linalg.norm(r_true - p[1]) - ct[0]
    p[2] = _a([0, 0.1, 0])
    ct[2] = np.linalg.norm(r_true - p[2]) - ct[0]
    p[3] = _a([-0.1, 0, 0])
    ct[3] = np.linalg.norm(r_true - p[3]) - ct[0]
    p[4] = _a([0, -0.1, 0])
    ct[4] = np.linalg.norm(r_true - p[4]) - ct[0]

    ct0 = ct[0]
    ct[0] = ct[0] - ct0

    # noise level
    err_pos = 0.2e-3                 # default 0.2mm
    sample_rate = 250e3              # default 250kHz
    err_ct = c * 2.0 / sample_rate   # default 2 samples

    # add measurement noise
    for j in range(m):
        p[j]  = p[j]  + err_pos * np.random.randn(3)
        ct[j] = ct[j] + err_ct  * np.random.randn()

    return p, ct, r_true, ct0

p, ct, r_true, ct0 = gen_sound_data_5p()
#print(p)
#print(ct)

def GPS_F(p, ct, r, ct0):
    m = len(ct)
    v = jnp.array([
        jnp.linalg.norm(r - p[j]) ** 2 - (ct0 + ct[j]) ** 2
        for j in range(m)
    ])
    return v

def GPS_dF_dr(p, ct, r, ct0):
    m = len(ct)
    dF = jnp.array([
        2 * (r - p[j])
        for j in range(m)
    ])
    return dF
#print(_a([2*(r_true - p[j]) for j in range(5)]))

def GPS_dF_dt(p, ct, r, ct0):
    m = len(ct)
    dF = jnp.array([
        -2 * (ct0 + ct[j])
        for j in range(m)
    ])
    return dF

def GPS_dF_dparam(p, ct, r, ct0):
    J = jnp.hstack([GPS_dF_dr(p, ct, r, ct0),
                    GPS_dF_dt(p, ct, r, ct0)[:, jnp.newaxis]])
    return J

print('GPS_F =', GPS_F(p, ct, r_true, ct0))

if 0:
    # For verification of derivatives

    # Get jacobian of GPS_F with respect to r
    dF_dr_auto = jacrev(GPS_F, argnums=2)
    dF_dt_auto = jacrev(GPS_F, argnums=3)

    print(dF_dr_auto(p, ct, r_true, ct0))
    print(dF_dt_auto(p, ct, r_true, ct0))

    print(GPS_dF_dr(p, ct, r_true, ct0))
    print(GPS_dF_dt(p, ct, r_true, ct0))

def NewtonIter(F, dF_dx, x_init, max_iter=10, tol=1e-7):
    """Newton's method for solving F(x) = 0 in least square sense."""
    x = x_init
    #print('x0', x)
    for i in range(max_iter):
        J = dF_dx(x)
        F_val = F(x)
        delta = jnp.linalg.lstsq(J, -F_val, rcond=None)[0].flatten()
        x = x + delta
        #print('x', x)
        if jnp.linalg.norm(delta) < tol:
            break
    #print('n_iter =', i)

    return x

def NewtonIterGPS(p, ct, r_n, ct0_n):
    #q0 = jnp.hstack([r_n, ct0_n])
    q0 = np.hstack([r_n, ct0_n])
    F  = lambda q: GPS_F(p, ct, q[:3], q[3])
    dF = lambda q: GPS_dF_dparam(p, ct, q[:3], q[3])
    q = NewtonIter(F, dF, q0)
    return q[:3], q[3]

r_n, ct0_n = NewtonIterGPS(p, ct, r_true, ct0)
print('Solution:')
print('r_n', r_n)
print('ct0_n', ct0_n)

if 0:
    # For verification of derivatives
    dF_dp_auto = jacrev(GPS_F, argnums=0)
    print(dF_dp_auto(p, ct, r_true, ct0))
    j = 0
    print(-2*(r_true-p[j]), -2*(ct[j]+ct0))

def GetK(r, ct0, p, ct, err_pos, err_ct):
    m = len(ct)
    K = jnp.array([
        jnp.dot(-2*(r_true-p[j]), p[j]) - 2*(ct[j]+ct0) * ct[j]
        for j in range(m)
    ])
    return K
