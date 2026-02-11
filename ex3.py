import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, pi


def create_circle_samples(n, r, noise_var):
    np.random.seed(42)
    samples = []
    
    for _ in range(n):
        phi = np.random.random() * 2.0 * pi
        e = np.random.normal(0, noise_var ** 0.5) # gaussian noise
        x =  r * cos(phi) + e
        y = r * sin(phi) + e
        samples.append(np.array([x, y]))
    assert len(samples) == n
    return samples

def plot_dots(xy: list):
    x = [x for x, _ in xy]
    y = [y for _, y in xy]
    plt.scatter(x, y)
    plt.show()

def sqrt(n):
    return n ** 0.5

def gradient_e(mx, my, rad):
    x = np.array([x for x, _ in xy])
    y = np.array([y for _, y in xy])
    residii = sqrt(2) * (sqrt((x - mx)**2  + (y - my)**2) - rad)

    gdrx = - sqrt(2) * (x - mx) / sqrt((x - mx)**2 + (y - my)**2)
    gdry =  - sqrt(2) * (y - my) / sqrt((x - mx)**2 + (y - my)**2)
    gdrrad = - np.ones(len(x)) * sqrt(2)
    
    J = np.array([gdrx, gdry, gdrrad]).T    
    return J.T@residii


def descent_gradiently(mx, my, r, iterations, dstep) -> tuple[float, float]:
    for i in range(iterations):
        gd = gradient_e(mx, my, r)
        
        d = -gd / np.linalg.norm(gd)
        assert (d @ gd.T) < 0, "Expected negative scalar, received " + (d @ gd)
        assert np.isclose(np.linalg.norm(d), 1), np.linalg.norm(d)
        
        dxyr = d * dstep
        xyr_new = np.array([mx, my, r]) + dxyr
        
        mx, my, r = xyr_new
    return mx, my, r

from commons import crosstest

def descent_gd_e_wrapped(iterations, dstep):
    mx = 0.5
    my = 0.1
    r = 0.9
    mx, my, r = descent_gradiently(mx, my, r, iterations, dstep)
    print(f"{iterations=:>7}\t{dstep=:7}\tx=({mx:>7.3f}, {my:>7.3f}, {r:>7.3f})")
    
