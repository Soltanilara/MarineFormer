import os

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def COMPUTE_INTERPOLATE_RADIAL(x0, y0, fx, fy, a=0.5, b=0.5, num=8):

    r = np.linspace(0, a / 2.0, num)
    theta = np.linspace(0, 2 * np.pi, num)
    r, theta = np.meshgrid(r, theta)

    XC = x0 + r * np.cos(theta)
    YC = y0 + r * np.sin(theta)
    XC, YC = np.reshape(XC, (-1,)), np.reshape(YC, (-1,))
    VxC = fx.ev(YC, XC)
    VyC = fy.ev(YC, XC)
    return XC, YC, VxC, VyC


def COMPUTE_INTERPOLATE_BOX(x0, y0, fx, fy, a=0.5, b=0.5, num=8):

    X_Box = np.linspace(x0 - a / 2.0, x0 + a / 2.0, num)
    Y_Box = np.linspace(y0 - b / 2.0, y0 + b / 2.0, num)
    XX_Box, YY_Box = np.meshgrid(X_Box, Y_Box)
    XX_Box, YY_Box = np.reshape(XX_Box, (-1,)), np.reshape(YY_Box, (-1,))
    VxB = fx.ev(YY_Box, XX_Box)
    VyB = fy.ev(YY_Box, XX_Box)
    return XX_Box, YY_Box, VxB, VyB


def COMPUTE_KINETIC_ENERGY_BOX(x0, y0, fx, fy, a=0.5, b=0.5, num=8):

    X_Box = np.linspace(x0 - a / 2.0, x0 + a / 2.0, num)
    Y_Box = np.linspace(y0 - b / 2.0, y0 + b / 2.0, num)
    XX_Box, YY_Box = np.meshgrid(X_Box, Y_Box)
    XX_Box, YY_Box = np.reshape(XX_Box, (-1,)), np.reshape(YY_Box, (-1,))
    VxB = fx.ev(YY_Box, XX_Box)
    VyB = fy.ev(YY_Box, XX_Box)
    K_E = np.linalg.norm(np.tanh(np.stack((VxB, VyB), axis=0)), axis=0).sum()

    return K_E


def COMPUTE_CIRCULATION(x0, y0, fx, fy, a=0.5, b=0.5, numT=50):
    t = np.linspace(0, 2 * np.pi, numT)
    xC = a * np.cos(t) + x0
    yC = b * np.sin(t) + y0
    VxC = fx.ev(yC, xC)
    VyC = fy.ev(yC, xC)
    Gamma = -(np.trapz(VxC, xC) + np.trapz(VyC, yC))

    return Gamma


def COMPUTE_VELOCITY(x0, y0, theta, fx, fy, a=0.5, b=0.5, numT=20):
    t = np.linspace((theta - 0.5) % (2 * np.pi), (theta + 0.5) % (2 * np.pi), numT)
    xC = a * np.cos(t) + x0
    yC = b * np.sin(t) + y0
    VxC = fx.ev(yC, xC)
    VyC = fy.ev(yC, xC)
    speed = np.linalg.norm(np.stack((VxC, VyC), axis=0), axis=0)

    return np.max(speed)


def COMPUTE_CIRCULATION2(a, b, x0, y0, numT, Vx, Vy, X, Y):
    t = np.linspace(0, 2 * np.pi, numT)
    xC = a * np.cos(t) + x0
    yC = b * np.sin(t) + y0
    fx = interpolate.RectBivariateSpline(Y, X, Vx)
    fy = interpolate.RectBivariateSpline(Y, X, Vy)
    VxC = fx.ev(yC, xC)
    VyC = fy.ev(yC, xC)
    Gamma = -(np.trapz(VxC, xC) + np.trapz(VyC, yC))

    return Gamma, xC, yC, VxC, VyC


def get_current_flow(
    gamma=[2],
    X0=[0],
    Y0=[0],
    XL=-10,
    XR=10,
    YL=-10,
    YR=10,
    numX=100,
    numY=100,
    Vinf=0,
    alpha=0,
):

    X = np.linspace(XL, XR, numX)
    Y = np.linspace(YL, YR, numY)
    XX, YY = np.meshgrid(X, Y)

    Vx = np.zeros([numX, numY])
    Vy = np.zeros([numX, numY])
    V = np.zeros([numX, numY])

    r = np.zeros([numX, numY])

    for i in range(numX):
        for j in range(numY):
            Vx[i, j] = Vinf * np.cos(alpha * (np.pi / 180))
            Vy[i, j] = Vinf * np.sin(alpha * (np.pi / 180))
            V[i, j] = np.sqrt(Vx[i, j] ** 2 + Vy[i, j] ** 2)

    for k in range(len(gamma)):
        for i in range(numX):
            for j in range(numY):
                x = XX[i, j]
                y = YY[i, j]
                dx = x - X0[k]
                dy = y - Y0[k]
                r = np.sqrt(dx**2 + dy**2)
                Vx[i, j] += (gamma[k] * dy) / (2 * np.pi * r**2)
                Vy[i, j] += (-gamma[k] * dx) / (2 * np.pi * r**2)
                V[i, j] += np.sqrt(Vx[i, j] ** 2 + Vy[i, j] ** 2)

    fx = interpolate.RectBivariateSpline(Y, X, Vx)
    fy = interpolate.RectBivariateSpline(Y, X, Vy)
    return fx, fy


def get_current_flow_vortex_sink(
    gamma=[2],
    X0_V=[0],
    Y0_V=[0],
    lamda=[2],
    X0_S=[0],
    Y0_S=[0],
    XL=-10,
    XR=10,
    YL=-10,
    YR=10,
    numX=100,
    numY=100,
    Vinf=0,
    alpha=0,
):

    X = np.linspace(XL, XR, numX)
    Y = np.linspace(YL, YR, numY)
    XX, YY = np.meshgrid(X, Y)

    Vx = np.zeros([numX, numY])
    Vy = np.zeros([numX, numY])
    V = np.zeros([numX, numY])

    r = np.zeros([numX, numY])

    for i in range(numX):
        for j in range(numY):
            Vx[i, j] = Vinf * np.cos(alpha * (np.pi / 180))
            Vy[i, j] = Vinf * np.sin(alpha * (np.pi / 180))

    for i in range(numX):
        for j in range(numY):
            x = XX[i, j]
            y = YY[i, j]
            for k in range(len(gamma)):
                dx = x - X0_V[k]
                dy = y - Y0_V[k]
                r = np.sqrt(dx**2 + dy**2)
                Vx[i, j] += (gamma[k] * dy) / (2 * np.pi * r**2)
                Vy[i, j] += (-gamma[k] * dx) / (2 * np.pi * r**2)

            for k in range(len(lamda)):
                dx = x - X0_S[k]
                dy = y - Y0_S[k]
                r = np.sqrt(dx**2 + dy**2)
                Vx[i, j] += (lamda[k] * dx) / (2 * np.pi * r**2)
                Vy[i, j] += (lamda[k] * dy) / (2 * np.pi * r**2)
            V[i, j] += np.sqrt(Vx[i, j] ** 2 + Vy[i, j] ** 2)
    fx = interpolate.RectBivariateSpline(Y, X, Vx)
    fy = interpolate.RectBivariateSpline(Y, X, Vy)
    return fx, fy


def render_current_flow(
    gamma=[2],
    X0=[0],
    Y0=[0],
    XL=-10,
    XR=10,
    YL=-10,
    YR=10,
    numX=100,
    numY=100,
    Vinf=0,
    alpha=0,
    ax=None,
):

    X = np.linspace(XL, XR, numX)
    Y = np.linspace(YL, YR, numY)
    fx, fy = get_current_flow(gamma, X0, Y0, XL, XR, YL, YR, numX, numY, Vinf, alpha)
    XX, YY = np.meshgrid(X, Y)
    XX, YY = np.reshape(XX, (-1,)), np.reshape(YY, (-1,))
    Vx = fx.ev(YY, XX)
    Vy = fy.ev(YY, XX)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    M = np.hypot(Vx, Vy)

    ax.quiver(XX, YY, Vx, Vy, pivot="mid", units="inches")
    ax.set_title("Vortex Flow")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.set_xlim([XL, XR])
    ax.set_ylim([YL, YR])

    saveFlnm = "Vortex_Flow_Pos1"
    savePth = "./Figures_Python/" + saveFlnm + ".png"
    plt.savefig(savePth, bbox_inches="tight", facecolor="w", edgecolor="w")
    plt.close()


def render_current_flow2(
    gamma=[2],
    X0=[0],
    Y0=[0],
    XL=-10,
    XR=10,
    YL=-10,
    YR=10,
    numX=100,
    numY=100,
    Vinf=0,
    alpha=0,
    ax=None,
):

    X = np.linspace(XL, XR, numX)
    Y = np.linspace(YL, YR, numY)
    fx, fy = get_current_flow(gamma, X0, Y0, XL, XR, YL, YR, numX, numY, Vinf, alpha)
    XX, YY = np.meshgrid(X, Y)

    Vx = fx.ev(YY, XX)
    Vy = fy.ev(YY, XX)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    M = np.hypot(Vx, Vy)
    ax.quiver(XX, YY, Vx, Vy, pivot="mid", units="inches")

    ax.set_title("Vortex Flow")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.set_xlim([XL, XR])
    ax.set_ylim([YL, YR])

    saveFlnm = "Vortex_Flow_Pos3"
    savePth = "./Figures_Python/" + saveFlnm + ".png"
    plt.savefig(savePth, bbox_inches="tight", facecolor="w", edgecolor="w")
    plt.close()


def render_current_flow_interpolator(
    fx,
    fy,
    Y0=[0],
    XL=-10,
    XR=10,
    YL=-10,
    YR=10,
    numX=100,
    numY=100,
    ax=None,
    save_folder_name="./Figures_Python/",
    save_file_name="flow_fx_fy",
):

    X = np.linspace(XL, XR, numX)
    Y = np.linspace(YL, YR, numY)
    XX, YY = np.meshgrid(X, Y)

    Vx = fx.ev(YY, XX)
    Vy = fy.ev(YY, XX)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    M = np.hypot(Vx, Vy)
    ax.quiver(XX, YY, Vx, Vy, pivot="mid", units="inches")

    ax.set_title("Vortex Flow")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.set_xlim([XL, XR])
    ax.set_ylim([YL, YR])

    save_file = os.path.join(save_folder_name, save_file_name + ".png")
    plt.savefig(save_file, bbox_inches="tight", facecolor="w", edgecolor="w")
    plt.close()


if __name__ == "__main__":
    render_current_flow()
