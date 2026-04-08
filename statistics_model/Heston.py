import torch
import sys
sys.path.append('/home/autodl-tmp/PCKAN_calibration')

def COS_Heston_c1c2_torch(sigma, r, v0, rho, T, thetav, alphav, device):
    # 定义 u 为复数类型张量
    u = torch.tensor(0.0, dtype=torch.complex128, requires_grad=True).to(device)

    # 计算 m 和 n
    m = torch.sqrt((alphav - 1j * u * rho * sigma) ** 2 + 1j * u * (1 - 1j * u) * sigma ** 2).to(device)
    n = 2 * m + (alphav - m - 1j * u * rho * sigma) * (1 - torch.exp(-m * T)).to(device)

    # 计算 A, B, C
    A = torch.log((2 * m / n) ** (2 * (thetav * alphav) / sigma ** 2)).to(device)
    B = (((thetav * alphav) * (alphav - m) * T) / sigma ** 2 - (1j * u * rho * (thetav * alphav) * T) / sigma).to(device)
    C = ((1j * u * (1j * u - 1) * (1 - torch.exp(-m * T))) / n).to(device)

    # 计算 phi_u
    phi_u = A + C * v0 + B + 1j * u * r * T

    phi_u_real = phi_u.real
    phi_u_imag = phi_u.imag

    # 计算 phi_u_real 相对于 u 的一阶导数
    dphi_du_real = torch.autograd.grad(phi_u_real, u, create_graph=True)[0]  # 计算实部的一阶导数
    dphi_du_imag = torch.autograd.grad(phi_u_imag, u, create_graph=True)[0]  # 计算虚部的一阶导数

    # 计算 c1（在 u = 0 时，dphi_du 的虚部）
    c1_real = dphi_du_real.imag  # 从实部导数中提取虚部
    c1_imag = dphi_du_imag.imag  # 从虚部导数中提取虚部

    # 计算 c1
    c1 = c1_real + c1_imag

    # **计算二阶导数**
    # 计算 phi_u_real 相对于 u 的二阶导数
    d2phi_du2_real = torch.autograd.grad(dphi_du_real.real, u, create_graph=True)[0]  # 计算实部的二阶导数
    d2phi_du2_imag = torch.autograd.grad(dphi_du_imag.real, u, create_graph=True)[0]  # 计算虚部的二阶导数

    # 计算 c2（在 u = 0 时，d2phi_du2 的负实部）
    c2_real = -d2phi_du2_real.real
    c2_imag = -d2phi_du2_imag.real

    # 计算 c2
    c2 = c2_real + c2_imag

    return float(c1), float(c2)


# COS_Heston pricing function
def COS_Heston_fun_torch(n, L, c, cp, model, S0, T, r, q, strike, sigma, V0, rho, thetav, alphav, device):
    Ngrid = 2 ** n
    x = torch.full((Ngrid, 1), torch.log(S0 / strike), dtype=torch.float32).to(device)
    a = c[0] + x - L * torch.sqrt(c[1] + torch.sqrt(c[2])).to(device)
    b = c[0] + x + L * torch.sqrt(c[1] + torch.sqrt(c[2])).to(device)

    Grid_i = torch.arange(Ngrid, dtype=torch.float32).reshape(-1, 1).to(device)
    fk_i = torch.exp(CF_torch(model, Grid_i * torch.pi / (b - a), T, r, q, sigma, V0, rho, thetav, alphav)).to(device)
    fk_i = 2 / (b - a) * torch.real(
        fk_i * torch.exp(1j * torch.pi * Grid_i * x / (b - a)) * torch.exp(-1j * torch.pi * Grid_i * a / (b - a))).to(device)

    Vk = calcvkp_torch(Grid_i, b, a, strike).to(device)
    y = torch.exp(-r * T) * (torch.sum(fk_i * Vk) - 0.5 * (fk_i[0, :] * Vk[0, :])).to(device)

    if cp == 1:  # Using put-call parity to calculate the call price
        y += S0 * torch.exp(-q * T) - strike * torch.exp(-r * T)

    return y


# Characteristic function
def CF_torch(model, u, T, r, d, sigma, V0, rho, thetav, alphav):
    if model == 'Heston':
        funobj = cf_bs_torch
    else:
        raise ValueError("Undefined model")

    return funobj(u, T, r, d, sigma, V0, rho, thetav, alphav)


# Black-Scholes characteristic function
def cf_bs_torch(u, T, r, d, sigma, V0, rho, thetav, alphav):
    m = torch.sqrt((alphav - 1j * u * rho * sigma) ** 2 + 1j * u * (1 - 1j * u) * sigma ** 2)
    n = 2 * m + (alphav - m - 1j * u * rho * sigma) * (1 - torch.exp(-m * T))
    A = torch.log((2 * m / n) ** (2 * (thetav * alphav) / sigma ** 2))
    B = ((thetav * alphav) * (alphav - m) * T) / sigma ** 2 - (1j * u * rho * (thetav * alphav) * T) / sigma
    C = (1j * u * (1j * u - 1) * (1 - torch.exp(-m * T))) / n
    return A + C * V0 + B + 1j * u * r * T


# Calculate option coefficients
def calcvkp_torch(k, b, a, strike):
    chi, psi = coeff_torch(k, a, 0, a, b)
    return (psi - chi) * strike


# Calculate COS coefficients
def coeff_torch(k, c, d, a, b):
    d = torch.tensor(d)
    chi = (torch.exp(d) - torch.exp(c))
    psi = (d - c)

    auxVar = (b - a) / (k * torch.pi) * (torch.sin(k * torch.pi * (d - a) / (b - a)) - torch.sin(k * torch.pi * (c - a) / (b - a)))
    psi[1:] = auxVar[1:]

    chi1 = 1 / (1 + (k * torch.pi / (b - a)) ** 2)
    chi2 = torch.exp(d) * torch.cos(k * torch.pi * (d - a) / (b - a)) - torch.exp(c) * torch.cos(k * torch.pi * (c - a) / (b - a))
    chi3 = k * torch.pi / (b - a) * (
        torch.exp(d) * torch.sin(k * torch.pi * (d - a) / (b - a)) - torch.exp(c) * torch.sin(k * torch.pi * (c - a) / (b - a)))

    auxVar = chi1 * (chi2 + chi3)

    chi[1:] = auxVar[1:]

    return chi, psi

def Heston_Price_torch_c(option_params, params, device):
    n = 6
    L = 10
    q = 0
    model_type = "Heston"

    # Unpack parameters
    cp, T, strike, S0, r = option_params
    sigma, v0, rho, thetav, alphav = params

    # Calculate c1 and c2 using the COS_Heston_c1c2_fun_torch
    c1, c2 = COS_Heston_c1c2_torch(sigma, r, v0, rho, T, thetav, alphav, device)
    c4 = 0  # Not used, but part of the input format
    c = torch.tensor([-c1, c2, c4])
    priceEuCOS = COS_Heston_fun_torch(n, L, c, cp, model_type, S0, T, r, q, strike, sigma, v0, rho, thetav, alphav, device)
    return priceEuCOS


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # heston sigma v0 rho theta kappa
    params = torch.tensor([100, 0.05, 1/6, 80, -1, 0.2, 1, -0.5, 2, 0.3]).to(device)

    print(Heston_Price_torch_c(params, device))