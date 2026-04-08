import torch
import sys
sys.path.append('/home/autodl-tmp/PCKAN_calibration')

def COS_FSVMJ_c1c2(sigma, kappa, theta, eta, thetav, p, q, pu, qd, lambda_, m, n, r, V, rho, T, epsilon, H):
    # 定义复数张量u
    u = torch.tensor(0.0, dtype=torch.complex128, requires_grad=True)

    # 定义A和Delta
    A = 1j * u * r * T

    Delta = torch.zeros(2, dtype=torch.complex128)
    Delta[0] = epsilon[0]**(H[0]-1/2) * sigma[0]
    Delta[1] = epsilon[1]**(H[1]-1/2) * sigma[1]

    # 计算gamma1和gamma2
    part1 = (kappa[0] - 1j * u * rho[0] * Delta[0])**2
    part2 = 1j * u * (1 - 1j * u) * Delta[0]**2
    gamma1 = torch.sqrt(part1 + part2)

    part1 = (kappa[1] - 1j * u * rho[1] * Delta[1])**2
    part2 = 1j * u * (1 - 1j * u) * Delta[1]**2
    gamma2 = torch.sqrt(part1 + part2)

    # 计算B1和B2
    numerator = (kappa[0] - gamma1) * kappa[0] * theta[0] * T / Delta[0]**2
    term1 = -1j * u * rho[0] * kappa[0] * theta[0] * T / Delta[0]
    term2 = 2 * kappa[0] * theta[0] / Delta[0]**2
    z = 2 * gamma1 / (2 * gamma1 + (kappa[0] - gamma1 - 1j * u * rho[0] * Delta[0]) * (1 - torch.exp(-gamma1 * T)))
    term3 = term2 * torch.log(z)
    B1 = numerator + term1 + term3

    numerator = (kappa[1] - gamma2) * kappa[1] * theta[1] * T / Delta[1]**2
    term1 = -1j * u * rho[1] * kappa[1] * theta[1] * T / Delta[1]
    term2 = 2 * kappa[1] * theta[1] / Delta[1]**2
    z = 2 * gamma2 / (2 * gamma2 + (kappa[1] - gamma2 - 1j * u * rho[1] * Delta[1]) * (1 - torch.exp(-gamma2 * T)))
    term3 = term2 * torch.log(z)
    B2 = numerator + term1 + term3

    B = B1 + B2

    # 计算C1和C2
    numerator = 1j * u * (1j * u - 1) * (1 - torch.exp(-gamma1 * T))
    denominator = 2 * gamma1 + (kappa[0] - gamma1 - 1j * u * rho[0] * Delta[0]) * (1 - torch.exp(-gamma1 * T))
    C1 = numerator / denominator * (V[0]**epsilon[0])

    numerator = 1j * u * (1j * u - 1) * (1 - torch.exp(-gamma2 * T))
    denominator = 2 * gamma2 + (kappa[1] - gamma2 - 1j * u * rho[1] * Delta[1]) * (1 - torch.exp(-gamma2 * T))
    C2 = numerator / denominator * (V[1]**epsilon[1])

    C = C1 + C2

    # 计算delta
    sum1 = torch.sum(p * eta / (eta - 1))
    sum2 = torch.sum(q * thetav / (thetav + 1))
    delta = pu * sum1 + qd * sum2 - 1

    # 计算D项
    sum1 = torch.sum(p * eta / (eta - 1j * u))
    sum2 = torch.sum(q * thetav / (thetav + 1j * u))
    D = lambda_ * T * (pu * sum1 + qd * sum2 - 1 - 1j * u * delta)

    # 计算phi_u
    phi_u = A + B + C + D

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

def coeff(k, c, d, a, b):
    chi = torch.exp(d) - torch.exp(c)
    psi = d - c

    auxVar = (b - a) / (k * torch.pi) * (
        torch.sin(k * torch.pi * (d - a) / (b - a)) -
        torch.sin(k * torch.pi * (c - a) / (b - a))
    )

    # Update psi values
    psi[1:] = auxVar[1:]

    # Calculate chi1, chi2, and chi3
    chi1 = 1 / (1 + (k * torch.pi / (b - a)) ** 2)
    chi2 = torch.exp(d) * torch.cos(k * torch.pi * (d - a) / (b - a)) - \
           torch.exp(c) * torch.cos(k * torch.pi * (c - a) / (b - a))
    chi3 = k * torch.pi / (b - a) * (
        torch.exp(d) * torch.sin(k * torch.pi * (d - a) / (b - a)) -
        torch.exp(c) * torch.sin(k * torch.pi * (c - a) / (b - a))
    )

    auxVar = chi1 * (chi2 + chi3)

    chi[1:] = auxVar[1:]

    return chi, psi


def calcvkp(k, b, a, strike):
    chi, psi = coeff(k, a, torch.tensor(0), a, b)

    y = (psi - chi) * strike
    return y

def CF(model, u, T, r, d, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device):
    funobj = cf_bs
    result = funobj(u, T, r, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device)

    return result

def cf_bs(u, T, r, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device):
    # A = 1i * u * r * T
    A = 1j * u * r * T

    # Calculate Delta for two components
    Delta1 = epsilon[0] ** (H[0] - 1 / 2) * sigma[0]
    Delta2 = epsilon[1] ** (H[1] - 1 / 2) * sigma[1]

    # Calculate gamma1 and gamma2
    part1_1 = (kappa[0] - 1j * u * rho[0] * Delta1) ** 2
    part2_1 = 1j * u * (1 - 1j * u) * Delta1 ** 2
    gamma1 = torch.sqrt(part1_1 + part2_1)

    part1_2 = (kappa[1] - 1j * u * rho[1] * Delta2) ** 2
    part2_2 = 1j * u * (1 - 1j * u) * Delta2 ** 2
    gamma2 = torch.sqrt(part1_2 + part2_2)

    # Calculate B1 and B2
    numerator1 = (kappa[0] - gamma1) * kappa[0] * theta[0] * T / Delta1 ** 2
    term1_1 = -1j * u * rho[0] * kappa[0] * theta[0] * T / Delta1
    term2_1 = 2 * kappa[0] * theta[0] / Delta1 ** 2
    z1 = 2 * gamma1 / (2 * gamma1 + (kappa[0] - gamma1 - 1j * u * rho[0] * Delta1) * (1 - torch.exp(-gamma1 * T)))
    term3_1 = term2_1 * torch.log(z1)
    B1 = numerator1 + term1_1 + term3_1

    numerator2 = (kappa[1] - gamma2) * kappa[1] * theta[1] * T / Delta2 ** 2
    term1_2 = -1j * u * rho[1] * kappa[1] * theta[1] * T / Delta2
    term2_2 = 2 * kappa[1] * theta[1] / Delta2 ** 2
    z2 = 2 * gamma2 / (2 * gamma2 + (kappa[1] - gamma2 - 1j * u * rho[1] * Delta2) * (1 - torch.exp(-gamma2 * T)))
    term3_2 = term2_2 * torch.log(z2)
    B2 = numerator2 + term1_2 + term3_2

    B = B1 + B2

    # Calculate C1 and C2
    numerator_C1 = 1j * u * (1j * u - 1) * (1 - torch.exp(-gamma1 * T))
    denominator_C1 = 2 * gamma1 + (kappa[0] - gamma1 - 1j * u * rho[0] * Delta1) * (1 - torch.exp(-gamma1 * T))
    C1 = numerator_C1 / denominator_C1 * (V[0] ** epsilon[0])

    numerator_C2 = 1j * u * (1j * u - 1) * (1 - torch.exp(-gamma2 * T))
    denominator_C2 = 2 * gamma2 + (kappa[1] - gamma2 - 1j * u * rho[1] * Delta2) * (1 - torch.exp(-gamma2 * T))
    C2 = numerator_C2 / denominator_C2 * (V[1] ** epsilon[1])

    C = C1 + C2

    # Calculate D
    sum1 = 0
    sum2 = 0
    for j in range(m):
        sum1 += p[j] * eta[j] / (eta[j] - 1)

    for j in range(n):
        sum2 += q[j] * thetav[j] / (thetav[j] + 1)

    delta = pu * sum1 + qd * sum2 - 1

    sum1_D = 0
    sum2_D = 0
    for j in range(m):
        sum1_D += p[j] * eta[j] / (eta[j] - 1j * u)

    for j in range(n):
        sum2_D += q[j] * thetav[j] / (thetav[j] + 1j * u)

    D = lambda_ * T * (pu * sum1_D + qd * sum2_D - 1 - 1j * u * delta)

    # Final result
    y = A + B + C + D

    return y


def COS_FSVMJ(L, c, cp, model, S0, t, r, strike, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device):
    Ngrid = 2 ** 6  # number of grid points
    Nstrike = 1  # number of strikes

    # Move all tensors to the GPU
    x = torch.log(S0 / strike).repeat(Ngrid, 1).to(device)  # center
    a = c[0] + x - L * torch.sqrt(c[1] + torch.sqrt(c[2])).to(device)  # low bound
    b = c[0] + x + L * torch.sqrt(c[1] + torch.sqrt(c[2])).to(device)  # up bound

    Grid_i = torch.arange(Ngrid).repeat(Nstrike, 1).T.to(device)  # Grid index

    # Define vk_p
    vk_p = lambda x: calcvkp(x, b, a, strike).to(device)  # coefficients for put

    # Compute fk_i
    fk_i = torch.exp(CF(model, Grid_i * torch.pi / (b - a), t, r, q, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device)).to(device)
    fk_i = 2 / (b - a) * torch.real(fk_i * torch.exp(1j * torch.pi * Grid_i * x / (b - a)) *
                                    torch.exp(-1j * (torch.pi * Grid_i * a / (b - a))))

    Vk = vk_p(Grid_i)

    # Compute y
    y = torch.exp(-r * t) * (torch.sum(fk_i * Vk, dim=0) - 0.5 * torch.sum(fk_i[0, :] * Vk[0, :]))

    if cp == 1:  # European call price using put-call parity
        y = y + S0 - strike * torch.exp(-r * t)

    return y


def FVSJ_fun(option_params, FVSJ_params, device):
    L = 10
    model = "FSVMJ"
    cp, T, strike, S0, r = option_params
    sigma = torch.tensor([FVSJ_params[0], FVSJ_params[1]], device=device)
    kappa = torch.tensor([FVSJ_params[2], FVSJ_params[3]], device=device)
    theta = torch.tensor([FVSJ_params[4], FVSJ_params[5]], device=device)
    rho = torch.tensor([FVSJ_params[6], FVSJ_params[7]], device=device)
    V = torch.tensor([FVSJ_params[8], FVSJ_params[9]], device=device)
    eta = torch.tensor([FVSJ_params[10], FVSJ_params[11]], device=device)
    epsilon = torch.tensor([FVSJ_params[12], FVSJ_params[13]], device=device)
    H = torch.tensor([FVSJ_params[14], FVSJ_params[15]], device=device)
    thetav = torch.tensor([FVSJ_params[16], FVSJ_params[17]], device=device)
    p = torch.tensor([FVSJ_params[18], FVSJ_params[19]], device=device)
    q = torch.tensor([FVSJ_params[20], FVSJ_params[21]], device=device)
    pu = torch.tensor(FVSJ_params[22], device=device)
    qd = torch.tensor(FVSJ_params[23], device=device)
    lambda_ = torch.tensor(FVSJ_params[24], device=device)
    m = torch.tensor(2,  device=device)
    n = torch.tensor(2,  device=device)

    c1, c2 = COS_FSVMJ_c1c2(sigma, kappa, theta, eta, thetav, p, q, pu, qd, lambda_, m, n, r, V, rho, T, epsilon, H)
    c4 = 0
    c = torch.tensor([-c1, c2, c4])
    price = COS_FSVMJ(L, c, cp, model, S0, T, r, strike, sigma, kappa, theta, rho, V, eta, epsilon, H, thetav, p, q, pu, qd, lambda_, m, n, device)

    return price

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FVSJ_p = [0.9, 0.9, 12, 16, 0.05, 0.03, -0.5, -0.5, 0.05, 0.02, 50, 50, 0.02, 0.02, 0.8, 0.7, 20, 20, 1.3, -0.3, 1.2, -0.2, 0.4, 0.6, 1, 2, 2]
    for i in range(80, 130, 10):
        option_p = torch.tensor([-1, 2, i, 100, 0.05])
        print(FVSJ_fun(option_p, FVSJ_p, device))