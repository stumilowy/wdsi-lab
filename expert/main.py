import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')

# Funkcja przeliczająca stan z Twojego screena na współrzędne (x, y, z) sfery Blocha
def oblicz_wspolrzedne_blocha(alpha_mod, theta_deg, phi_deg):
    # Przeliczenie stopni na radiany
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    # Warunek normalizacji: |alpha|^2 + |beta|^2 = 1
    beta_mod = np.sqrt(max(0.0, 1.0 - alpha_mod ** 2))

    # Pełne zespolone amplitudy dokładnie jak na screenie:
    # Stan = (e^(i*theta) * alpha)|0> + (e^(i*phi) * beta)|1>
    A = np.exp(1j * theta) * alpha_mod
    B = np.exp(1j * phi) * beta_mod

    # Przeliczenie na współrzędne kartezjańskie sfery Blocha
    x = 2 * np.real(np.conj(A) * B)
    y = 2 * np.imag(np.conj(A) * B)
    z = np.abs(A) ** 2 - np.abs(B) ** 2
    return x, y, z, beta_mod


# Konfiguracja okna i wykresu 3D
fig = plt.figure(figsize=(9, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.32)  # Miejsce na suwaki na dole


def rysuj_sfere():
    ax.clear()
    # Siatka sfery
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", alpha=0.3)

    # Równik dla lepszej orientacji przestrzennej
    eq_angle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(eq_angle), np.sin(eq_angle), 0, color="gray", linestyle="dashed", alpha=0.6)

    # Osie X, Y, Z
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='black', arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='black', arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='black', arrow_length_ratio=0.06)

    ax.text(1.4, 0, 0, "X (|+⟩)", fontweight='bold')
    ax.text(0, 1.4, 0, "Y (|+i⟩)", fontweight='bold')
    ax.text(0, 0, 1.4, "Z (|0⟩)", fontweight='bold')
    ax.text(0, 0, -1.4, "|1⟩", fontweight='bold')

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1, 1, 1])


# Wartości początkowe (stan superpozycji 50/50, kąty = 0)
init_alpha = np.sqrt(0.5)  # ok. 0.707
init_theta = 0.0
init_phi = 0.0

# Tworzenie osi dla suwaków
ax_theta = plt.axes([0.25, 0.20, 0.55, 0.03])
ax_phi = plt.axes([0.25, 0.15, 0.55, 0.03])
ax_alpha = plt.axes([0.25, 0.10, 0.55, 0.03])

# Definicja suwaków
s_theta = Slider(ax_theta, 'Kąt θ (przy |0⟩) [°]', 0.0, 360.0, valinit=init_theta, color='red')
s_phi = Slider(ax_phi, 'Kąt φ (przy |1⟩) [°]', 0.0, 360.0, valinit=init_phi, color='gold')
s_alpha = Slider(ax_alpha, 'Wartość α (0 do 1)', 0.0, 1.0, valinit=init_alpha, color='skyblue')


# Funkcja aktualizująca wykres przy ruchu suwakiem
def aktualizuj(val):
    rysuj_sfere()
    a_val = s_alpha.val
    t_val = s_theta.val
    p_val = s_phi.val

    x, y, z, b_val = oblicz_wspolrzedne_blocha(a_val, t_val, p_val)

    # Rysowanie głównego wektora stanu
    ax.quiver(0, 0, 0, x, y, z, color='crimson', linewidth=4, arrow_length_ratio=0.12)

    # Rzut wektora na płaszczyznę równika (ułatwia obserwację kąta fazowego)
    ax.plot([0, x], [0, y], [0, 0], color='blue', linestyle='dotted', linewidth=2)
    ax.plot([x, x], [y, y], [0, z], color='gray', linestyle='dotted', linewidth=1.5)

    roznica_faz = (p_val - t_val) % 360
    ax.set_title(
        f"Stan: $e^{{i\\cdot{t_val:.0f}^\\circ}}({a_val:.2f})|0\\rangle + e^{{i\\cdot{p_val:.0f}^\\circ}}({b_val:.2f})|1\\rangle$\n"
        f"Różnica faz (φ - θ) = {roznica_faz:.0f}°  |  P(|0⟩) = {a_val ** 2 * 100:.0f}%,  P(|1⟩) = {b_val ** 2 * 100:.0f}%",
        fontsize=11
    )
    fig.canvas.draw_idle()


# Podpięcie funkcji pod suwaki
s_theta.on_changed(aktualizuj)
s_phi.on_changed(aktualizuj)
s_alpha.on_changed(aktualizuj)

# Pierwsze rysowanie
aktualizuj(None)
plt.show()