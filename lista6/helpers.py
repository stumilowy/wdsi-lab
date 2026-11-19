import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import sympy as sp
import folium


def generate_noisy_poly_samples(degree=20,
                                n_samples=300,
                                x_range=(-1.0, 1.0),
                                low=-1.0,
                                high=1.0,
                                noise_std=0.1,
                                seed=42):
    rng = np.random.default_rng(seed)
    # losowe współczynniki
    poly_coeffs = rng.uniform(low=low, high=high, size=(degree + 1,))
    if np.abs(poly_coeffs[0]) < 1e-12:
        poly_coeffs[0] += np.sign(rng.normal()) * 1e-3
    poly_coeffs = poly_coeffs + np.linspace(0, degree/2, degree+1)[::-1]
    # uniform sampling
    x_samples = rng.uniform(x_range[0], x_range[1], size=n_samples)
    x_samples = np.sort(x_samples)

    y_true = np.polyval(poly_coeffs, x_samples)
    y_noisy = y_true + rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    poly20_df = pd.DataFrame({'x': x_samples, 'y': y_noisy})

    return poly_coeffs, x_samples, y_noisy, y_true, poly20_df


def make_parabolic_example(f, df, learning_rate, compute_new_x, start_x=0.0, range_x=(-1, 7), iterations=25):
    # Najpierw obliczamy całą historię kroków, a dopiero potem ją animujemy.
    idx = [0, iterations//4, iterations//2, 3*iterations//4, iterations]
    history_x = [start_x]
    current_x = start_x
    for _ in range(iterations):
        current_x = compute_new_x(current_x, df, learning_rate)
        history_x.append(current_x)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.linspace(range_x[0], range_x[1], 200)
    y_vals = f(x_vals)

    ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.6, label='Funkcja kosztu')
    ax.set_xlim(range_x)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Parametr x')
    ax.set_ylabel('Koszt f(x)')
    ax.set_title('Animacja Gradient Descent')
    ax.grid(True)

    point, = ax.plot([], [], 'ro', markersize=12, label='Aktualna pozycja', zorder=5)
    tangent_line, = ax.plot([], [], 'g--', linewidth=2, label='Styczna (nachylenie / pochodna)')
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    def update(frame):
        x_curr = history_x[frame]
        y_curr = f(x_curr)
        grad = df(x_curr)
        point.set_data([x_curr], [y_curr])
        tangent_range = 1.5 # Długość linii w lewo i w prawo
        x_tan = np.linspace(x_curr - tangent_range, x_curr + tangent_range, 10)
        y_tan = grad * (x_tan - x_curr) + y_curr
        tangent_line.set_data(x_tan, y_tan)
        
        # Zaktualizuj tekst
        text_info.set_text(f'Iteracja: {frame}\n'
                        f'x = {x_curr:.4f}\n'
                        f'Gradient = {grad:.4f}')
        
        return point, tangent_line, text_info

    anim = FuncAnimation(fig, update, frames=len(history_x), interval=500, blit=True)
    plt.close()
    
    # Tekst
    print("Generowanie animacji...")
    print(f"Podsumowanie optymalizacji:\n- Start: x = {history_x[0]:.4f}\n- Koniec (po {iterations} iteracjach): x = {history_x[-1]:.4f}\n- Gradient końcowy: {df(history_x[-1]):.4f}")
    df_summary = pd.DataFrame({
        'Iteracja': [0, iterations//4  , iterations//2, 3*iterations//4, iterations],
        'Pozycja x': [history_x[0], history_x[iterations//4], history_x[iterations//2], history_x[3*iterations//4], history_x[-1]],
        "Pochodna f'(x)": [df(history_x[i]) for i in idx],
        "Długość kroku η·|f'(x)|": [learning_rate * abs(df(history_x[i])) for i in idx],
        'Koszt f(x)': [f(history_x[0]), f(history_x[iterations//4]), f(history_x[iterations//2]), f(history_x[3*iterations//4]), f(history_x[-1])]
    })
    print("Tabela przebiegu optymalizacji (kluczowe kroki):")
    display(df_summary)
    return HTML(anim.to_jshtml())


def make_higher_order_example(f_num, f_grad_num, scenarios, learning_rate, compute_new_x, steps=50, show_trace=True):
    all_histories = []

    for s in scenarios:
        history = [s["start"]]
        curr_x = s["start"]
        for _ in range(steps):
            curr_x = compute_new_x(curr_x, f_grad_num, learning_rate)
            history.append(curr_x)
        all_histories.append(history)

    # Przygotowanie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.linspace(-2.3, 2.3, 400)
    y_vals = f_num(x_vals)

    # Tło
    ax.plot(x_vals, y_vals, 'k-', alpha=0.3, linewidth=2, label='f(x)')
    ax.set_title(f"Gradient Descent (Trace: {'ON' if show_trace else 'OFF'})")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-1, 9)
    ax.grid(True, alpha=0.3)

    # Inicjalizacja obiektów graficznych
    points_plots = []
    labels_plots = []
    trails_plots = []  # Nowa lista na ślady

    for s in scenarios:
        t, = ax.plot([], [], '-', color=s["color"], linewidth=3, alpha=0.6)
        trails_plots.append(t)
        p, = ax.plot([], [], 'o', color=s["color"], markersize=10)
        points_plots.append(p)
        l = ax.text(0, 0, s["name"], color=s["color"], fontweight='bold', ha='center', fontsize=9)
        labels_plots.append(l)

    def update(frame):
        artists = []
        for i, history in enumerate(all_histories):
            idx = min(frame, len(history)-1)
            x_curr = history[idx]
            y_curr = f_num(x_curr)
            if show_trace:
                path_x = history[:idx+1]
                path_y = f_num(np.array(path_x))
                trails_plots[i].set_data(path_x, path_y)
                artists.append(trails_plots[i])
            points_plots[i].set_data([x_curr], [y_curr])
            artists.append(points_plots[i])
            labels_plots[i].set_position((x_curr, y_curr + 0.6))
            artists.append(labels_plots[i])
            
        return artists

    anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
    plt.close()

    
    # Tekst
    print(f"Podsumowanie optymalizacji dla podanych scenariuszy ({len(scenarios)}):")
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        print(f"[{s['name']}] Start: x = {history[0]:.4f}, Koniec (po {steps} iteracjach): x = {history[-1]:.4f}, Gradient końcowy: {f_grad_num(history[-1]):.4f}")
    df_list = []
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        df_list.append(pd.DataFrame({
            'Scenariusz': [s['name']]*3,
            'Iteracja': [0, steps//2, steps],
            'Pozycja x': [history[0], history[steps//2], history[-1]],
            'Koszt f(x)': [f_num(history[0]), f_num(history[steps//2]), f_num(history[-1])]
        }))
    if df_list:
        print("Tabela przebiegu optymalizacji (kluczowe kroki dla każdego scenariusza):")
        display(pd.concat(df_list, ignore_index=True))
    return HTML(anim.to_jshtml())


def make_2d_parabolic_example(f_num, grads_num, scenarios, learning_rate, compute_new_position_with_gradient, steps=50, show_trace=True):
    all_histories = []
    for s in scenarios:
        pos = s["start"]
        history = [pos] # Lista pozycji [x, y]
        for _ in range(steps):
            pos = compute_new_position_with_gradient(pos, grads_num, learning_rate)
            history.append(pos)
        all_histories.append(np.array(history))

    fig = plt.figure(figsize=(16, 7))

    # Mapa konturowa
    ax2d = fig.add_subplot(1, 2, 1)
    x_range = np.linspace(-2.5, 2.5, 150)
    y_range = np.linspace(-2.5, 2.5, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f_num((X, Y))

    # Rysujemy mapę konturową
    contour = ax2d.contourf(X, Y, Z, levels=35, cmap='viridis')
    fig.colorbar(contour, ax=ax2d, shrink=0.6, label='Funkcja Kosztu')
    ax2d.set_title("Widok z góry (2D)")
    ax2d.set_xlabel("Parametr x")
    ax2d.set_ylabel("Parametr y")

    # Wykres 3D
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # Rysujemy powierzchnię (tylko raz, jako tło)
    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, 
                            edgecolor='none', rstride=2, cstride=2)
    ax3d.set_title("Widok przestrzenny (3D)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("Koszt")
    ax3d.set_zlim(-1.5, 46)
    # Ustawiamy widok, żeby widzieć doliny
    ax3d.view_init(elev=0, azim=-60)

    # Obiekty dla 2D i 3D przechowujemy osobno
    lines_2d, points_2d = [], []
    lines_3d, points_3d = [], []

    for s in scenarios:
        l2d, = ax2d.plot([], [], '-', color=s["color"], lw=1.5, alpha=0.8)
        p2d, = ax2d.plot([], [], 'o', color=s["color"], markeredgecolor='black', markersize=8)
        lines_2d.append(l2d)
        points_2d.append(p2d)
        
        l3d, = ax3d.plot([], [], [], '-', color=s["color"], lw=2, alpha=0.9)
        p3d, = ax3d.plot([], [], [], 'o', color=s["color"], markeredgecolor='black', markersize=8)
        lines_3d.append(l3d)
        points_3d.append(p3d)

    # Animacja
    def update(frame):
        artists = []
        
        for i, history in enumerate(all_histories):
            idx = min(frame, len(history)-1)
            
            # Dane do aktualnej klatki
            # shape (N, 2) -> path_x, path_y
            path_x = history[:idx+1, 0]
            path_y = history[:idx+1, 1]
            curr_x = path_x[-1]
            curr_y = path_y[-1]
            
            # Obliczamy Z (wysokość) dla widoku 3D
            path_z = f_num((path_x, path_y))
            curr_z = path_z[-1]
            
            # 2D
            if show_trace:
                lines_2d[i].set_data(path_x, path_y)
                artists.append(lines_2d[i])
            
            points_2d[i].set_data([curr_x], [curr_y])
            artists.append(points_2d[i])
            
            # 3D
            if show_trace:
                lines_3d[i].set_data(path_x, path_y)
                lines_3d[i].set_3d_properties(path_z)
                artists.append(lines_3d[i])
                
            points_3d[i].set_data([curr_x], [curr_y])
            points_3d[i].set_3d_properties([curr_z])
            artists.append(points_3d[i])
            
        return artists

    # Generowanie
    print("Generowanie animacji 2D + 3D...")
    anim = FuncAnimation(fig, update, frames=steps, interval=80, blit=True)
    plt.close()

    
    # Tekst
    print(f"Podsumowanie optymalizacji (przestrzeń 2D) dla podanych scenariuszy:")
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        start_pos = history[0]
        end_pos = history[-1]
        start_cost = f_num((start_pos[0], start_pos[1]))
        end_cost = f_num((end_pos[0], end_pos[1]))
        print(f"[{s.get('name', s['color'])}] Start: (x={start_pos[0]:.4f}, y={start_pos[1]:.4f}), Koszt początkowy: {start_cost:.4f}")
        print(f"        Koniec (po {steps} iteracjach): (x={end_pos[0]:.4f}, y={end_pos[1]:.4f}), Koszt końcowy: {end_cost:.4f}")
    
    df_list = []
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        df_list.append(pd.DataFrame({
            'Scenariusz (Kolor)': [s.get('name', s['color'])]*5,
            'Iteracja': [0, steps//4, steps//2, 3*steps//4, steps],
            'Pozycja x': [history[0, 0], history[steps//4, 0], history[steps//2, 0], history[3*steps//4, 0], history[-1, 0]],
            'Pozycja y': [history[0, 1], history[steps//4, 1], history[steps//2, 1], history[3*steps//4, 1], history[-1, 1]],
            'Koszt f(x,y)': [f_num((history[0, 0], history[0, 1])), f_num((history[steps//4, 0], history[steps//4, 1])), f_num((history[steps//2, 0], history[steps//2, 1])), f_num((history[3*steps//4, 0], history[3*steps//4, 1])), f_num((history[-1, 0], history[-1, 1]))]
        }))
    if df_list:
        print("Tabela przebiegu optymalizacji (kluczowe kroki):")
        display(pd.concat(df_list, ignore_index=True))
    return HTML(anim.to_jshtml())


def make_3D_higher_order_example(f_num, grads, scenarios, learning_rate, compute_new_position_with_gradient, steps=50, show_trace=True):
    # Pre-kalkulacja ścieżek (Symulacja)
    # Obliczamy wszystko przed animacją dla płynności
    all_histories = []
    for s in scenarios:
        pos = s["start"]
        history = [pos] # Lista pozycji [x, y]
        for _ in range(steps):
            pos = compute_new_position_with_gradient(pos, grads, learning_rate)
            history.append(pos)
        all_histories.append(np.array(history))

    # Przygotowanie Sceny (Figury i Subploty)
    fig = plt.figure(figsize=(16, 7))

    ax2d = fig.add_subplot(1, 2, 1)
    x_range = np.linspace(-4, 4, 150)
    y_range = np.linspace(-2.5, 2.5, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f_num((X, Y))

    # Rysujemy mapę konturową
    contour = ax2d.contourf(X, Y, Z, levels=35, cmap='viridis')
    fig.colorbar(contour, ax=ax2d, shrink=0.6, label='Funkcja Kosztu')
    ax2d.set_title("Widok z góry (2D)")
    ax2d.set_xlabel("Parametr x")
    ax2d.set_ylabel("Parametr y")

    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # Rysujemy powierzchnię (tylko raz, jako tło)
    # rstride/cstride = 2 dla optymalizacji wydajności renderowania
    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, 
                            edgecolor='none', rstride=2, cstride=2)
    ax3d.set_title("Widok przestrzenny (3D)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("Koszt")
    ax3d.set_zlim(-1.5, 6)
    # Ustawiamy widok, żeby widzieć doliny
    ax3d.view_init(elev=0, azim=-60)

    # Inicjalizacja Obiektów Animowanych (Kropki i Linie)
    # Musimy przechowywać obiekty dla 2D i 3D osobno
    lines_2d, points_2d = [], []
    lines_3d, points_3d = [], []

    for s in scenarios:
        # 2D
        l2d, = ax2d.plot([], [], '-', color=s["color"], lw=1.5, alpha=0.8)
        p2d, = ax2d.plot([], [], 'o', color=s["color"], markeredgecolor='black', markersize=8)
        lines_2d.append(l2d)
        points_2d.append(p2d)
        
        # W 3D używamy ax.plot zamiast ax.scatter dla łatwiejszej animacji
        l3d, = ax3d.plot([], [], [], '-', color=s["color"], lw=2, alpha=0.9)
        p3d, = ax3d.plot([], [], [], 'o', color=s["color"], markeredgecolor='black', markersize=8)
        lines_3d.append(l3d)
        points_3d.append(p3d)

    # Pętla Animacji
    def update(frame):
        artists = []
        
        for i, history in enumerate(all_histories):
            idx = min(frame, len(history)-1)
            
            # Dane do aktualnej klatki
            # shape (N, 2) -> path_x, path_y
            path_x = history[:idx+1, 0]
            path_y = history[:idx+1, 1]
            curr_x = path_x[-1]
            curr_y = path_y[-1]
            
            # Obliczamy Z (wysokość) dla widoku 3D
            path_z = f_num((path_x, path_y))
            curr_z = path_z[-1]
            
            # 2D
            if show_trace:
                lines_2d[i].set_data(path_x, path_y)
                artists.append(lines_2d[i])
            
            points_2d[i].set_data([curr_x], [curr_y])
            artists.append(points_2d[i])
            
            # 3D
            if show_trace:
                lines_3d[i].set_data(path_x, path_y)
                lines_3d[i].set_3d_properties(path_z)
                artists.append(lines_3d[i])
                
            points_3d[i].set_data([curr_x], [curr_y])
            points_3d[i].set_3d_properties([curr_z])
            artists.append(points_3d[i])
            
        return artists

    print("Generowanie animacji 2D + 3D...")
    anim = FuncAnimation(fig, update, frames=steps, interval=80, blit=True)
    plt.close()

    # Tekst
    print(f"Podsumowanie optymalizacji 3D dla podanych scenariuszy:")
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        start_pos = history[0]
        end_pos = history[-1]
        start_cost = f_num((start_pos[0], start_pos[1]))
        end_cost = f_num((end_pos[0], end_pos[1]))
        print(f"[{s.get('name', s['color'])}] Start: (x={start_pos[0]:.4f}, y={start_pos[1]:.4f}), Koszt początkowy: {start_cost:.4f}")
        print(f"        Koniec (po {steps} iteracjach): (x={end_pos[0]:.4f}, y={end_pos[1]:.4f}), Koszt końcowy: {end_cost:.4f}")
    
    df_list = []
    for i, s in enumerate(scenarios):
        history = all_histories[i]
        df_list.append(pd.DataFrame({
            'Scenariusz (Kolor)': [s.get('name', s['color'])]*5,
            'Iteracja': [0, steps//4, steps//2, 3*steps//4, steps],
            'Pozycja x': [history[0, 0], history[steps//4, 0], history[steps//2, 0], history[3*steps//4, 0], history[-1, 0]],
            'Pozycja y': [history[0, 1], history[steps//4, 1], history[steps//2, 1], history[3*steps//4, 1], history[-1, 1]],
            'Koszt f(x,y)': [f_num((history[0, 0], history[0, 1])), f_num((history[steps//4, 0], history[steps//4, 1])), f_num((history[steps//2, 0], history[steps//2, 1])), f_num((history[3*steps//4, 0], history[3*steps//4, 1])), f_num((history[-1, 0], history[-1, 1]))]
        }))
    if df_list:
        print("Tabela przebiegu optymalizacji (kluczowe kroki):")
        display(pd.concat(df_list, ignore_index=True))
    return HTML(anim.to_jshtml())




def make_linear_3d_example(predict_func, grad_funcs, weights, loss_func, learning_rate, iterations=300):
    # Generowanie danych
    X_data = np.linspace(-2, 2, 50)
    noise = np.random.normal(0, 1.0, X_data.shape)
    Y_data = 3 * X_data + 2 + noise
    
    history_weights = [] 
    mses = []
    for i in range(iterations):
        # Zapisz obecny stan wag (kopiujemy słownik)
        history_weights.append(weights.copy())
        
        # Obliczamy gradienty dla całego zbioru danych (średnia)
        current_grads = {k: 0.0 for k in weights}
        
        # Wartości argumentów do funkcji gradientu
        args = (X_data, Y_data, weights['a_1'], weights['a_0'])
        
        mses.append(np.mean(loss_func(args)))
        for key in weights:
            # Obliczamy gradient dla wszystkich punktów naraz (numpy)
            g_values = grad_funcs[key](args)
            # Uśredniamy gradient (Mean Squared Error)
            current_grads[key] = np.mean(g_values)
        
        # Aktualizacja wag (Gradient Descent Step)
        for key in weights:
            weights[key] = weights[key] - learning_rate *current_grads[key]

    # Wizualizacja funkcji kosztu
    a_range = np.linspace(-6, 6, 40)
    b_range = np.linspace(-2, 8, 40)
    A_grid, B_grid = np.meshgrid(a_range, b_range)
    MSE_grid = np.zeros_like(A_grid)

    # MSE dla kazdego punktu
    for i in range(A_grid.shape[0]):
        for j in range(A_grid.shape[1]):
            a_val, b_val = A_grid[i, j], B_grid[i, j]
            y_p = predict_func((X_data, a_val, b_val))
            MSE_grid[i, j] = np.mean((Y_data - y_p)**2)

    # Wizualizacja
    fig = plt.figure(figsize=(14, 6))

    # Dopasowanie linii
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X_data, Y_data, color='blue', alpha=0.5, label='Dane')
    line, = ax1.plot([], [], 'r-', linewidth=3, label='Model (y=ax+b)')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-5, 10)
    ax1.set_title("2D: Dopasowanie prostej")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    
    text_info = ax1.text(0.05, 0.90, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Krajobraz straty
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Powierzchnia
    ax2.plot_surface(A_grid, B_grid, MSE_grid, cmap='viridis', alpha=0.4, edgecolor='none')
    ax2.set_title("3D: Ścieżka gradientu prostego")
    ax2.set_xlabel("Parametr a1 (nachylenie)")
    ax2.set_ylabel("Parametr a0 (wyraz wolny)")
    ax2.set_zlabel("Koszt (MSE)")
    
    # Obecny model
    scatter_3d = ax2.scatter([], [], [], color='red', s=50, label='Obecny stan')
    
    # Trajektoria
    trajectory_line, = ax2.plot([], [], [], color='black', linestyle='--', linewidth=1)

    def update(frame):
        w = history_weights[frame]
        current_mse = mses[frame]
        
        x_plot = np.linspace(-2.5, 2.5, 100)
        y_plot = w['a_1'] * x_plot + w['a_0']
        line.set_data(x_plot, y_plot)
        
        text_info.set_text(f"Iteracja: {frame}\n"
                           f"a1 (nachylenie): {w['a_1']:.2f}\n"
                           f"a0 (wyraz wolny): {w['a_0']:.2f}\n"
                           f"MSE: {current_mse:.2f}")
        
        # Aktualizacja trajektorii
        past_a = [history_weights[k]['a_1'] for k in range(frame+1)]
        past_b = [history_weights[k]['a_0'] for k in range(frame+1)]
        past_mse = mses[:frame+1]
        
        # Aktualizacja obeznej pozycji
        scatter_3d._offsets3d = ([w['a_1']], [w['a_0']], [current_mse])
        
        # Aktualizacja ściezki
        trajectory_line.set_data(past_a, past_b)
        trajectory_line.set_3d_properties(past_mse)
        
        return line, text_info, scatter_3d, trajectory_line

    print("Animacja...")
    anim = FuncAnimation(fig, update, frames=len(history_weights), interval=100, blit=False)
    plt.close()

    # Tekst
    iters = len(history_weights) - 1
    print(f"Podsumowanie optymalizacji regresji liniowej:\n- Start: a_1 = {history_weights[0]['a_1']:.4f}, a_0 = {history_weights[0]['a_0']:.4f}\n- Koniec: a_1 = {history_weights[-1]['a_1']:.4f}, a_0 = {history_weights[-1]['a_0']:.4f}\n- MSE początkowe: {mses[0]:.4f}, MSE końcowe: {mses[-1]:.4f}")
    idx_mid = len(history_weights)//2
    if iters < 1: idx_mid = 0
    df_summary = pd.DataFrame({
        'Iteracja': [0, idx_mid, iters],
        'Waga a_1': [history_weights[0]['a_1'], history_weights[idx_mid]['a_1'], history_weights[-1]['a_1']],
        'Waga a_0': [history_weights[0]['a_0'], history_weights[idx_mid]['a_0'], history_weights[-1]['a_0']],
        'Koszt (MSE)': [mses[0], mses[idx_mid], mses[-1]]
    })
    print("Tabela przebiegu dopasowywania prostej:")
    display(df_summary)
    return HTML(anim.to_jshtml())


def make_polynomial_example(predict_func, grad_funcs, weights, loss_func, learning_rate, iterations=300):
    # Ograniczamy zakres x, bo duże potęgi x rosną bardzo szybko. 
    # Przy dużych x funkcja błędu jest bardzo stroma i GD przy danym η byłby rozbieżny
    X_data = np.linspace(-1.6, 1.6, 50) 
    # Dodajemy szum (np. błędy pomiarowe)
    noise = np.random.normal(0, 0.5, X_data.shape)
    # Prawdziwa funkcja: x^6 - 5x^4 + 7x^2
    Y_data = X_data**6 - 5*X_data**4 + 7*X_data**2 + noise

    history_weights = [] # Tu będziemy zapisywać postępy do animacji
    mses = []
    for i in range(iterations):
        # Zapisz obecny stan wag (kopiujemy słownik)
        history_weights.append(weights.copy())
        
        # Obliczamy gradienty dla CAŁEGO zbioru danych (średnia)
        current_grads = {k: 0.0 for k in weights}
        
        # Sumujemy gradienty ze wszystkich punktów danych
        # Wartości argumentów do funkcji gradientu
        args = (X_data, Y_data, weights['a_6'], weights['a_5'], weights['a_4'],
                weights['a_3'], weights['a_2'], weights['a_1'], weights['a_0'])
        
        mses.append(np.mean(loss_func(args)))
        for key in weights:
            # Obliczamy gradient dla wszystkich punktów naraz (numpy)
            g_values = grad_funcs[key](args)
            # Uśredniamy gradient (Mean Squared Error)
            current_grads[key] = np.mean(g_values)
        
        # Aktualizacja wag (Gradient Descent Step)
        for key in weights:
            weights[key] = weights[key] - learning_rate *current_grads[key]

    # Animacja
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysujemy prawdziwe dane (niebieskie kropki)
    ax.scatter(X_data, Y_data, color='blue', alpha=0.5, label='Pomiary (Dane)')
    ax.set_ylim(-2, 8)
    ax.set_xlim(-1.8, 1.8)
    ax.set_title("Dopasowywanie wielomianu 6. stopnia")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Linia naszego modelu (zacznie się losowo, skończy dopasowana)
    line, = ax.plot([], [], 'r-', linewidth=3, label='Model AI')
    epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    mse_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    equation_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=8, verticalalignment='bottom')

    ax.legend(loc='upper right')

    def update(frame):
        # Pobieramy wagi z danej epoki
        w = history_weights[frame]
        mse = mses[frame]
        # Generujemy linię predykcji dla tych wag
        x_plot = np.linspace(-1.8, 1.8, 200)
        # args: x, a_6, a_5, a_4, a_3, a_2, a_1, a_0
        y_plot = predict_func((x_plot, w['a_6'], w['a_5'], w['a_4'], w['a_3'], w['a_2'], w['a_1'], w['a_0']))
        
        line.set_data(x_plot, y_plot)
        epoch_text.set_text(f'Epoka: {frame}/{iterations}')
        mse_text.set_text(f"MSE: {mse:.3e}")
        
        # Wyświetlanie aktualnego równania (formatowanie)
        eq_str = (f"h(x) = {w['a_6']:.2f}x^6 + {w['a_5']:.2f}x^5 + {w['a_4']:.2f}x^4 + "
                    f"{w['a_3']:.2f}x^3 + {w['a_2']:.2f}x^2 + {w['a_1']:.2f}x + {w['a_0']:.2f}")
        equation_text.set_text(eq_str)
        
        return line, epoch_text, equation_text

    print("Generowanie animacji dopasowywania...")
    anim = FuncAnimation(fig, update, frames=len(history_weights), interval=50, blit=True)
    plt.close()

    # Tekst    
    iters = len(history_weights) - 1
    print(f"Podsumowanie optymalizacji wielomianu:\n- Zmiana MSE: z {mses[0]:.4e} na {mses[-1]:.4e}")
    idx_mid = len(history_weights)//2
    if iters < 1: idx_mid = 0
    df_summary = pd.DataFrame({
        'Iteracja': [0, idx_mid//2, idx_mid, 3*idx_mid//2, iters],
        'MSE': [mses[0], mses[idx_mid//2], mses[idx_mid], mses[3*idx_mid//2], mses[-1]]
    })
    for k in ['a_6', 'a_5', 'a_4', 'a_3', 'a_2', 'a_1', 'a_0']:
        df_summary[f'Waga {k}'] = [history_weights[0][k], history_weights[idx_mid//2][k], history_weights[idx_mid][k], history_weights[3*idx_mid//2][k], history_weights[-1][k]]
    print("Tabela przebiegu dopasowania wielomianu:")
    display(df_summary)
    return HTML(anim.to_jshtml())


def plot_new_home_location(optimal_location, path, attractors, repulsors, radius_meters=1000):
    m = folium.Map(location=optimal_location, zoom_start=13)

    # Rysujemy punkty zdefiniowane (Cele)
    for p in attractors:
        folium.Marker(
            location=p["coords"],
            popup=f"{p['name']} (Waga: {p['weight']})",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
        # Linia od celu do optimum (wizualizacja "napięcia")
        folium.PolyLine(
            locations=[p["coords"], optimal_location],
            color="gray", weight=1, opacity=0.5, dash_array='5, 5'
        ).add_to(m)

    for p in repulsors:
        folium.Marker(
            location=p["coords"],
            popup=f"{p['name']} (Waga: {p['weight']})",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        # Linia od celu do optimum (wizualizacja "napięcia")
        folium.PolyLine(
            locations=[p["coords"], optimal_location],
            color="gray", weight=1, opacity=0.5, dash_array='5, 5'
        ).add_to(m)


    # Rysujemy ścieżkę Gradient Descent
    folium.PolyLine(
        locations=path,
        color="red", weight=3, opacity=0.7,
        tooltip="Ścieżka optymalizacji"
    ).add_to(m)

    folium.Marker(
        location=optimal_location,
        popup="<b>Znaleziona lokalizacja</b>",
        icon=folium.Icon(color="green", icon="home", prefix='fa')
    ).add_to(m)

    folium.Circle(
        location=optimal_location,
        radius=radius_meters,
        color="green",
        fill=True,
        fill_opacity=0.2,
        popup=f"Promień ({radius_meters}m)"
    ).add_to(m)

    return m


def generate_distance_summary(start_pos, final_pos, attractors, repulsors):
    # Funkcja pomocnicza wyciągająca same współrzędne ze słownika
    def extract_coords(point):
        if isinstance(point, dict):
            # Próbujemy najpopularniejszych kluczy dla współrzędnych.
            # Zmień je, jeśli w Twoim notatniku klucz nazywa się inaczej!
            for key in ['coords', 'pos', 'location', 'wspolrzedne']:
                if key in point:
                    return point[key]

            # Alternatywa: jeśli słownik to np. {'x': 10, 'y': 20}
            if 'x' in point and 'y' in point:
                return [point['x'], point['y']]

        # Jeśli to już jest lista/krotka, po prostu ją zwracamy
        return point

    # Odległość w km między punktami [szerokość, długość].
    # Przybliżenie dla małych odległości (w obrębie miasta), tak jak w describe_path_journey:
    # 1° szerokości ≈ 111,2 km, 1° długości ≈ 111,2 km · cos(szerokości).
    def calc_dist_km(p1, p2):
        lat1, lon1 = extract_coords(p1)
        lat2, lon2 = extract_coords(p2)
        mean_lat = np.radians((lat1 + lat2) / 2)
        return np.hypot((lat2 - lat1) * 111.2, (lon2 - lon1) * 111.2 * np.cos(mean_lat))

    data = []
    # Punkty przyciągające: poprawa = o ile się zbliżyliśmy
    for i, attr in enumerate(attractors):
        dist_start = calc_dist_km(start_pos, attr)
        dist_end = calc_dist_km(final_pos, attr)
        name = attr.get('name', f"Punkt {i+1}") if isinstance(attr, dict) else f"Punkt {i+1}"
        data.append(["Atraktor (pożądany)", name, dist_start, dist_end, dist_start - dist_end])

    # Punkty odpychające: poprawa = o ile się oddaliliśmy
    for i, rep in enumerate(repulsors):
        dist_start = calc_dist_km(start_pos, rep)
        dist_end = calc_dist_km(final_pos, rep)
        name = rep.get('name', f"Punkt {i+1}") if isinstance(rep, dict) else f"Punkt {i+1}"
        data.append(["Repulsor (unikany)", name, dist_start, dist_end, dist_end - dist_start])

    df = pd.DataFrame(data, columns=["Typ punktu", "Nazwa", "Dystans początkowy [km]",
                                    "Dystans końcowy [km]", "Poprawa [km]"])

    print("Podsumowanie odległości (start i koniec):")
    # Poprawa > 0 zawsze oznacza zmianę na lepsze: zbliżenie do punktu
    # przyciągającego albo oddalenie od punktu odpychającego.
    display(df.round(2))


def describe_path_journey(path):
    start = path[0]
    end = path[-1]
    total_steps = len(path)
    
    # Całkowity dystans w linii prostej
    total_displacement = np.linalg.norm(np.array(end) - np.array(start))
    
    # Kierunek przesunięcia (w uproszczeniu kartezjańskim)
    delta_lat = end[0] - start[0]
    delta_lon = end[1] - start[1]

    direction_ns = "na północ" if delta_lat > 0 else "na południe"
    direction_ew = "na wschód" if delta_lon > 0 else "na zachód"
    total_steps = len(path) - 1
    dist_km = np.hypot(delta_lat * 111.2, delta_lon * 111.2 * np.cos(np.radians(start[0])))
    
    print("Opis działania algorytmu")
    print(f"1. Algorytm rozpoczął poszukiwania w punkcie: {np.round(start, 4)}.")
    print(f"2. Po {total_steps} krokach dotarł do lokalizacji: {np.round(end, 4)}.")
    print(f"3. Ogólny wektor przesunięcia był skierowany {direction_ns} i {direction_ew}.")
    print(f"4. Całkowite przesunięcie w linii prostej wyniosło ok. {dist_km:.2f} km.")