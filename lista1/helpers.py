# Kod w tym pliku został częściowo przygotowany z pomocą modelu językowego Gemini 3.1 Pro. 

import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from tqdm.notebook import tqdm
from IPython.display import display, HTML, Audio


def load_data(example_name):
    return pd.read_csv(f"data/{example_name}.csv")


def make_manual_experiment(x, y):
    # Wybór kilku punktów do wyświetlenia/odczytania
    sample_indices = [0, len(x)//4, len(x)//2, 3*len(x)//4, len(x)-1]
    
    def generate_audio_wave(y_values, y_min, y_max, duration=2.0, sample_rate=44100):
        if y_max == y_min:
            freqs = np.full(len(y_values), 400)
        else:
            freqs = np.interp(y_values, (y_min, y_max), (150, 1500))
        
        t_audio = np.linspace(0, duration, int(sample_rate * duration), False)
        t_data = np.linspace(0, duration, len(freqs))
        interpolated_freqs = np.interp(t_audio, t_data, freqs)
        
        phase = 2 * np.pi * np.cumsum(interpolated_freqs) / sample_rate
        return np.sin(phase)

    def manual_plot(mode, a4, a3, a2, a1, a0):
        coeffs = [a0, a1, a2, a3, a4]
        y_pred = sum(c * x**i for i, c in enumerate(coeffs))
        
        # Rezidua i MSE
        residuals = y - y_pred
        mse = np.mean(residuals**2)

        # Wybór reprezentacji
        if mode == 'Wizualizacja (Wykres)':
            fig = plt.figure(figsize=(10,6))
            plt.scatter(x=x, y=y, label='Dane zaszumione', c='gray')
            plt.plot(x, y_pred, label='Twój model', c='orange', lw=2)
            fig.legend()
            plt.title(f"Błąd dopasowania (MSE): {mse:.2f}")
            plt.show()
        else:
            # Ekstrema dla poprawnej skali dźwięku
            global_min = min(np.min(y), np.min(y_pred))
            global_max = max(np.max(y), np.max(y_pred))
            
            wave_data = generate_audio_wave(y, global_min, global_max, duration=2.0)
            wave_model = generate_audio_wave(y_pred, global_min, global_max, duration=2.0)
            
            # Sonifikacja błędów
            max_abs_err = np.max(np.abs(residuals))
            if max_abs_err == 0: max_abs_err = 0.1
            wave_residuals = generate_audio_wave(residuals, -max_abs_err, max_abs_err, duration=2.0)
            
            display(HTML("<div aria-live='polite'><h3>Sonifikacja dopasowania wielomianu</h3>"))
            
            display(HTML("<p><b>1. Oryginalne dane zaszumione:</b></p>"))
            display(Audio(wave_data, rate=44100))
            
            display(HTML(f"<p><b>2. Twój model dopasowania:</b></p>"))
            display(Audio(wave_model, rate=44100))
            
            display(HTML(f"<p><b>3. Błędy dopasowania (MSE: {mse:.2f}):</b></p>"))
            display(HTML("<p style='font-size: 14px; color: #555;'><i>Wskazówka: wysokość dźwięku odpowiada wartości błędu w kolejnych punktach (od lewej do prawej).</i></p>"))
            display(Audio(wave_residuals, rate=44100))
            
            display(HTML("</div>"))

        # Generowanie tekstu HTML
        html_content = "<div aria-live='polite' style='margin-top: 20px; font-size: 16px;'>"
        
        # MSE
        html_content += f"<b>Błąd dopasowania: {mse:.2f}</b><br><br>"
        
        html_content += "<b>Aktualne wartości parametrów:</b><ul>"
        for i, c in enumerate(coeffs[::-1]):
            html_content += f"<li>a{len(coeffs) - 1 - i} = {c:.2f}</li>"
        html_content += "</ul><br>"

        html_content += "<b>Przykładowe predykcje:</b><ul>"
        for idx in sample_indices:
            xi = x[idx]
            yi = y[idx]
            yp = y_pred[idx]
            html_content += f"<li>Dla x={xi:.2f} zmierzona wartość y={yi:.2f}. Predykcja to y={yp:.2f}.</li>"
            
        html_content += "</ul></div>"
        
        display(HTML(html_content))

    # Pola tekstowe
    text_inputs = [
        widgets.FloatText(value=0.0, step=0.01, description=f'a{i}:', layout=widgets.Layout(width='95%')) 
        for i in range(5)
    ]
    
    # Przełącznik trybu
    mode_selector = widgets.RadioButtons(
        options=['Wizualizacja (Wykres)', 'Dźwięk (Sonifikacja)'],
        description='Reprezentacja:',
        disabled=False
    )
    
    kwargs = {'mode': mode_selector}
    kwargs.update({f'a{i}': text_inputs[i] for i in range(4, -1, -1)})

    interactive_plot = widgets.interactive(manual_plot, **kwargs)
    
    output = interactive_plot.children[-1]
    output.layout.height = 'auto' 
    
    return interactive_plot


def make_mse_example():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    true_y = 2.1 * x + 0.4
    noise = rng.normal(0, 2, size=x.shape)
    y_noisy = true_y + noise  
    
    sample_indices = [0, len(x)//4, len(x)//2, 3*len(x)//4, len(x)-1]

    def generate_audio_wave(y_values, y_min, y_max, duration=2.0, sample_rate=44100):
        if y_max == y_min:
            freqs = np.full(len(y_values), 400)
        else:
            freqs = np.interp(y_values, (y_min, y_max), (150, 1500))
            
        t_audio = np.linspace(0, duration, int(sample_rate * duration), False)
        t_data = np.linspace(0, duration, len(freqs))
        interpolated_freqs = np.interp(t_audio, t_data, freqs)
        
        phase = 2 * np.pi * np.cumsum(interpolated_freqs) / sample_rate
        return np.sin(phase)

    def plot_mse(mode, slope=2.0, intercept=1.0):
        predicted_y = slope * x + intercept
        residuals = y_noisy - predicted_y
        mse = np.mean(residuals ** 2)
        
        if mode == 'Wizualizacja (Wykres)':
            fig = plt.figure(figsize=(10,6))
            plt.scatter(x=x, y=y_noisy, label='Dane pomiarowe', c='gray')
            plt.plot(x, true_y, label='Rzeczywista funkcja', c='blue', linestyle='dashed')
            plt.plot(x, predicted_y, label='Model użytkownika', c='orange')
            for i, (xi, yi, y_pred) in enumerate(zip(x, y_noisy, predicted_y)):
                plt.plot([xi, xi], [yi, y_pred], color='red', linewidth=1, label='Błędy predykcji' if i==0 else None)
            fig.legend()
            plt.title(f"MSE: {mse:.2f}")
            plt.show()
        else:
            # Globalne ekstrems dla danych i modelu
            global_min = min(np.min(y_noisy), np.min(predicted_y))
            global_max = max(np.max(y_noisy), np.max(predicted_y))
            
            wave_data = generate_audio_wave(y_noisy, global_min, global_max, duration=2.0)
            wave_model = generate_audio_wave(predicted_y, global_min, global_max, duration=2.0)
            
            # Sonifikacja reziduów
            max_abs_err = np.max(np.abs(residuals))

            if max_abs_err == 0: max_abs_err = 0.1 
            wave_residuals = generate_audio_wave(residuals, -max_abs_err, max_abs_err, duration=2.0)
            
            display(HTML("<div aria-live='polite'><h3>Sonifikacja danych</h3>"))
            
            display(HTML("<p><b>1. Zmierzone dane pomiarowe:</b></p>"))
            display(Audio(wave_data, rate=44100))
            
            display(HTML(f"<p><b>2. Twój model predykcji:</b></p>"))
            display(Audio(wave_model, rate=44100))

            display(HTML(f"<p><b>3. Błędy (Rezidua) - MSE wynosi {mse:.2f}:</b></p>"))
            display(HTML("<p style='font-size: 14px; color: #555;'><i>Wskazówka: Jeśli model jest dobrze dopasowany, usłyszysz tylko płaskie 'skakanie' szumu. Jeśli źle - usłyszysz wyraźnie rosnący lub opadający pisk.</i></p>"))
            display(Audio(wave_residuals, rate=44100))
            
            display(HTML("</div>"))

        # Generowanie tekstu HTML
        html_content = "<div aria-live='polite' style='margin-top: 20px; font-size: 16px;'>"
        html_content += "<b>Aktualne parametry i błąd modelu:</b><ul>"
        html_content += f"<li>Nachylenie (slope) = {slope:.2f}</li>"
        html_content += f"<li>Wyraz wolny (intercept) = {intercept:.2f}</li>"
        html_content += f"<li><b>Błąd średniokwadratowy (MSE) = {mse:.2f}</b></li>"
        html_content += "</ul><br>"

        html_content += "<b>Przykładowe predykcje:</b><ul>"
        for idx in sample_indices:
            xi = x[idx]
            yi = y_noisy[idx]
            yp = predicted_y[idx]
            html_content += f"<li>Dla x={xi:.2f} zmierzona wartość y={yi:.2f}. Predykcja to y={yp:.2f}.</li>"
            
        html_content += "</ul></div>"
        display(HTML(html_content))

    mode_selector = widgets.RadioButtons(
        options=['Wizualizacja (Wykres)', 'Dźwięk (Sonifikacja)'],
        description='Reprezentacja:',
        disabled=False
    )
    
    slope_input = widgets.FloatText(value=0.0, step=0.1, description='Nachylenie:', 
                                    layout=widgets.Layout(width='95%'))
    intercept_input = widgets.FloatText(value=-2.0, step=0.1, description='Przecięcie:', 
                                        layout=widgets.Layout(width='95%'))

    interactive_plot = widgets.interactive(plot_mse, mode=mode_selector, slope=slope_input, intercept=intercept_input)
    
    output = interactive_plot.children[-1]
    output.layout.height = 'auto'
    
    return interactive_plot