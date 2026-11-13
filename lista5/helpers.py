from IPython.display import display, HTML


def display_accessible_graph(graph):
    html_content = "<div aria-live='polite' style='margin-top: 20px; font-size: 16px;'>"
    html_content += "<h2>Tekstowa reprezentacja Grafu Wiedzy</h2>"
    
    # Tabela krawędzi (Relacji/Faktów)
    html_content += "<h3>Relacje między obiektami</h3>"
    html_content += "<table border='1' style='text-align: left; width: 100%; border-collapse: collapse;'>"
    html_content += "<tr><th scope='col' style='padding: 8px;'>Podmiot</th><th scope='col' style='padding: 8px;'>Relacja</th><th scope='col' style='padding: 8px;'>Obiekt</th></tr>"
    
    for u, v, data in graph.edges(data=True):
        relacja = data.get('relacja', 'BRAK_DANYCH')
        html_content += f"<tr><td style='padding: 8px;'>{u}</td><td style='padding: 8px;'>{relacja}</td><td style='padding: 8px;'>{v}</td></tr>"
    html_content += "</table><br>"

    # Tabela węzłów (Atrybutów)
    html_content += "<h3>Słownik obiektów w grafie</h3>"
    html_content += "<table border='1' style='text-align: left; width: 100%; border-collapse: collapse;'>"
    html_content += "<tr><th scope='col' style='padding: 8px;'>Nazwa obiektu</th><th scope='col' style='padding: 8px;'>Typ obiektu</th></tr>"
    
    for node, data in graph.nodes(data=True):
        typ = data.get('typ', 'Brak przypisanego typu')
        html_content += f"<tr><td style='padding: 8px;'>{node}</td><td style='padding: 8px;'>{typ}</td></tr>"
    html_content += "</table>"
    
    html_content += "</div>"
    display(HTML(html_content))
