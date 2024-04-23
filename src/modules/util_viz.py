#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

def get_correlation_heatmap(df:pd.DataFrame, 
                            title="Correlation Matrix of data",
                            cmap='coolwarm', fmt=".2f", 
                            figsize=(10,8), annot=True):
    # Assuming returns is a DataFrame containing the returns for selected tickers

    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt, ax=ax)
    ax.set_title(title)
    
    # Return the Matplotlib figure object
    return fig


def generate_html_with_figures(figures, titles, output_file='output.html'):
    """
    Generate HTML content with multiple Plotly figures and titles.

    Parameters:
    - figures (list): List of Plotly figures.
    - titles (list): List of titles for each figure.
    - output_file (str): Output file path for saving the HTML content (default: 'output.html').
    """
    # Create HTML content for each figure
        # Add each figure container with its title
    div_figures = []
    for title, figure in zip(titles, figures):
        if isinstance(figure, plt.Figure):  # Check if figure is a Matplotlib figure
            img_file = f'./data/{title}.png'
            figure.savefig(img_file)
            # Encode the image to base64
            with open(img_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            # Create HTML code for the image
            img_html = f'<img src="data:image/png;base64,{img_data}">'
            # Remove the saved image file
            import os
            os.remove(img_file)
            # Add HTML code for the image to the figure container
            div_figure = img_html
        else:  # Assume figure is a Plotly figure
            # Convert Plotly figure to HTML
            div_figure = figure.to_html(full_html=False)
        div_figures.append(div_figure)
    #div_figures = [fig.to_html(full_html=False) for fig in figures]

    # Create HTML layout with titles and scroll bars
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multiple Figures</title>
        <style>
            .figure-container {{
                margin-bottom: 50px;  /* Adjust the spacing between figures */
                overflow-x: auto;  /* Enable horizontal scrolling */
                white-space: nowrap;  /* Keep figures in a single row */
            }}
            .figure-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
    """

    # Add each figure container with its title
    for title, div_figure in zip(titles, div_figures):
        html_content += f"""
        <div class="figure-container">
            <div class="figure-title">{title}</div>
            {div_figure}
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Save the HTML content to a file
    with open(output_file, 'w') as f:
        f.write(html_content)




def display_several_times_series(df:pd.DataFrame, 
                                 title:str="Time Series plot",
                                 xaxis_title:str="Date",
                                 yaxis_title:str="Value"):
    # Assuming close_values_selected is a DataFrame containing the close values for selected tickers
    fig = go.Figure()
    
    for ticker in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=ticker))
    
    fig.update_layout(title=title,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title)
    
    #fig.show()
    return fig