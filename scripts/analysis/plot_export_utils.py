from pathlib import Path

import plotly.io as pio
from IPython.display import Image, display


def setup_plotly_renderer(renderer="notebook_connected"):
    # Plotly renderer for local notebook display
    pio.renderers.default = renderer


def save_plotly_for_project(
    fig,
    name,
    figure_output_dir="outputs/figures",
    figure_png_dir="outputs/figures_png",
    width=1500,
    height=900,
    scale=2,
    show_png=True,
    show_html=False,
):

    figure_output_dir = Path(figure_output_dir)
    figure_png_dir = Path(figure_png_dir)

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    figure_png_dir.mkdir(parents=True, exist_ok=True)

    html_path = figure_output_dir / f"{name}.html"
    png_path = figure_png_dir / f"{name}.png"

    fig.write_html(html_path, include_plotlyjs="cdn")

    fig.write_image(png_path, width=width, height=height, scale=scale)

    # print("Saved interactive HTML:", html_path)
    # print("Saved static PNG:", png_path)

    if show_html:
        fig.show()

    if show_png:
        display(Image(filename=str(png_path)))

    return html_path, png_path
