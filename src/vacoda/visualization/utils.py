from pathlib import Path

from vacoda.io.paths import get_figure_dir


def resolve_figure_dir(figure_dir=None) -> Path:
    """
    Resolve and create the figure output directory.
    """
    figure_dir = get_figure_dir() if figure_dir is None else figure_dir
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def save_figure(fig, 
                filename: str, 
                figure_dir=None, 
                dpi=300, 
                bbox_inches="tight", 
                close=False):
    """
    Save a matplotlib figure to the configured figure directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename : str
        Output filename, e.g. 'my_plot.png'.
    figure_dir : str or Path, optional
        Directory to save into. Defaults to configured figure directory.
    dpi : int, default=300
        Save resolution.
    bbox_inches : str, default='tight'
        Bounding box option for savefig.
    close : bool, default=False
        Whether to close the figure after saving.

    Returns
    -------
    Path
        Full saved path.
    """
    out_dir = resolve_figure_dir(figure_dir)
    output_path = out_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return output_path
