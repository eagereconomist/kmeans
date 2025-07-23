import typer
from kmflow.cli import (
    wrangle,
    kmeans,
    pca,
    cluster_prep,
    process,
    plots,
    evaluation,
)

app = typer.Typer()
app.add_typer(wrangle.app, name="wrangle")
app.add_typer(kmeans.app, name="kmeans")
app.add_typer(pca.app, name="pca")
app.add_typer(process.app, name="process")
app.add_typer(plots.app, name="plots")
app.add_typer(evaluation.app, name="evaluate")
app.add_typer(cluster_prep.app, name="cluster-prep")

if __name__ == "__main__":
    app()
