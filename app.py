import logging
import click


logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
def data():
    # code to create trainable data
    pass
    

if __name__ == '__main__':
    cli()

