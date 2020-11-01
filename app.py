import logging
import click

from src.data.preprocsesor import create_trainable
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
def data():
    # code to create trainable data
    create_trainable()
    

if __name__ == '__main__':
    cli()

