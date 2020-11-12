import logging
import click

from src.data.preprocsesor import create_trainable, create_h5_from_openpose, create_3d_from_h5, create_2d_dataset, create_video_id_mapping
from src.data.util import combine_openpose
from src.train.train_transformer import train
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
def data():
    # code to create trainable data
    create_trainable()
    
    
@cli.command()
def combine():
    combine_openpose()


@cli.command()
def create_h5():
    create_h5_from_openpose()

    
@cli.command()
def create_3d():
    create_3d_from_h5()


@cli.command()
def create_2d():
    create_2d_dataset()

    
@cli.command()
def create_video_id_map():
    create_video_id_mapping()
    

@cli.command()
def train_transformer():
    train()
    


if __name__ == '__main__':
    cli()

