import logging
import click

from src.data.preprocsesor import create_trainable, create_h5_from_openpose, create_3d_from_h5, create_2d_dataset, create_video_id_mapping, create_word_embedding_map
from src.data.util import combine_openpose, create_training_json
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
    combine_openpose(['h'])


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
def create_training_data_json():
    create_training_json(['h'])
    
    
@cli.command()
def train_transformer():
    train(batch_size=4, epochs=2)
    
    
@cli.command()
def create_embedding_map():
    create_word_embedding_map()
    

if __name__ == '__main__':
    cli()

