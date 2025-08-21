#!/usr/bin/env python3

import json
import logging 
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path = 'config.json'):
    logger.info('Loading config file...')
    with open(config_path, 'r') as f:
        return json.load(f)