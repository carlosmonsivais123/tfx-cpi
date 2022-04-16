import absl
import os
import tempfile
import time

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
import tfx

from pprint import pprint
from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2, anomalies_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import Tuner
from tfx.dsl.components.base import executor_spec
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto.evaluator_pb2 import SingleSlicingSpec

from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import HyperParameters
from tfx.types.standard_artifacts import ModelBlessing
from tfx.types.standard_artifacts import InfraBlessing

ARTIFACT_STORE = os.path.join(os.sep, 'Users', 'CarlosMonsivais', 'Desktop', 'CPI Project', 'artifact-store')
SERVING_MODEL_DIR=os.path.join(os.sep, 'Users', 'CarlosMonsivais', 'Desktop', 'CPI Project', 'Model')
DATA_ROOT = 'gs://cpi_bucket/Airline fares'

PIPELINE_NAME = 'pipeline-trial1'
PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, PIPELINE_NAME, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(PIPELINE_ROOT, exist_ok=True)

input = example_gen_pb2.Input(splits=[example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                                      example_gen_pb2.Input.Split(name='eval', pattern='eval/*')])
example_gen = CsvExampleGen(input_base = DATA_ROOT, 
                            input_config = input)
print(example_gen)