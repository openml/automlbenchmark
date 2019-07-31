"""
This script is/was never run. It just creates code snippets used to create the
OpenML Flows associated with the benchmark.
"""
from collections import OrderedDict
import openml

# auto-sklearn==0.5.1 flow created through openml-python scikit-learn extension interface.
# TPOT==0.9.6 flow created through openml-python scikit-learn extension interface.

if __name__ == '__main__':
    autosklearn_flow = openml.flows.get_flow(15275)  # auto-sklearn 0.5.1
    autosklearn_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_autosklearn',
        description=('Auto-sklearn as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        external_version='amlb==0.9',
        parameters=OrderedDict(
            time='240',
            memory='32',
            cores='8'
        ),
        parameters_meta_info=OrderedDict(
            cores=OrderedDict(description='number of available cores', data_type='int'),
            memory=OrderedDict(description='memory in gigabytes', data_type='int'),
            time=OrderedDict(description='time in minutes', data_type='int'),
        ),
        language='English',
        components=OrderedDict(automl_tool=autosklearn_flow),
        tags=['amlb', 'benchmark'],
        dependencies='amlb==0.9',
        model=None
    )
    autosklearn_amlb_flow.publish()
    print(f'autosklearn flow created: {autosklearn_amlb_flow.flow_id}')
    # for dev purposes, since we're rerunning this often, we want to double-check no new flows are created
    assert autosklearn_amlb_flow.flow_id == 15509, "! NEW FLOW CREATED UNEXPECTEDLY!"

    autosklearn_flow = openml.flows.get_flow(15275)  # auto-sklearn 0.5.1
    autosklearn_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_autosklearn',
        description=('Auto-sklearn as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        external_version='amlb==0.9',
        parameters=OrderedDict(
            time='240',
            memory='32',
            cores='8'
        ),
        parameters_meta_info=OrderedDict(
            cores=OrderedDict(description='number of available cores', data_type='int'),
            memory=OrderedDict(description='memory in gigabytes', data_type='int'),
            time=OrderedDict(description='time in minutes', data_type='int'),
        ),
        language='English',
        components=OrderedDict(automl_tool=autosklearn_flow),
        tags=['amlb', 'benchmark'],
        dependencies='amlb==0.9',
        model=None
    )
    autosklearn_amlb_flow.publish()
    print(f'autosklearn flow created: {autosklearn_amlb_flow.flow_id}')
