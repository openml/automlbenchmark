"""
This script is/was never run. It just creates code snippets used to create the
OpenML Flows associated with the benchmark.
"""
from collections import OrderedDict
import openml

# auto-sklearn==0.5.1 flow created through openml-python scikit-learn extension interface.
# TPOT==0.9.6 flow created through openml-python scikit-learn extension interface.

if __name__ == '__main__':
    standard_kwargs = dict(
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
        tags=['amlb', 'benchmark', 'study_218'],
        dependencies='amlb==0.9',
        model=None
    )

    autosklearn_flow = openml.flows.get_flow(15275)  # auto-sklearn 0.5.1
    autosklearn_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_autosklearn',
        description=('Auto-sklearn as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(automl_tool=autosklearn_flow),
        **standard_kwargs
    )
    autosklearn_amlb_flow.publish()
    print(f'autosklearn flow created: {autosklearn_amlb_flow.flow_id}')
    # for dev purposes, since we're rerunning this often, we want to double-check no new flows are created
    assert autosklearn_amlb_flow.flow_id == 15509, "! NEW FLOW CREATED UNEXPECTEDLY!"

    tpot_flow = openml.flows.get_flow(15508)  # TPOT 0.9.6
    tpot_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_tpot',
        description=('TPOT as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(automl_tool=tpot_flow),
        **standard_kwargs
    )
    tpot_amlb_flow.publish()
    print(f'tpot flow created: {tpot_amlb_flow.flow_id}')
    assert tpot_amlb_flow.flow_id == 16114, "! NEW FLOW CREATED UNEXPECTEDLY!"

    h2o_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_h2oautoml',
        description=('H2O AutoML 3.24.0.1 as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(),
        **standard_kwargs
    )
    h2o_amlb_flow.publish()
    print(f'h2o flow created: {h2o_amlb_flow.flow_id}')
    assert h2o_amlb_flow.flow_id == 16115, "! NEW FLOW CREATED UNEXPECTEDLY!"

    autoweka_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_autoweka',
        description=('Auto-WEKA 2.6 as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(),
        **standard_kwargs
    )
    autoweka_amlb_flow.publish()
    print(f'autoweka flow created: {autoweka_amlb_flow.flow_id}')
    assert autoweka_amlb_flow.flow_id == 16116, "! NEW FLOW CREATED UNEXPECTEDLY!"

    rf_flow = openml.flows.get_flow(16117)
    rf_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_randomforest',
        description=('Random Forest baseline as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(randomforest=rf_flow),
        **standard_kwargs
    )
    rf_amlb_flow.publish()
    print(f'rf flow created: {rf_amlb_flow.flow_id}')
    assert rf_amlb_flow.flow_id == 16118, "! NEW FLOW CREATED UNEXPECTEDLY!"

    trf_amlb_flow = openml.flows.OpenMLFlow(
        name='automlbenchmark_tunedrandomforest',
        description=('Tuned Random Forest baseline as set up by the AutoML Benchmark'
                     'Source: source: https://github.com/openml/automlbenchmark/releases/tag/v0.9'),
        components=OrderedDict(randomforest=rf_flow),
        **standard_kwargs
    )
    trf_amlb_flow.publish()
    print(f'trf flow created: {trf_amlb_flow.flow_id}')
    assert trf_amlb_flow.flow_id == 16119, "! NEW FLOW CREATED UNEXPECTEDLY!"

