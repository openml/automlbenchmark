import os
import pytest
from amlb.frameworks.definitions import default_tag, load_framework_definitions

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')

framework_file = f"{res}/frameworks.yaml"


@pytest.mark.use_disk
def test_version_inheritance(simple_resource):
    definitions_by_tag = load_framework_definitions(framework_file, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert len(definitions) == 10

    parent = definitions['unit_test_framework']
    child1 = definitions['unit_test_framework_child1']
    grandchild1 = definitions['unit_test_framework_grandchild1']
    child2 = definitions['unit_test_framework_child2']
    grandchild2 = definitions['unit_test_framework_grandchild2']

    assert parent.version == 'latest'
    assert child1.version == 'child1'
    assert grandchild1.version == 'grandchild1'
    assert child2.version == 'child2'
    assert grandchild2.version == 'child2'


@pytest.mark.use_disk
def test_docker_image_inheritance(simple_resource):
    definitions_by_tag = load_framework_definitions(framework_file, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert len(definitions) == 10

    parent = definitions['unit_test_framework']
    child1 = definitions['unit_test_framework_child1']
    grandchild1 = definitions['unit_test_framework_grandchild1']
    child2 = definitions['unit_test_framework_child2']
    grandchild2 = definitions['unit_test_framework_grandchild2']
    child3 = definitions['unit_test_framework_child3']

    assert _get_image_desc(parent) == ('author', 'unit_test_framework', 'latest')
    assert _get_image_desc(child1) == ('author', 'unit_test_framework', 'child1')
    assert _get_image_desc(grandchild1) == ('author', 'unit_test_framework', 'grandchild1')
    assert _get_image_desc(child2) == ('author', 'unit_test_framework', 'child2')
    assert _get_image_desc(grandchild2) == ('author', 'unit_test_framework', 'child2')
    assert _get_image_desc(child3) == ('author_child3', 'image_child3', 'tag_child3')


def _get_image_desc(definition):
    return definition.image.author, definition.image.image, definition.image.tag
