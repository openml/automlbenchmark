import os
import pytest
from amlb.frameworks.definitions import default_tag, load_framework_definitions

here = os.path.realpath(os.path.dirname(__file__))
res = os.path.join(here, 'resources')

inheritance_def = f"{res}/frameworks_inheritance.yaml"


@pytest.mark.use_disk
def test_version_inheritance(simple_resource):
    definitions_by_tag = load_framework_definitions(inheritance_def, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert len(definitions) == 6

    parent = definitions['framework']
    child1 = definitions['framework_child1']
    grandchild1 = definitions['framework_grandchild1']
    child2 = definitions['framework_child2']
    grandchild2 = definitions['framework_grandchild2']

    assert parent.version == 'latest'
    assert child1.version == 'child1'
    assert grandchild1.version == 'grandchild1'
    assert child2.version == 'child2'
    assert grandchild2.version == 'child2'

    assert parent.abstract
    for f in [child1, child2, grandchild1, grandchild2]:
        assert not f.abstract, f"{f.name} unexpectedly abstract"


@pytest.mark.use_disk
def test_docker_image_inheritance(simple_resource):
    definitions_by_tag = load_framework_definitions(inheritance_def, simple_resource.config)
    assert len(definitions_by_tag) == 1
    definitions = definitions_by_tag[default_tag]
    assert len(definitions) == 6

    parent = definitions['framework']
    child1 = definitions['framework_child1']
    grandchild1 = definitions['framework_grandchild1']
    child2 = definitions['framework_child2']
    grandchild2 = definitions['framework_grandchild2']
    child3 = definitions['framework_child3']

    assert _get_image_desc(parent) == ('author', 'framework', 'latest')
    assert _get_image_desc(child1) == ('author', 'framework', 'child1')
    assert _get_image_desc(grandchild1) == ('author', 'framework', 'grandchild1')
    assert _get_image_desc(child2) == ('author', 'framework', 'child2')
    assert _get_image_desc(grandchild2) == ('author', 'framework', 'child2')
    assert _get_image_desc(child3) == ('author_child3', 'image_child3', 'tag_child3')


def _get_image_desc(definition):
    return definition.image.author, definition.image.image, definition.image.tag
