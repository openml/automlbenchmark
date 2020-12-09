import pytest
from amlb import Resources
from amlb.utils import Namespace


@pytest.fixture
def simple_resource():
    return Resources(
        Namespace(
            input_dir="my_input",
            output_dir="my_output",
            user_dir="my_user_dir",
            root_dir="my_root_dir",
            docker=Namespace(
                image_defaults=Namespace(
                    author="author",
                    image=None,
                    tag=None,
                )
            ),
            frameworks=Namespace(
                root_module="frameworks",
                definition_file=[],
                allow_duplicates=False,
                tags=[]
            )
        )
    )
